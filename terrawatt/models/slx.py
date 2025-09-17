"""
SLX Model Module
Spatial Lag of X (SLX) Logistic and Probit Regression implementations with spatial cross-validation.
Refactored to properly inherit from SpatialModel base class.
"""

import os
import joblib
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from spreg import Probit
from esda.moran import Moran

from terrawatt.models.terrawatt import SpatialModel, ModelMetrics, ModelDiagnostics

warnings.filterwarnings("ignore")

def pearson_residuals(y, y_hat):
    """Calculate Pearson residuals."""
    return (y - y_hat) / np.sqrt(y_hat * (1 - y_hat))


class SLXLogitModel(SpatialModel):
    """
    Spatial Lag of X Logistic Regression Model.
    
    This implementation includes spatially lagged features (WX) in a logistic
    regression framework with spatial cross-validation.
    """
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """
        Initialize the SLX Logit model.
        
        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the configuration YAML file
        """
        # Initialize parent class first
        super().__init__(config_path)
        
        # Initialize SLX-specific attributes
        self.pipeline: Optional[Pipeline] = None
        self._training_predictions: Optional[pd.DataFrame] = None
        self._coefficients: Optional[pd.DataFrame] = None
    
    def _create_spatial_folds(self, coords: np.ndarray, n_folds: int) -> np.ndarray:
        """Create spatial folds using KMeans clustering."""
        self.logger.info(f"Creating {n_folds} spatial folds using KMeans clustering...")
        
        try:
            # Validate coordinates
            if coords.shape[1] != 2:
                raise ValueError(f"Coordinates must be 2D, got shape: {coords.shape}")
            
            if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
                raise ValueError("Coordinates contain NaN or infinite values")
            
            kmeans = KMeans(n_clusters=n_folds, random_state=42, n_init='auto')
            fold_ids = kmeans.fit_predict(coords)
            
            # Log fold distribution
            fold_counts = np.bincount(fold_ids)
            self.logger.info(f"Fold distribution: {dict(enumerate(fold_counts))}")
            
            min_fold_size = min(fold_counts)
            if min_fold_size < 10:
                self.logger.warning(f"Small fold size detected: {min_fold_size}. Consider reducing n_folds.")
            
            return fold_ids
            
        except Exception as e:
            raise ValueError(f"Error creating spatial folds: {e}")
    
    def _create_pipeline(self) -> Pipeline:
        """Create the scikit-learn pipeline for SLX Logit."""
        feature_cfg = self.config['feature']
        
        pipeline_steps = []
        
        # Add scaler
        scaling_method = feature_cfg.get('scaling', 'standard').lower()
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'none':
            scaler = None
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
        
        if scaler is not None:
            pipeline_steps.append(('scaler', scaler))
            self.logger.info(f"Added {scaler.__class__.__name__} to pipeline")
        
        # Add logistic regression
        logit = LogisticRegression(
            max_iter=2000, 
            class_weight="balanced", 
            solver="lbfgs",
            random_state=42
        )
        pipeline_steps.append(('logit', logit))
        
        self.logger.info("Added LogisticRegression to pipeline")
        
        return Pipeline(pipeline_steps)
    
    def _train(self) -> None:
        """Train the SLX Logit model using spatial cross-validation."""
        if self.train is None:
            self._load_data()
        
        train_cfg = self.config['training']
        model_cfg = self.config['model']
        
        # Create pipeline
        self.pipeline = self._create_pipeline()
        self.model = self.pipeline  # Store reference for parent class compatibility
        
        # Get data from parent class Dataset structure
        X, y = self.train.X, self.train.y
        coords = self.train.df[self.coord_cols].values
        
        # Create spatial folds
        n_folds = train_cfg.get('scv_folds', 5)
        fold_ids = self._create_spatial_folds(coords, n_folds)
        
        # Cross-validation
        records = []
        coef_records = []
        all_residuals = np.zeros(len(self.train.df))
        fold_aucs = []
        
        for fold in range(n_folds):
            self.logger.info(f"Processing fold {fold + 1}/{n_folds}...")
            
            train_idx = np.where(fold_ids != fold)[0]
            test_idx = np.where(fold_ids == fold)[0]
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            self.logger.info(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Fit pipeline for this fold
            try:
                self.pipeline.fit(X_train, y_train)
            except Exception as e:
                self.logger.error(f"Error training fold {fold}: {e}")
                continue
            
            # Make predictions
            try:
                prob = self.pipeline.predict_proba(X_test)[:, 1]
                threshold = model_cfg.get('threshold', 0.5)
                pred = (prob >= threshold).astype(int)
            except Exception as e:
                self.logger.error(f"Error making predictions for fold {fold}: {e}")
                continue
            
            # Calculate AUC
            if len(np.unique(y_test)) > 1:
                auc = average_precision_score(y_test, prob)
                fold_aucs.append(auc)
                self.logger.info(f"  Fold {fold + 1}: AUC={auc:.4f}")
            else:
                self.logger.warning(f"  Fold {fold + 1}: Only one class in test set, skipping AUC calculation")
            
            # Store residuals and predictions
            all_residuals[test_idx] = y_test - prob
            
            for i, idx in enumerate(test_idx):
                records.append({
                    self.id_field: self.train.df.iloc[idx][self.id_field],
                    "fold": fold,
                    "y_true": int(y_test[i]),
                    "y_prob": float(prob[i]),
                    "y_pred": int(pred[i])
                })
            
            # Store coefficients if available
            if hasattr(self.pipeline.named_steps['logit'], 'coef_'):
                coef = self.pipeline.named_steps['logit'].coef_.ravel()
                for feature, c in zip(X.columns, coef):
                    coef_records.append({
                        "fold": fold,
                        "feature": feature,
                        "coef": float(c)
                    })
        
        # Create result DataFrames
        pred_df = pd.DataFrame.from_records(records)
        coef_df = pd.DataFrame.from_records(coef_records)
        
        # Calculate spatial autocorrelation of residuals
        try:
            mi = Moran(all_residuals, self.train.w)
            global_moran_i = float(mi.I)
            global_moran_p = float(mi.p_norm)
            self.logger.info(f"Moran's I on residuals: {global_moran_i:.4f} (p-value: {global_moran_p:.4f})")
            
            # Store diagnostics in parent class format
            self.diagnostics = ModelDiagnostics(
                global_moran_i=global_moran_i,
                global_moran_p=global_moran_p,
                local_moran_i=None,
                local_moran_p=None
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating Moran's I: {e}")
            self.diagnostics = ModelDiagnostics(
                global_moran_i=np.nan,
                global_moran_p=np.nan,
                local_moran_i=None,
                local_moran_p=None
            )
        
        # Train final model on full dataset
        self.logger.info("Training final model on full dataset...")
        try:
            self.pipeline.fit(X, y)
            self._is_fitted = True
            self.logger.info("Final model training complete.")
        except Exception as e:
            raise RuntimeError(f"Error training final model: {e}")
        
        # Store training results
        self._training_predictions = pred_df
        self._coefficients = coef_df
        self._cv_metrics = {
            "mean_cv_auc": np.mean(fold_aucs) if fold_aucs else np.nan,
            "std_cv_auc": np.std(fold_aucs) if fold_aucs else np.nan,
            "n_successful_folds": len(fold_aucs)
        }
    
    def _evaluate(self) -> None:
        """Evaluate model performance using cross-validation results."""
        if not hasattr(self, '_training_predictions') or self._training_predictions.empty:
            raise ValueError("No training predictions available for evaluation")
        
        pred_df = self._training_predictions
        y_true = pred_df["y_true"].values
        y_prob = pred_df["y_prob"].values  
        y_pred = pred_df["y_pred"].values
        
        try:
            # Calculate metrics using parent class structure
            self.metrics = ModelMetrics(
                pr_auc=float(average_precision_score(y_true, y_prob)),
                brier_score=float(brier_score_loss(y_true, y_pred)),
                log_loss=float(log_loss(y_true, y_prob)),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
                confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
                classification_report=classification_report(y_true, y_pred)
            )
            
            self.logger.info("Evaluation Metrics:")
            self.logger.info(f"  PR AUC: {self.metrics.pr_auc:.4f}")
            self.logger.info(f"  Brier Score Loss: {self.metrics.brier_score:.4f}")
            self.logger.info(f"  Precision: {self.metrics.precision:.4f}")
            self.logger.info(f"  Recall: {self.metrics.recall:.4f}")
            self.logger.info(f"  F1-Score: {self.metrics.f1_score:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Error calculating evaluation metrics: {e}")
    
    def predict(self, X_new: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Make predictions on new data."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
        
        if X_new is None:
            X_new = self.test.X.copy()
        
        # Ensure column order matches training data
        feature_cols_training = list(self.train.X.columns)
        missing_cols = set(feature_cols_training) - set(X_new.columns)
        if missing_cols:
            self.logger.warning(f"Missing features in prediction data: {missing_cols}")
            for col in missing_cols:
                X_pred_features[col] = 0
        
        # Reorder columns to match training
        X_pred_features = X_new[feature_cols_training]
        
        # Make predictions
        try:
            self.logger.info("Making predictions...")
            probabilities = self.pipeline.predict_proba(X_pred_features)[:, 1]
            threshold = self.config['model'].get('threshold', 0.5)
            predictions = (probabilities >= threshold).astype(int)
        except Exception as e:
            raise RuntimeError(f"Error making predictions: {e}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'id': self.test.df[self.id_field],
            'probability': probabilities,
            'prediction': predictions
        })
        
        # Add confidence categories
        results['confidence'] = pd.cut(
            results['probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Save predictions using parent class output configuration
        output_cfg = self.config['output']
        output_dir = Path(output_cfg['outdir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / output_cfg.get('predictions', 'predictions.csv')
        results.to_csv(output_path, index=False)
        self.logger.info(f"Predictions saved to {output_path}")
        
        return results
    
    def run(self, X_path: Optional[str] = None, y_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Public method to train the model.
        
        Parameters
        ----------
        X_path : Optional[str]
            Path to features CSV file
        y_path : Optional[str] 
            Path to target CSV file
            
        Returns
        -------
        Dict[str, Any]
            Training results including metrics and diagnostics
        """
        try:
            # Load data using parent class method
            if X_path or y_path:
                # Override config paths if provided
                if X_path:
                    self.config['data']['xtrain_path'] = Path(X_path).name
                    self.config['data']['datadir'] = str(Path(X_path).parent)
                if y_path:
                    self.config['data']['ytrain_path'] = Path(y_path).name
            
            self._load_data()
            
            # Train model
            self._train()
            
            # Evaluate performance
            self._evaluate()
            
            # Generate report
            self._generate_report()
            
            return {
                'predictions': self._training_predictions,
                'coefficients': self._coefficients,
                'metrics': self.metrics,
                'diagnostics': self.diagnostics,
                'cv_metrics': getattr(self, '_cv_metrics', {}),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_report(self) -> str:
        """Generate a comprehensive model performance report."""
        if not self.metrics or not self.diagnostics:
            raise ValueError("No metrics or diagnostics available for report generation")
        
        out_cfg = self.config['output']
        output_dir = Path(out_cfg['outdir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / out_cfg.get('report', 'model_report.txt')
        
        try:
            with open(report_path, "w") as f:
                f.write("=" * 70 + "\n")
                f.write(f"SLX Logistic Regression Model Report\n")
                f.write(f"Model: {self.name}\n")
                f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                
                # Performance Metrics
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"AUC-ROC: {self.metrics.pr_auc:.4f}\n")
                f.write(f"Accuracy: {self.metrics.brier_score:.4f}\n")
                f.write(f"Precision: {self.metrics.precision:.4f}\n")
                f.write(f"Recall: {self.metrics.recall:.4f}\n")
                f.write(f"F1-Score: {self.metrics.f1_score:.4f}\n")
                f.write(f"Log Loss: {self.metrics.log_loss:.4f}\n\n")
                
                # Spatial Diagnostics
                f.write("SPATIAL DIAGNOSTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Moran's I (residuals): {self.diagnostics.global_moran_i:.4f}\n")
                f.write(f"P-value: {self.diagnostics.global_moran_p:.4f}\n")
                interpretation = "Significant" if self.diagnostics.global_moran_p < 0.05 else "Not significant"
                f.write(f"Spatial autocorrelation: {interpretation}\n\n")
                
                # Cross-validation results
                if hasattr(self, '_cv_metrics'):
                    f.write("CROSS-VALIDATION RESULTS\n")
                    f.write("-" * 35 + "\n")
                    cv = self._cv_metrics
                    f.write(f"Mean CV AUC: {cv.get('mean_cv_auc', 'N/A'):.4f}\n")
                    f.write(f"Std CV AUC: {cv.get('std_cv_auc', 'N/A'):.4f}\n")
                    f.write(f"Successful folds: {cv.get('n_successful_folds', 'N/A')}\n\n")
                
                # Feature coefficients
                if hasattr(self, '_coefficients') and not self._coefficients.empty:
                    f.write("AVERAGE FEATURE COEFFICIENTS\n")
                    f.write("-" * 40 + "\n")
                    avg_coefs = (self._coefficients.groupby("feature")["coef"]
                               .mean().sort_values(key=abs, ascending=False))
                    for feature, coef in avg_coefs.items():
                        f.write(f"{feature:25s}: {coef:8.4f}\n")
                    f.write("\n")
                
                f.write("=" * 70 + "\n")
            
            self.logger.info(f"Model report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise IOError(f"Could not generate report: {e}")
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get average coefficients from cross-validation."""
        if not hasattr(self, '_coefficients') or self._coefficients.empty:
            raise ValueError("No coefficients available. Model may not be trained.")
        
        avg_coefs = (self._coefficients.groupby("feature")["coef"]
                    .agg(['mean', 'std', 'count'])
                    .sort_values('mean', key=abs, ascending=False))
        
        return avg_coefs


class SLXProbitModel(SpatialModel):
    """
    Spatial Lag of X Probit Regression Model for inference.
    
    This implementation uses the spreg library for Probit regression with
    spatial lag features and comprehensive diagnostics.
    """
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """
        Initialize the SLX Probit model.
        
        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the configuration YAML file
        """
        # Initialize parent class first
        super().__init__(config_path)
        
        # Initialize SLX Probit-specific attributes
        self.probit_model: Optional[Any] = None
        self.results: Optional[Any] = None
        self.residuals: Optional[np.ndarray] = None
        self.parameters: Optional[np.ndarray] = None
        self._coefficients_df: Optional[pd.DataFrame] = None
    
    def _perform_diagnostics(self) -> Dict[str, Any]:
        """
        Perform comprehensive diagnostic tests on the regression results.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing diagnostic test results
        """
        if self.residuals is None:
            raise ValueError("No residuals available. Model may not be trained.")
        
        if np.isnan(self.residuals).any():
            return {"error": "Residuals contain NaN values, cannot perform diagnostics."}
        
        diagnostics_dict = {}
        
        try:
            self.logger.info("Performing diagnostic tests...")
            
            # Moran's I for spatial autocorrelation of residuals
            moran_test = Moran(self.residuals, self.train.w, permutations=9999)
            diagnostics_dict['morans_i'] = {
                'stat': float(moran_test.I),
                'expected': float(moran_test.EI),
                'p_value': float(moran_test.p_sim)
            }
            
            self.logger.info(f"Observed Moran's I: {moran_test.I:.4f}")
            self.logger.info(f"Expected I under null: {moran_test.EI:.4f}")
            self.logger.info(f"p-value: {moran_test.p_sim:.4f}")
            
            # Interpret the p-value
            if moran_test.p_sim <= 0.05:
                self.logger.info("Result: Significant spatial autocorrelation in residuals (Reject H0).")
            else:
                self.logger.info("Result: No significant spatial autocorrelation (Fail to reject H0).")
            
            # Extract diagnostics from spreg model
            if hasattr(self.probit_model, 'Pinkse_error'):
                diagnostics_dict['Pinkse_error'] = {
                    'stat': float(self.probit_model.Pinkse_error[0]),
                    'p_value': float(self.probit_model.Pinkse_error[1]),
                }
            
            if hasattr(self.probit_model, 'KP_error'):
                diagnostics_dict['KP_error'] = {
                    'stat': float(self.probit_model.KP_error[0]),
                    'p_value': float(self.probit_model.KP_error[1]),
                }
            
            if hasattr(self.probit_model, 'PS_error'):
                diagnostics_dict['PS_error'] = {
                    'stat': float(self.probit_model.PS_error[0]),
                    'p_value': float(self.probit_model.PS_error[1]),
                }
            
            if hasattr(self.probit_model, 'mcfadrho'):
                diagnostics_dict['mcfaddens_rho'] = {
                    'stat': float(self.probit_model.mcfadrho),
                    'p_value': None,
                }
            
            if hasattr(self.probit_model, 'LR'):
                diagnostics_dict['lr_test'] = {
                    'stat': float(self.probit_model.LR[0]),
                    'p_value': float(self.probit_model.LR[1]),
                }
            
            # Store in parent class format
            self.diagnostics = ModelDiagnostics(
                global_moran_i=float(moran_test.I),
                global_moran_p=float(moran_test.p_sim),
                local_moran_i=None,
                local_moran_p=None
            )
            
            # Save diagnostics to file
            output_cfg = self.config['output']
            output_dir = Path(output_cfg['outdir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            diagnostics_df = pd.DataFrame(
                {k: [v['stat'], v.get('p_value', None)] for k, v in diagnostics_dict.items() 
                 if isinstance(v, dict) and 'stat' in v},
                index=['Statistic', 'p-value']
            )
            
            diag_path = output_dir / output_cfg.get('diagnostics', 'diagnostics.csv')
            diagnostics_df.to_csv(diag_path, index=True)
            self.logger.info(f"Diagnostics saved to {diag_path}")
            
            # Create Moran's I distribution plot if possible
            try:
                import matplotlib.pyplot as plt
                if np.all(np.isfinite(moran_test.sim)):
                    plot_path = output_dir / 'moran_distribution.png'
                    plt.figure(figsize=(8, 6))
                    plt.hist(moran_test.sim, bins=50, color='lightblue', alpha=0.7)
                    plt.axvline(moran_test.I, color='red', linestyle='dashed', linewidth=2, 
                               label=f"Observed I = {moran_test.I:.4f}")
                    plt.title("Reference Distribution of Moran's I under Null Hypothesis")
                    plt.xlabel("Moran's I Value")
                    plt.ylabel("Frequency")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()
                    self.logger.info(f"Moran's I distribution plot saved to {plot_path}")
            except ImportError:
                self.logger.warning("Matplotlib not available, skipping Moran's I plot")
            except Exception as e:
                self.logger.warning(f"Could not generate Moran's I plot: {e}")
            
        except Exception as e:
            self.logger.warning(f"Error in diagnostic tests: {e}")
            diagnostics_dict['error'] = str(e)
        
        return diagnostics_dict
    
    def _train(self) -> None:
        """Train the SLX Probit model."""
        if self.train is None:
            self._load_data()
        
        model_cfg = self.config['model']
        feature_cfg = self.config['feature']
        
        # Get data from parent class Dataset structure
        X, y = self.train.raw_X, self.train.y.ravel()  # spreg expects 1D y
        
        # Apply scaling if specified
        if feature_cfg.get('scaling') == 'standard':
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
        elif feature_cfg.get('scaling') == 'minmax':
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), 
                columns=X.columns, 
                index=X.index
            )
        else:
            X_scaled = X.copy()
        
        self.logger.info("Training SLX Probit model using spreg...")
        
        try:
            # Train Probit model with spatial lag features
            lag_order = model_cfg.get('lag_order', 1)
            self.model = Probit(
                y, X_scaled, 
                w=self.train.w, 
                slx_lags=lag_order,
                name_y=self.target_name, 
                name_x=list(X.columns), 
                name_w="spatial_weights", 
                spat_diag=True
            )
            
            # Store results and parameters
            self.results = self.probit_model
            self.residuals = self.model.u_gen.flatten() if hasattr(self.model, 'u_gen') else None
            self.parameters = self.model.betas.flatten() if hasattr(self.model, 'betas') else None
            self._is_fitted = True
            
            self.logger.info("Model training complete.")
            self.logger.info(f"\nModel Summary:\n{self.model.summary}")
            
            # Extract and store coefficients with statistical significance
            if hasattr(self.model, 'betas') and hasattr(self.model, 'z_stat'):
                z_stat, p_val = zip(*self.model.z_stat)
                
                self._coefficients_df = pd.DataFrame({
                    'Feature': self.model.name_x,
                    'Coefficient': self.model.betas.flatten(),
                    'Std_Error': self.model.std_err.flatten() if hasattr(self.model, 'std_err') else np.nan,
                    'z_value': z_stat,
                    'P_value': p_val
                })
                
                # Add significance indicators
                self._coefficients_df['Significant'] = self._coefficients_df['P_value'] < 0.05
                self._coefficients_df['Significance'] = pd.cut(
                    self._coefficients_df['P_value'], 
                    bins=[0, 0.001, 0.01, 0.05, 1], 
                    labels=['***', '**', '*', ''],
                    include_lowest=True
                )
                
                # Save coefficients
                output_cfg = self.config['output']
                output_dir = Path(output_cfg['outdir'])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                coef_path = output_dir / output_cfg.get('coefficients', 'coefficients.csv')
                self._coefficients_df.to_csv(coef_path, index=False)
                self.logger.info(f"Model coefficients saved to {coef_path}")
            
        except Exception as e:
            raise RuntimeError(f"Error training SLX Probit model: {e}")
    
    def _evaluate(self) -> None:
        """Evaluate model performance using diagnostic tests."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet.")
        
        # Perform comprehensive diagnostics
        self._perform_diagnostics()
        
        # For inference models, we don't have traditional prediction metrics
        # Instead, we focus on model fit and diagnostics
        self.logger.info("SLX Probit model evaluation complete.")
        self.logger.info("Check diagnostics for model adequacy and spatial autocorrelation tests.")
    
    def run(self, X_path: Optional[str] = None, y_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Public method to train the model.
        
        Parameters
        ----------
        X_path : Optional[str]
            Path to features CSV file
        y_path : Optional[str] 
            Path to target CSV file
            
        Returns
        -------
        Dict[str, Any]
            Training results including diagnostics and coefficients
        """
        try:
            # Load data using parent class method
            if X_path or y_path:
                # Override config paths if provided
                if X_path:
                    self.config['data']['xtrain_path'] = Path(X_path).name
                    self.config['data']['datadir'] = str(Path(X_path).parent)
                if y_path:
                    self.config['data']['ytrain_path'] = Path(y_path).name
            
            self._load_data()
            
            # Train model
            self._train()
            
            # Evaluate performance (diagnostics)
            self._evaluate()
            
            # Generate report
            self._generate_report()
            
            return {
                'coefficients': self._coefficients_df,
                'diagnostics': self.diagnostics,
                'model_summary': str(self.results.summary) if self.results else None,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_report(self) -> str:
        """Generate a comprehensive model performance report."""
        if not self.diagnostics:
            raise ValueError("No diagnostics available for report generation")
        
        out_cfg = self.config['output']
        output_dir = Path(out_cfg['outdir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / out_cfg.get('report', 'model_report.txt')
        
        try:
            with open(report_path, "w") as f:
                f.write("=" * 70 + "\n")
                f.write(f"SLX Probit Model Report\n")
                f.write(f"Model: {self.name}\n")
                f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                
                # Model Configuration
                f.write("MODEL CONFIGURATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Model Type: {self.kind}\n")
                f.write(f"Features: {len(self.feature_names)}\n")
                f.write(f"Target: {self.target_name}\n\n")
                
                # Data Summary
                f.write("DATA SUMMARY\n")
                f.write("-" * 20 + "\n")
                if self.train:
                    f.write(f"Training samples: {len(self.train.df)}\n")
                    f.write(f"Target distribution: {dict(zip(*np.unique(self.train.y, return_counts=True)))}\n\n")
                
                # Model Summary
                if self.results:
                    f.write("MODEL SUMMARY\n")
                    f.write("-" * 20 + "\n")
                    f.write(str(self.results.summary))
                    f.write("\n\n")
                
                # Coefficients
                if hasattr(self, '_coefficients_df') and self._coefficients_df is not None:
                    f.write("MODEL COEFFICIENTS\n")
                    f.write("-" * 30 + "\n")
                    significant_coefs = self._coefficients_df[self._coefficients_df['Significant']]
                    if not significant_coefs.empty:
                        f.write("Significant coefficients (p < 0.05):\n")
                        for _, row in significant_coefs.iterrows():
                            f.write(f"{row['Feature']:25s}: {row['Coefficient']:8.4f} {row['Significance']}\n")
                    else:
                        f.write("No statistically significant coefficients found.\n")
                    f.write("\n")
                
                # Spatial Diagnostics
                f.write("SPATIAL DIAGNOSTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Moran's I (residuals): {self.diagnostics.global_moran_i:.4f}\n")
                f.write(f"P-value: {self.diagnostics.global_moran_p:.4f}\n")
                interpretation = "Significant" if self.diagnostics.global_moran_p < 0.05 else "Not significant"
                f.write(f"Spatial autocorrelation: {interpretation}\n\n")
                
                # Interpretation
                f.write("MODEL INTERPRETATION\n")
                f.write("-" * 30 + "\n")
                
                if self.diagnostics.global_moran_p < 0.05:
                    f.write("Spatial Residuals: Significant spatial autocorrelation detected.\n")
                    f.write("  Consider additional spatial features or different modeling approach.\n")
                else:
                    f.write("Spatial Residuals: No significant spatial autocorrelation.\n")
                    f.write("  Model adequately captures spatial patterns.\n")
                
                f.write("\n" + "=" * 70 + "\n")
            
            self.logger.info(f"Model report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise IOError(f"Could not generate report: {e}")
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients with significance tests."""
        if not hasattr(self, '_coefficients_df') or self._coefficients_df is None:
            raise ValueError("No coefficients available. Model may not be trained.")
        
        return self._coefficients_df.copy()
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics."""
        if not self.diagnostics:
            raise ValueError("No diagnostics available. Model may not be trained.")
        
        # Return the diagnostics from the comprehensive test
        return self._perform_diagnostics()