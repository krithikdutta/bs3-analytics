"""
Spatial Random Forest Module
Improved Spatial Random Forest (SRF) classifier implementation with spatial cross-validation.
"""
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from esda.moran import Moran, Moran_Local

from terrawatt.models.terrawatt import SpatialModel, ModelMetrics, ModelDiagnostics

warnings.filterwarnings("ignore", category=FutureWarning)


class SpatialFeatureTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for creating spatial lag features."""
    
    def __init__(self, spatial_weights, feature_cols: List[str], include_coords: bool = False):
        """
        Initialize the spatial feature transformer.
        
        Parameters
        ----------
        spatial_weights : libpysal.weights object
            Spatial weights matrix
        feature_cols : List[str]
            List of feature column names to create lags for
        include_coords : bool
            Whether to include coordinate features
        """
        self.spatial_weights = spatial_weights
        self.feature_cols = feature_cols
        self.include_coords = include_coords
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """Transform features by adding spatial lags."""
        X_copy = X.copy()
        
        # Create spatial lag features
        for col in self.feature_cols:
            if col in X_copy.columns:
                lag_values = self.spatial_weights.sparse.dot(X_copy[col].values)
                X_copy[f"W_{col}"] = lag_values
        
        return X_copy


class SpatialRandomForestModel(SpatialModel):
    """
    Spatial Random Forest (SRF) Classifier Model inheriting from SpatialModel.
    """
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """
        Initialize the Enhanced Spatial Random Forest model.
        
        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the configuration YAML file
        """
        # Initialize parent class first
        super().__init__(config_path)
        
        # Initialize SRF-specific attributes
        self.pipeline: Optional[Pipeline] = None
        self._prediction_results: Optional[pd.DataFrame] = None
    
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
            
            # Validate fold distribution
            fold_counts = np.bincount(fold_ids)
            self.logger.info(f"Fold distribution: {dict(enumerate(fold_counts))}")
            
            min_fold_size = min(fold_counts)
            if min_fold_size < 10:
                self.logger.warning(f"Small fold size detected: {min_fold_size}. Consider reducing n_folds.")
            
            return fold_ids
            
        except Exception as e:
            raise ValueError(f"Error creating spatial folds: {e}")
    
    def _create_pipeline(self) -> Pipeline:
        """Create the scikit-learn pipeline."""
        feature_cfg = self.config['feature']
        model_cfg = self.config['model']
        
        pipeline_steps = []
        
        # Add scaler
        scaling_method = feature_cfg.get('scaling', 'standard').lower()
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        elif scaling_method == 'none':
            scaler = None
        else:
            raise ValueError(f"Unsupported scaling method: {scaling_method}")
        
        if scaler is not None:
            pipeline_steps.append(('scaler', scaler))
            self.logger.info(f"Added {scaler.__class__.__name__} to pipeline")
        
        # Add classifier
        rf_params = model_cfg.get('params', {})
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        
        # Merge default and user params
        rf_params = {**default_params, **rf_params}
        
        classifier = RandomForestClassifier(**rf_params)
        pipeline_steps.append(('classifier', classifier))
        
        self.logger.info(f"Added RandomForestClassifier with params: {rf_params}")
        
        return Pipeline(pipeline_steps)
    
    def _train(self) -> None:
        """Train the model using enhanced spatial cross-validation."""
        if self.train is None:
            self._load_data()
        
        train_cfg = self.config['training']
        model_cfg = self.config['model']
        
        # Create pipeline
        self.pipeline = self._create_pipeline()
        self.model = self.pipeline  # Store reference for parent class compatibility
        
        # Create spatial folds
        coords = self.train.df[self.coord_cols].values
        n_folds = train_cfg.get('scv_folds', 5)
        fold_ids = self._create_spatial_folds(coords, n_folds)
        
        # Cross-validation
        records = []
        importance_records = []
        all_residuals = np.zeros(len(self.train.df))
        fold_aucs = []
        
        X, y = self.train.X, self.train.y
        
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
                auc = roc_auc_score(y_test, prob)
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
            
            # Store feature importances
            if hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
                importances = self.pipeline.named_steps['classifier'].feature_importances_
                for feature, imp in zip(X.columns, importances):
                    importance_records.append({
                        "fold": fold,
                        "feature": feature,
                        "importance": float(imp)
                    })
        
        # Create result DataFrames
        pred_df = pd.DataFrame.from_records(records)
        importance_df = pd.DataFrame.from_records(importance_records)
        
        # Calculate spatial autocorrelation of residuals using parent class method
        try:
            mi = Moran(all_residuals, self.train.w)
            global_moran_i = float(mi.I)
            global_moran_p = float(mi.p_norm)
            self.logger.info(f"Moran's I on residuals: {global_moran_i:.4f} (p-value: {global_moran_p:.4f})")
            
            # Store diagnostics in parent class format
            self.diagnostics = ModelDiagnostics(
                global_moran_i=global_moran_i,
                global_moran_p=global_moran_p,
                local_moran_i=None,  # Could be computed if needed
                local_moran_p=None   # Could be computed if needed
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
        self.logger.info("Training final pipeline on full dataset...")
        try:
            self.pipeline.fit(X, y)
            self._is_fitted = True
            self.logger.info("Final pipeline training complete.")
        except Exception as e:
            raise RuntimeError(f"Error training final pipeline: {e}")
        
        # Store training results
        self._training_predictions = pred_df
        self._feature_importances = importance_df
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
                overall_auc=float(roc_auc_score(y_true, y_prob)),
                overall_accuracy=float(accuracy_score(y_true, y_pred)),
                log_loss=float(log_loss(y_true, y_prob)),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
                confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
                classification_report=classification_report(y_true, y_pred)
            )
            
            self.logger.info("Evaluation Metrics:")
            self.logger.info(f"  AUC: {self.metrics.overall_auc:.4f}")
            self.logger.info(f"  Accuracy: {self.metrics.overall_accuracy:.4f}")
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
    
    def _create_prediction_features(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features for prediction data matching training approach."""
        # Create prediction dataset structure similar to training
        pred_df = X_new.copy()
        
        # Convert to binary target (placeholder - not used for prediction)
        pred_df[self.target_name] = 0
        
        # Create features using parent class method
        X_pred, _, w_pred = self._create_features(pred_df)
        
        return X_pred
    
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
                'feature_importances': getattr(self, '_feature_importances', pd.DataFrame()),
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
                f.write(f"Spatial Random Forest Model Report\n")
                f.write(f"Model: {self.name}\n")
                f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                
                # Model Configuration
                f.write("MODEL CONFIGURATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Features: {len(self.feature_names)}\n")
                f.write(f"Target: {self.target_name}\n\n")
                
                # Data Summary
                f.write("DATA SUMMARY\n")
                f.write("-" * 20 + "\n")
                if self.train:
                    f.write(f"Training samples: {len(self.train.df)}\n")
                    f.write(f"Target distribution: {dict(zip(*np.unique(self.train.y, return_counts=True)))}\n\n")
                
                # Performance Metrics
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"AUC-ROC: {self.metrics.overall_auc:.4f}\n")
                f.write(f"Accuracy: {self.metrics.overall_accuracy:.4f}\n")
                f.write(f"Precision: {self.metrics.precision:.4f}\n")
                f.write(f"Recall: {self.metrics.recall:.4f}\n")
                f.write(f"F1-Score: {self.metrics.f1_score:.4f}\n\n")
                
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
                
                f.write("=" * 70 + "\n")
            
            self.logger.info(f"Model report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise IOError(f"Could not generate report: {e}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances from the trained model."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet.")
        
        if not hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            raise ValueError("Classifier does not support feature importances.")
        
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        feature_cols = list(self.train.X.columns) if self.train else self.feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df