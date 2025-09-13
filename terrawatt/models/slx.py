"""
SLX Logit Model Module
Spatial Lag of X (SLX) Logistic Regression implementation with spatial cross-validation.
"""

import os
import yaml
import joblib
import warnings
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from spreg import Probit
from libpysal.weights import Queen, KNN
from libpysal.weights import lag_spatial
from esda.moran import Moran

import terrawatt.models.terrawatt as twm

warnings.filterwarnings("ignore")

class SLXLogitModel(twm.PredictionModel):
    """Spatial Lag of X Logistic Regression Model."""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the SLX Logit model.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration YAML file (default is "config.yaml")
        """
        super().__init__()
        self.config = None
        self.model = None
        self.weights = None
        self.scaler = None
        self.logger = None
        self.X_all = None
        self.y = None
        self.df = None
        self.w = None
        
        # Initialize
        self._load_config(config_path)
        self._setup_logging()
        self._check_compatibility()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.config['logging']['logdir']
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, self.config['logging']['logpath'])
        log_level = getattr(logging, self.config['logging']['level'].upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("SLX Logit Model initialized")

    def _check_compatibility(self):
        """Check if the model kind is compatible."""
        model_cfg = self.config['model']
        if model_cfg['kind'] != self.kind:
            raise ValueError(f"Model kind '{model_cfg['kind']}' is not compatible with SLXLogitModel.")
    
    def _validate_columns(self, df, cols, df_name):
        """
        Validate that required columns exist in DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to validate
        cols : list
            List of required column names
        df_name : str
            Name of the DataFrame for error messages
        
        Raises
        ------
        ValueError
            If any required columns are missing
        """
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in {df_name}.")
    
    def _load_data(self, X_path=None, y_path=None):
        """
        Load and prepare the training data.
        
        Parameters
        ----------
        X_path : str, optional
            Path to features CSV file (uses config if not provided)
        y_path : str, optional
            Path to target CSV file (uses config if not provided)
        
        Returns
        -------
        tuple
            (X, y) prepared feature matrix and target vector
        """
        data_cfg = self.config['data']
        
        # Use provided paths or config paths
        X_path = X_path or os.path.join(data_cfg['datadir'], data_cfg['xtrain_path'])
        y_path = y_path or os.path.join(data_cfg['datadir'], data_cfg['ytrain_path'])
        
        self.logger.info("Loading data...")
        X_df = pd.read_csv(X_path)
        Y_df = pd.read_csv(y_path)
        self.logger.info(f"X data shape: {X_df.shape}, Y data shape: {Y_df.shape}")
        
        # Data Validation
        self._validate_columns(
            X_df, 
            [data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors'] + data_cfg['bbox_cols'], 
            "X CSV"
        )
        self._validate_columns(
            Y_df, 
            [data_cfg['id_field'], data_cfg['target']] + data_cfg['bbox_cols'], 
            "Y CSV"
        )
        
        # Merge data
        self.df = Y_df.merge(
            X_df[[data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors']], 
            on=data_cfg['id_field'], 
            how="inner"
        )
        self.logger.info(f"Merged data shape: {self.df.shape}")
        
        if len(self.df) == 0:
            raise ValueError("Merge resulted in an empty DataFrame. Check ID fields.")
        
        # Create binary target variable
        self.y = (self.df[data_cfg['target']] > 0).astype(int).values.reshape(-1, 1)
        self.logger.info(f"Binary target distribution (0s, 1s): {np.bincount(self.y.ravel())}")
        
        return self._create_features()
    
    def _create_features(self):
        """
        Create spatial features including spatially lagged variables.
        
        Returns
        -------
        tuple
            (X_all, y) feature matrix with spatial lags and target vector
        """
        data_cfg = self.config['data']
        feature_cfg = self.config['feature']
        
        self.logger.info("Building spatial weights matrix...")
        polygons = [Polygon([(r[data_cfg['bbox_cols'][0]], r[data_cfg['bbox_cols'][3]]),
                            (r[data_cfg['bbox_cols'][2]], r[data_cfg['bbox_cols'][3]]),
                            (r[data_cfg['bbox_cols'][2]], r[data_cfg['bbox_cols'][1]]),
                            (r[data_cfg['bbox_cols'][0]], r[data_cfg['bbox_cols'][1]])])
                for _, r in self.df.iterrows()]
        temp_gdf = gpd.GeoDataFrame(self.df, geometry=polygons)
        
        if feature_cfg['contiguity'] == 'queen':
            self.w = Queen.from_dataframe(temp_gdf)
        elif feature_cfg['contiguity'] == 'knn':
            centroids = temp_gdf.geometry.centroid
            coords_knn = np.vstack([centroids.x.values, centroids.y.values]).T
            self.w = KNN.from_array(coords_knn, k=feature_cfg['n_neighbors'])
        else:
            raise ValueError(f"Unsupported contiguity type: {feature_cfg['contiguity']}")
        
        self.w.transform = "r"
        self.logger.info(f"Spatial weights created ({feature_cfg['contiguity']}): {self.w.n} obs, {self.w.s0:.2f} total weights")
        
        if self.w.islands:
            self.logger.warning(f"{len(self.w.islands)} observations have no neighbors (islands).")
        
        # Create feature matrix (X) and spatially lagged features (WX)
        self.logger.info("Creating spatially lagged features...")
        X_raw = self.df[data_cfg['predictors']].copy()
        
        # Impute missing values
        if feature_cfg['imputer'] != 'none':
            for col in X_raw.columns:
                if X_raw[col].isna().any():
                    if feature_cfg['imputer'] == 'median':
                        fill_value = X_raw[col].median()
                    elif feature_cfg['imputer'] == 'mean':
                        fill_value = X_raw[col].mean()
                    X_raw[col] = X_raw[col].fillna(fill_value)
                    self.logger.info(f"Filled {X_raw[col].isna().sum()} NaNs in '{col}' using {feature_cfg['imputer']}.")
        
        # Create spatially lagged features
        X_lag = pd.DataFrame({f"W_{col}": lag_spatial(self.w, X_raw[col].values) for col in X_raw.columns})
        self.X_all = pd.concat([X_raw, X_lag], axis=1)
        self.logger.info(f"Final feature matrix shape: {self.X_all.shape}")
        
        return self.X_all, self.y
    
    def _train(self, X=None, y=None):
        """
        Train the SLX Logit model using spatial cross-validation.
        
        Parameters
        ----------
        X : array-like, optional
            Feature matrix (uses internal data if not provided)
        y : array-like, optional
            Target vector (uses internal data if not provided)
        
        Returns
        -------
        tuple
            (predictions_df, coefficients_df, metrics_dict)
        """
        if X is None or y is None:
            if self.X_all is None or self.y is None:
                raise ValueError("No data available. Call load_data() first or provide X and y.")
            X = self.X_all
            y = self.y
        
        train_cfg = self.config['training']
        model_cfg = self.config['model']
        data_cfg = self.config['data']
        
        # Get coordinates for spatial CV
        coords = self.df[data_cfg['coord_cols']].values
        
        self.logger.info(f"Creating {train_cfg['scv_folds']} spatial folds using KMeans clustering...")
        kmeans = KMeans(n_clusters=train_cfg['scv_folds'], random_state=42, n_init='auto')
        fold_ids = kmeans.fit_predict(coords)
        self.logger.info(f"Fold sizes: {np.bincount(fold_ids)}")
        
        # Define scaling step from config
        feature_cfg = self.config['feature']
        if feature_cfg['scaling'] == 'standard':
            scaler = StandardScaler()
        elif feature_cfg['scaling'] == 'minmax':
            scaler = MinMaxScaler()
        else:  # 'none'
            scaler = None
        
        # Build pipeline
        pipeline_steps = []
        if scaler:
            pipeline_steps.append(("scaler", scaler))
            self.scaler = scaler
        
        pipeline_steps.append(("logit", LogisticRegression(
            max_iter=2000, 
            class_weight="balanced", 
            solver="lbfgs"
        )))
        
        self.model = Pipeline(pipeline_steps)
        
        records, coef_records = [], []
        all_residuals = np.zeros(len(self.df))
        
        for fold in range(train_cfg['scv_folds']):
            self.logger.info(f"Processing fold {fold + 1}/{train_cfg['scv_folds']}...")
            train_idx, test_idx = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx].ravel(), y[test_idx].ravel()
            
            self.model.fit(X_train, y_train)
            prob = self.model.predict_proba(X_test)[:, 1]
            pred = (prob >= model_cfg['threshold']).astype(int)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan
            acc = accuracy_score(y_test, pred)
            self.logger.info(f"  Fold {fold + 1}: AUC={auc:.4f}, Accuracy={acc:.4f}")
            
            # Store results and residuals
            all_residuals[test_idx] = y_test - prob
            for i, idx in enumerate(test_idx):
                records.append({
                    data_cfg['id_field']: self.df.iloc[idx][data_cfg['id_field']], 
                    "fold": fold,
                    "y_true": int(y_test[i]), 
                    "y_prob": float(prob[i]), 
                    "y_pred": int(pred[i])
                })
            
            # Store coefficients
            coef = self.model.named_steps["logit"].coef_.ravel()
            for f, c in zip(X.columns, coef):
                coef_records.append({"fold": fold, "feature": f, "coef": c})
        
        # Convert to DataFrames
        pred_df = pd.DataFrame.from_records(records)
        coef_df = pd.DataFrame.from_records(coef_records)
        
        # Compute spatial autocorrelation of residuals
        self.logger.info("Computing spatial autocorrelation of residuals...")
        mi = Moran(all_residuals, self.w)
        self.logger.info(f"Moran's I on residuals: {mi.I:.4f} (p-value: {mi.p_norm:.4f})")
        
        # Train final model on full dataset
        self.logger.info("Training final model on full dataset...")
        self.model.fit(X, y.ravel())
        
        self.logger.info("Final model training complete.")

        # Prepare diagnostics
        diagnostics = {
            "morans_i": mi.I,
            "morans_i_pvalue": mi.p_norm,
            "residual_spatial_autocorrelation": "Significant" if mi.p_norm < 0.05 else "Not significant"
        }
        
        return pred_df, coef_df, diagnostics
    
    def _evaluate(self, pred_df):
        """
        Evaluate model performance using various metrics.
        
        Parameters
        ----------
        pred_df : pandas.DataFrame
            DataFrame containing true labels and predicted probabilities
        
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        y_true = pred_df["y_true"].values
        y_prob = pred_df["y_prob"].values
        y_pred = pred_df["y_pred"].values
        
        metrics = {
            "overall_auc": roc_auc_score(y_true, y_prob),
            "overall_accuracy": accuracy_score(y_true, y_pred),
            "log_loss": log_loss(y_true, y_prob),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        self.logger.info("Evaluation Metrics:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                self.logger.info(f"  {k}: {v:.4f}")
            else:
                self.logger.info(f"  {k}: {v}")
        
        self.metrics = metrics
        return metrics
    
    def _predict(self, X_new=None):
        """
        Make predictions using the trained model.
        
        Parameters
        ----------
        X_new : pandas.DataFrame
            New data to make predictions on
        
        Returns
        -------
        numpy.ndarray
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        data_cfg = self.config['data']
        feature_cfg = self.config['feature']

        if X_new is None:
            X_new = pd.read_csv(os.path.join(data_cfg['datadir'], data_cfg['xpred_path']))

        self.logger.info(f"Prediction X data shape: {X_new.shape}")
        
        # Data Validation
        self._validate_columns(
            X_new, 
            [data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors'] + data_cfg['bbox_cols'], 
            "X PRED CSV"
        )

        self.logger.info("Building spatial weights matrix for prediction data...")
        polygons = [Polygon([(r[data_cfg['bbox_cols'][0]], r[data_cfg['bbox_cols'][3]]),
                            (r[data_cfg['bbox_cols'][2]], r[data_cfg['bbox_cols'][3]]),
                            (r[data_cfg['bbox_cols'][2]], r[data_cfg['bbox_cols'][1]]),
                            (r[data_cfg['bbox_cols'][0]], r[data_cfg['bbox_cols'][1]])])
                for _, r in X_new.iterrows()]
        temp_gdf = gpd.GeoDataFrame(X_new, geometry=polygons)
        
        if feature_cfg['contiguity'] == 'queen':
            w_pred = Queen.from_dataframe(temp_gdf)
        elif feature_cfg['contiguity'] == 'knn':
            centroids = temp_gdf.geometry.centroid
            coords_knn = np.vstack([centroids.x.values, centroids.y.values]).T
            w_pred = KNN.from_array(coords_knn, k=feature_cfg['n_neighbors'])
        else:
            raise ValueError(f"Unsupported contiguity type: {feature_cfg['contiguity']}")
        
        w_pred.transform = "r"
        self.logger.info(f"Spatial weights created ({feature_cfg['contiguity']}): {w_pred.n} obs, {w_pred.s0:.2f} total weights")
        
        if w_pred.islands:
            self.logger.warning(f"{len(w_pred.islands)} observations have no neighbors (islands).")
        
        # Create feature matrix (X) and spatially lagged features (WX)
        self.logger.info("Creating spatially lagged features for prediction data...")
        X_raw = X_new[data_cfg['predictors']].copy()
        
        # Impute missing values
        if feature_cfg['imputer'] != 'none':
            for col in X_raw.columns:
                if X_raw[col].isna().any():
                    if feature_cfg['imputer'] == 'median':
                        fill_value = X_raw[col].median()
                    elif feature_cfg['imputer'] == 'mean':
                        fill_value = X_raw[col].mean()
                    X_raw[col] = X_raw[col].fillna(fill_value)
                    self.logger.info(f"Filled {X_raw[col].isna().sum()} NaNs in '{col}' using {feature_cfg['imputer']}.")
        
        # Create spatially lagged features
        X_lag = pd.DataFrame({f"W_{col}": lag_spatial(w_pred, X_raw[col].values) for col in X_raw.columns})
        X_new_all = pd.concat([X_raw, X_lag], axis=1)
        # Ensure feature order matches training
        X_pred = X_new_all[self.X_all.columns]
        self.logger.info(f"Final feature matrix shape: {X_pred.shape}")
    
        # Make predictions
        self.logger.info("Making predictions...")
        probabilities = self.model.predict_proba(X_pred)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        # Prepare output
        results = pd.DataFrame({
            data_cfg['id_field']: X_new[data_cfg['id_field']],
            'probability': probabilities,
            'prediction': predictions
        })
        
        # Add confidence categories
        results['confidence'] = pd.cut(
            results['probability'], 
            bins=[0, 0.3, 0.7, 1.0], 
            labels=['Low', 'Medium', 'High']
        )

        OUTPUT_PATH = os.path.join(self.config['output']['outdir'], self.config['output']['predictions'])
        os.makedirs(self.config['output']['outdir'], exist_ok=True)

        # Save results
        results.to_csv(OUTPUT_PATH, index=False)

        # Print summary
        self.logger.info(f"\nPrediction Summary:")
        self.logger.info(f"  Total predictions: {len(results)}")
        self.logger.info(f"  Predicted positive: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        self.logger.info(f"  Mean probability: {probabilities.mean():.3f}")
        self.logger.info(f"  Probability range: {probabilities.min():.3f} - {probabilities.max():.3f}")
        self.logger.info(f"\nConfidence distribution:")
        self.logger.info(results['confidence'].value_counts().to_string())
        self.logger.info(f"\nPredictions saved to: {OUTPUT_PATH}")

        return results
    
    def save(self, output_dir=None):
        """
        Save the model and results to disk.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save outputs (uses config if not provided)
        
        Returns
        -------
        dict
            Paths to saved files
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train() first.")
        
        out_cfg = self.config['output']
        output_dir = output_dir or out_cfg['outdir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and weights
        model_path = os.path.join(output_dir, out_cfg['model'])
        weights_path = os.path.join(output_dir, out_cfg['weights'])
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.w, weights_path)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Spatial weights saved to {weights_path}")
        
        return {
            "model": model_path,
            "weights": weights_path
        }
    
    def _generate_report(self, pred_df, coef_df, metrics, diagnostics, output_dir=None):
        """
        Generate a model performance report.
        
        Parameters
        ----------
        pred_df : pandas.DataFrame
            Predictions DataFrame from training
        coef_df : pandas.DataFrame
            Coefficients DataFrame from training
        metrics : dict
            Model metrics dictionary
        output_dir : str, optional
            Directory to save report (uses config if not provided)
        
        Returns
        -------
        str
            Path to the generated report
        """
        out_cfg = self.config['output']
        output_dir = output_dir or out_cfg['outdir']
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, out_cfg['report'])
        
        with open(report_path, "w") as f:
            f.write(f"Spatial Logit Model Report: {self.config['model']['name']}\n")
            f.write("="*40 + "\n\n")
            f.write(f"Overall AUC: {metrics['overall_auc']:.4f}\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n\n")
            f.write(f"Log Loss: {metrics['log_loss']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
            f.write(f"Confusion Matrix:\n")
            f.write(f"{metrics['confusion_matrix']}\n\n")
            f.write(f"Spatial Autocorrelation of Residuals (Moran's I):\n")
            f.write(f"  I-value: {diagnostics['morans_i']:.4f}\n")
            f.write(f"  p-value: {diagnostics['morans_i_pvalue']:.4f}\n")
            
            if diagnostics['morans_i_pvalue'] < 0.05:
                f.write("  *** Significant spatial clustering detected in residuals. Model may be misspecified.\n")
            else:
                f.write("  No significant spatial autocorrelation in residuals.\n")
            
            avg_coefs = coef_df.groupby("feature")["coef"].mean().sort_values(key=abs, ascending=False)
            f.write("\nAverage Feature Coefficients:\n")
            f.write(avg_coefs.to_string())
        
        # Also save predictions and coefficients
        pred_df.to_csv(os.path.join(output_dir, out_cfg['predictions']), index=False)
        coef_df.to_csv(os.path.join(output_dir, out_cfg['coefficients']), index=False)
        
        self.logger.info(f"Report saved to {report_path}")
        return report_path
    

class SLXModel(twm.InferenceModel):
    """Spatial Lag of X Logistic Regression Model for inference."""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the SLX Logit model.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration YAML file (default is "config.yaml")
        """
        super().__init__()
        self.config = None
        self.model = None
        self.results = None
        self.scaler = None
        self.logger = None
        self.X_all = None
        self.y = None
        self.df = None
        self.w = None
        self.residuals = None
        
        # Initialize
        self._load_config(config_path)
        self._setup_logging()
        self._check_compatibility()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.config['logging']['logdir']
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, self.config['logging']['logpath'])
        log_level = getattr(logging, self.config['logging']['level'].upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("SLX Logit Model initialized")

    def _check_compatibility(self):
        """Check if the model kind is compatible."""
        model_cfg = self.config['model']
        if model_cfg['kind'] != self.kind:
            raise ValueError(f"Model kind '{model_cfg['kind']}' is not compatible with SLXLogitModel.")
    
    def _validate_columns(self, df, cols, df_name):
        """
        Validate that required columns exist in DataFrame.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to validate
        cols : list
            List of required column names
        df_name : str
            Name of the DataFrame for error messages
        
        Raises
        ------
        ValueError
            If any required columns are missing
        """
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in {df_name}.")
    
    def _load_data(self, X_path=None, y_path=None):
        """
        Load and prepare the training data.
        
        Parameters
        ----------
        X_path : str, optional
            Path to features CSV file (uses config if not provided)
        y_path : str, optional
            Path to target CSV file (uses config if not provided)
        
        Returns
        -------
        tuple
            (X, y) prepared feature matrix and target vector
        """
        data_cfg = self.config['data']
        
        # Use provided paths or config paths
        X_path = X_path or os.path.join(data_cfg['datadir'], data_cfg['xtrain_path'])
        y_path = y_path or os.path.join(data_cfg['datadir'], data_cfg['ytrain_path'])
        
        self.logger.info("Loading data...")
        X_df = pd.read_csv(X_path)
        Y_df = pd.read_csv(y_path)
        self.logger.info(f"X data shape: {X_df.shape}, Y data shape: {Y_df.shape}")
        
        # Data Validation
        self._validate_columns(
            X_df, 
            [data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors'] + data_cfg['bbox_cols'], 
            "X CSV"
        )
        self._validate_columns(
            Y_df, 
            [data_cfg['id_field'], data_cfg['target']] + data_cfg['bbox_cols'], 
            "Y CSV"
        )
        
        # Merge data
        self.df = Y_df.merge(
            X_df[[data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors']], 
            on=data_cfg['id_field'], 
            how="inner"
        )
        self.logger.info(f"Merged data shape: {self.df.shape}")
        
        if len(self.df) == 0:
            raise ValueError("Merge resulted in an empty DataFrame. Check ID fields.")
        
        # Create binary target variable
        self.y = (self.df[data_cfg['target']] > 0).astype(int).values.reshape(-1, 1)
        self.logger.info(f"Binary target distribution (0s, 1s): {np.bincount(self.y.ravel())}")
        
        return self._create_features()
    
    def _create_features(self):
        """
        Create spatial features including spatially lagged variables.
        
        Returns
        -------
        tuple
            (X_all, y) feature matrix with spatial lags and target vector
        """
        data_cfg = self.config['data']
        feature_cfg = self.config['feature']
        
        self.logger.info("Building spatial weights matrix...")
        polygons = [Polygon([(r[data_cfg['bbox_cols'][0]], r[data_cfg['bbox_cols'][3]]),
                            (r[data_cfg['bbox_cols'][2]], r[data_cfg['bbox_cols'][3]]),
                            (r[data_cfg['bbox_cols'][2]], r[data_cfg['bbox_cols'][1]]),
                            (r[data_cfg['bbox_cols'][0]], r[data_cfg['bbox_cols'][1]])])
                for _, r in self.df.iterrows()]
        temp_gdf = gpd.GeoDataFrame(self.df, geometry=polygons)
        
        if feature_cfg['contiguity'] == 'queen':
            self.w = Queen.from_dataframe(temp_gdf)
        elif feature_cfg['contiguity'] == 'knn':
            centroids = temp_gdf.geometry.centroid
            coords_knn = np.vstack([centroids.x.values, centroids.y.values]).T
            self.w = KNN.from_array(coords_knn, k=feature_cfg['n_neighbors'])
        else:
            raise ValueError(f"Unsupported contiguity type: {feature_cfg['contiguity']}")
        
        self.w.transform = "r"
        self.logger.info(f"Spatial weights created ({feature_cfg['contiguity']}): {self.w.n} obs, {self.w.s0:.2f} total weights")
        
        if self.w.islands:
            self.logger.warning(f"{len(self.w.islands)} observations have no neighbors (islands).")
        
        # Create feature matrix (X) and spatially lagged features (WX)
        self.logger.info("Creating spatially lagged features...")
        X_raw = self.df[data_cfg['predictors']].copy()
        
        # Impute missing values
        if feature_cfg['imputer'] != 'none':
            for col in X_raw.columns:
                if X_raw[col].isna().any():
                    if feature_cfg['imputer'] == 'median':
                        fill_value = X_raw[col].median()
                    elif feature_cfg['imputer'] == 'mean':
                        fill_value = X_raw[col].mean()
                    X_raw[col] = X_raw[col].fillna(fill_value)
                    self.logger.info(f"Filled {X_raw[col].isna().sum()} NaNs in '{col}' using {feature_cfg['imputer']}.")
        
        # Create spatially lagged features
        # X_lag = pd.DataFrame({f"W_{col}": lag_spatial(self.w, X_raw[col].values) for col in X_raw.columns})
        self.X_all = X_raw  # No lagged features for inference
        self.logger.info(f"Final feature matrix shape: {self.X_all.shape}")
        
        return self.X_all, self.y
    
    def _perform_diagnostics(self):
        """
        Perform diagnostic tests on the regression results.
            
        Returns
        -------
        dict
            Dictionary containing diagnostic test results
        """
        out_cfg = self.config['output']
        self.diagnostics = {}
        if np.isnan(self.residuals).any():
            return {"error": "Residuals contain NaN values, cannot perform diagnostics."}
        
        try:
            self.logger.info("Performing diagnostic tests...")

            # Moran's I for spatial autocorrelation of residuals (Spatial CV)
            moran_test = Moran(self.residuals, self.w, permutations=9999)
            self.diagnostics['morans_i'] = {
                'stat': moran_test.I,
                'p_value': moran_test.p_sim
            }
            self.logger.info(f"Observed Moran's I: {moran_test.I}")
            self.logger.info(f"Expected I under null: {moran_test.EI}")
            self.logger.info(f"p-value: {moran_test.p_sim}")

            # Interpret the p-value
            if moran_test.p_sim <= 0.05:
                self.logger.info("Result: Significant spatial autocorrelation in the residuals (Reject H0).")
            else:
                self.logger.info("Result: No significant spatial autocorrelation (Fail to reject H0).")

            # You can also access the full distribution of permuted I values
            import matplotlib.pyplot as plt
            plot_path = os.path.join(out_cfg['outdir'], out_cfg['plots'], "moran_distribution.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            # Check if the simulation results are all valid numbers before plotting
            if np.all(np.isfinite(moran_test.sim)):
                plt.hist(moran_test.sim, bins=50, color='lightblue')
                plt.axvline(moran_test.I, color='red', linestyle='dashed')
                plt.title("Reference Distribution of Moran's I under Null Hypothesis")
                plt.xlabel("Moran's I Value")
                plt.savefig(plot_path)
                plt.close()
                self.logger.info(f"Moran's I distribution plot saved to {plot_path}")
            else:
                # If there are bad values, log an error and skip the plot
                self.logger.error("Could not generate Moran's I plot because simulation results contain non-finite values.")

            self.logger.info(self.model.summary)
            self.diagnostics['Pinkse_error'] = {
                'stat': self.model.Pinkse_error[0],
                'p_value': self.model.Pinkse_error[1],
            }
            self.diagnostics['KP_error'] = {
                'stat': self.model.KP_error[0],
                'p_value': self.model.KP_error[1],
            }
            self.diagnostics['PS_error'] = {
                'stat': self.model.PS_error[0],
                'p_value': self.model.PS_error[1],
            }
            self.diagnostics['mcfaddens_rho'] = {
                'stat': self.model.mcfadrho,
                'p_value': None,
            }
            self.diagnostics['lr_test'] = {
                'stat': self.model.LR[0],
                'p_value': self.model.LR[1],
            }

            self.logger.info("Diagnostic test results:")
            for test, result in self.diagnostics.items():
                self.logger.info(f"  {test}: Statistic={result['stat']}, p-value={result['p_value']}")

            # diagnostics_df = pd.DataFrame(
            #     {k: [v['stat'], v['p_value']] for k, v in self.diagnostics.items()},
            #     index=['Statistic', 'p-value']
            # )
            # diag_path = os.path.join(out_cfg['outdir'], out_cfg['diagnostics'])
            # os.makedirs(os.path.dirname(diag_path), exist_ok=True)
            # diagnostics_df.to_csv(diag_path, index=True)
            # self.logger.info(f"Diagnostics saved to {diag_path}")

        except Exception as e:
            self.logger.warning(f"Error in diagnostic tests: {e}")
            self.diagnostics['error'] = str(e)
        
        return self.diagnostics
    
    def _train(self, X=None, y=None):
        """
        Train the SLX Logit model using spatial cross-validation.
        
        Parameters
        ----------
        X : array-like, optional
            Feature matrix (uses internal data if not provided)
        y : array-like, optional
            Target vector (uses internal data if not provided)
        
        Returns
        -------
        tuple
            (predictions_df, coefficients_df, metrics_dict)
        """
        if X is None or y is None:
            if self.X_all is None or self.y is None:
                raise ValueError("No data available. Call load_data() first or provide X and y.")
            X = self.X_all
            y = self.y
        
        train_cfg = self.config['training']
        model_cfg = self.config['model']
        data_cfg = self.config['data']
        out_cfg = self.config['output']
        
        if train_cfg['scv_method'] == 'kfold':
            # Get coordinates for spatial CV
            coords = self.df[data_cfg['coord_cols']].values
            
            self.logger.info(f"Creating {train_cfg['scv_folds']} spatial folds using KMeans clustering...")
            kmeans = KMeans(n_clusters=train_cfg['scv_folds'], random_state=42, n_init='auto')
            fold_ids = kmeans.fit_predict(coords)
            self.logger.info(f"Fold sizes: {np.bincount(fold_ids)}")
        
        # Define scaling step from config
        feature_cfg = self.config['feature']
        if feature_cfg['scaling'] == 'standard':
            scaler = StandardScaler()
        elif feature_cfg['scaling'] == 'minmax':
            scaler = MinMaxScaler()
        else:  # 'none'
            scaler = None

        # Apply scaling outside pipeline
        if scaler:
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        else:
            X_scaled = X.copy()

        self.logger.info("Training SLX Probit model using spreg...")

        self.model = Probit(y, X_scaled, w=self.w, slx_lags=model_cfg['lag_order'], name_y="BSS", name_x=list(X.columns), name_w="queen", spat_diag=True)
        self.results = self.model.fit
        self.residuals = self.model.u_gen
        z_stat, p_val = zip(*self.model.z_stat)
        
        self.logger.info("Model training complete.")
        self.logger.info(f"\nModel Summary:\n{self.results}")

        # self.logger.info(len(self.model.betas.flatten()))
        # self.logger.info(len(self.model.std_err.flatten()))
        # self.logger.info(len(self.model.name_x))
        
        # Extract coefficients with statistical significance
        coefficients_df = pd.DataFrame({
            'Feature': self.model.name_x,
            'Coefficient': self.model.betas.flatten(),
            'Std_Error': self.model.std_err.flatten(),
            'z_value': z_stat,
            'P_value': p_val
        })
        
        # Add significance indicators
        coefficients_df['Significant'] = coefficients_df['P_value'] < 0.05
        coefficients_df['Significance'] = pd.cut(
            coefficients_df['P_value'], 
            bins=[0, 0.001, 0.01, 0.05, 1], 
            labels=['***', '**', '*', ''],
            include_lowest=True
        )
        
        # Save coefficients
        self.parameters = self.model.betas
        coef_path = os.path.join(out_cfg['outdir'], out_cfg['coefficients'])
        os.makedirs(out_cfg['outdir'], exist_ok=True)
        coefficients_df.to_csv(coef_path, index=False)
        self.logger.info(f"Model coefficients saved to {coef_path}")
        
        return self.results