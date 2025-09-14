"""
Spatial Random Forest Module
Improved Spatial Random Forest (SRF) classifier implementation with spatial cross-validation.
"""
import yaml
import joblib
import warnings
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

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

from libpysal.weights import Queen, KNN, Rook
from libpysal.weights import lag_spatial
from esda.moran import Moran

import terrawatt.models.terrawatt as twm

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ModelMetrics:
    """Data class to hold model evaluation metrics."""
    overall_auc: float
    overall_accuracy: float
    log_loss: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    classification_report: str


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
                lag_values = lag_spatial(self.spatial_weights, X_copy[col].values)
                X_copy[f"W_{col}"] = lag_values
        
        return X_copy


class SpatialRandomForestModel(twm.PredictionModel):
    """
    Spatial Random Forest (SRF) Classifier Model using scikit-learn Pipeline.
    
    This implementation includes:
    - Better error handling and validation
    - More flexible spatial weights options
    - Enhanced cross-validation strategies
    - Improved feature engineering
    - Better logging and reporting
    """
    
    kind = "prediction"
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """
        Initialize the Enhanced Spatial Random Forest model.
        
        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the configuration YAML file
        """
        super().__init__()
        self.config: Optional[Dict] = None
        self.pipeline: Optional[Pipeline] = None
        self.spatial_weights: Optional[Any] = None
        self.logger: Optional[logging.Logger] = None
        self.X_all: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self.df: Optional[pd.DataFrame] = None
        self.feature_names: Optional[List[str]] = None
        self.metrics: Optional[ModelMetrics] = None
        self._is_fitted = False
        
        # Initialize
        self._load_config(config_path)
        self._setup_logging()
        self._validate_config()
    
    def _load_config(self, config_path: Union[str, Path]) -> None:
        """Load and validate configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def _setup_logging(self) -> None:
        """Set up comprehensive logging configuration."""
        log_config = self.config.get('logging', {})
        log_dir = Path(log_config.get('logdir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = log_dir / log_config.get('logpath', 'srf_model.log')
        log_level = getattr(logging, log_config.get('level', 'INFO').upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Set up logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Enhanced Spatial Random Forest Model initialized")

    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = ['data', 'model', 'training', 'feature', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Check model compatibility
        model_cfg = self.config['model']
        if model_cfg.get('kind') != self.kind:
            self.logger.warning(f"Config model kind '{model_cfg.get('kind')}' differs from class kind '{self.kind}'")
        
        # Validate data configuration
        data_cfg = self.config['data']
        required_data_fields = ['id_field', 'target', 'predictors', 'coord_cols', 'bbox_cols']
        for field in required_data_fields:
            if field not in data_cfg:
                raise ValueError(f"Missing required data configuration field: {field}")
    
    def _validate_columns(self, df: pd.DataFrame, cols: List[str], df_name: str) -> None:
        """Validate that required columns exist in DataFrame."""
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in {df_name}. Available columns: {list(df.columns)}")
    
    def _load_data(self, X_path: Optional[str] = None, y_path: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and prepare the training data with enhanced validation."""
        data_cfg = self.config['data']
        
        # Use provided paths or default from config
        X_path = X_path or Path(data_cfg['datadir']) / data_cfg['xtrain_path']
        y_path = y_path or Path(data_cfg['datadir']) / data_cfg['ytrain_path']
        
        self.logger.info(f"Loading data from X: {X_path}, y: {y_path}")
        
        # Load data with error handling
        try:
            X_df = pd.read_csv(X_path)
            Y_df = pd.read_csv(y_path)
        except Exception as e:
            raise IOError(f"Error loading data files: {e}")
        
        self.logger.info(f"Loaded X: {X_df.shape}, Y: {Y_df.shape}")
        
        # Validate columns
        required_x_cols = [data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors'] + data_cfg['bbox_cols']
        required_y_cols = [data_cfg['id_field'], data_cfg['target']] + data_cfg['bbox_cols']
        
        self._validate_columns(X_df, required_x_cols, "X CSV")
        self._validate_columns(Y_df, required_y_cols, "Y CSV")
        
        # Merge datasets
        self.df = Y_df.merge(
            X_df[[data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors']], 
            on=data_cfg['id_field'], 
            how="inner"
        )
        
        if len(self.df) == 0:
            raise ValueError("Merge resulted in an empty DataFrame. Check ID field consistency.")
        
        self.logger.info(f"Merged dataset shape: {self.df.shape}")
        
        # Create binary target with validation
        target_col = data_cfg['target']
        if self.df[target_col].isna().any():
            self.logger.warning(f"Target column '{target_col}' contains NaN values. These will be dropped.")
            self.df = self.df.dropna(subset=[target_col])
        
        self.y = (self.df[target_col] > 0).astype(int).values
        target_counts = np.bincount(self.y)
        self.logger.info(f"Binary target distribution - 0s: {target_counts[0]}, 1s: {target_counts[1]}")
        
        # Check for class imbalance
        minority_ratio = min(target_counts) / sum(target_counts)
        if minority_ratio < 0.1:
            self.logger.warning(f"Severe class imbalance detected. Minority class ratio: {minority_ratio:.3f}")
        
        return self._create_spatial_features()
    
    def _create_spatial_weights(self, temp_gdf: gpd.GeoDataFrame) -> Any:
        """Create spatial weights matrix with multiple options."""
        feature_cfg = self.config['feature']
        contiguity_type = feature_cfg.get('contiguity', 'queen').lower()
        
        self.logger.info(f"Creating spatial weights using '{contiguity_type}' contiguity...")
        
        if contiguity_type == 'queen':
            weights = Queen.from_dataframe(temp_gdf)
        elif contiguity_type == 'rook':
            weights = Rook.from_dataframe(temp_gdf)
        elif contiguity_type == 'knn':
            centroids = temp_gdf.geometry.centroid
            coords = np.column_stack([centroids.x.values, centroids.y.values])
            k = feature_cfg.get('n_neighbors', 8)
            weights = KNN.from_array(coords, k=k)
        else:
            raise ValueError(f"Unsupported contiguity type: {contiguity_type}")
        
        # Apply row standardization
        weights.transform = "r"
        
        # Log islands (observations with no neighbors)
        if weights.islands:
            self.logger.warning(f"{len(weights.islands)} observations have no neighbors (islands). IDs: {weights.islands}")
        
        return weights
    
    def _create_spatial_features(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Create spatial features including spatially lagged variables with enhanced validation."""
        data_cfg = self.config['data']
        feature_cfg = self.config['feature']
        
        self.logger.info("Building spatial weights matrix...")
        
        # Create polygons from bounding boxes
        bbox_cols = data_cfg['bbox_cols']
        try:
            polygons = []
            for _, row in self.df.iterrows():
                # Assuming bbox_cols = [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = [row[col] for col in bbox_cols]
                polygon = Polygon([(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
                polygons.append(polygon)
            
            temp_gdf = gpd.GeoDataFrame(self.df, geometry=polygons)
        except Exception as e:
            raise ValueError(f"Error creating polygons from bounding box columns: {e}")
        
        # Create spatial weights
        self.spatial_weights = self._create_spatial_weights(temp_gdf)
        
        # Prepare feature matrix
        self.logger.info("Creating feature matrix...")
        X_raw = self.df[data_cfg['predictors']].copy()
        
        # Handle missing values
        imputer_method = feature_cfg.get('imputer', 'none')
        if imputer_method != 'none':
            self.logger.info(f"Applying imputation using method: {imputer_method}")
            for col in X_raw.columns:
                if X_raw[col].isna().any():
                    missing_count = X_raw[col].isna().sum()
                    self.logger.info(f"Imputing {missing_count} missing values in column '{col}'")
                    
                    if imputer_method == 'median':
                        fill_value = X_raw[col].median()
                    elif imputer_method == 'mean':
                        fill_value = X_raw[col].mean()
                    elif imputer_method == 'mode':
                        fill_value = X_raw[col].mode().iloc[0]
                    else:
                        fill_value = 0
                    
                    X_raw[col] = X_raw[col].fillna(fill_value)
        
        # Create spatially lagged features
        self.logger.info("Creating spatially lagged features...")
        try:
            X_lag = pd.DataFrame({
                f"W_{col}": lag_spatial(self.spatial_weights, X_raw[col].values) 
                for col in X_raw.columns
            })
        except Exception as e:
            raise ValueError(f"Error creating spatial lag features: {e}")
        
        # Combine original and lagged features
        self.X_all = pd.concat([X_raw, X_lag], axis=1)
        
        # Add coordinate features if requested
        if feature_cfg.get('include_coords', False):
            coord_cols = data_cfg['coord_cols']
            self.X_all['coord_x'] = self.df[coord_cols[0]]
            self.X_all['coord_y'] = self.df[coord_cols[1]]
            self.logger.info("Added coordinate features")
        
        # Store feature names for later use
        self.feature_names = list(self.X_all.columns)
        
        self.logger.info(f"Final feature matrix shape: {self.X_all.shape}")
        self.logger.info(f"Feature columns: {self.feature_names}")
        
        return self.X_all, self.y
    
    def _create_spatial_folds(self, coords: np.ndarray, n_folds: int) -> np.ndarray:
        """Create spatial folds using KMeans clustering with enhanced validation."""
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
        """Create the scikit-learn pipeline with enhanced configuration."""
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
    
    def _train(self, X: Optional[pd.DataFrame] = None, y: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Train the model using enhanced spatial cross-validation."""
        if X is None or y is None:
            if self.X_all is None or self.y is None:
                raise ValueError("No training data available. Call _load_data() first or provide X and y.")
            X, y = self.X_all, self.y
        
        train_cfg = self.config['training']
        data_cfg = self.config['data']
        model_cfg = self.config['model']
        
        # Create pipeline
        self.pipeline = self._create_pipeline()
        
        # Create spatial folds
        coords = self.df[data_cfg['coord_cols']].values
        n_folds = train_cfg.get('scv_folds', 5)
        fold_ids = self._create_spatial_folds(coords, n_folds)
        
        # Cross-validation
        records = []
        importance_records = []
        all_residuals = np.zeros(len(self.df))
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
                auc = roc_auc_score(y_test, prob)
                fold_aucs.append(auc)
                self.logger.info(f"  Fold {fold + 1}: AUC={auc:.4f}")
            else:
                self.logger.warning(f"  Fold {fold + 1}: Only one class in test set, skipping AUC calculation")
            
            # Store residuals and predictions
            all_residuals[test_idx] = y_test - prob
            
            for i, idx in enumerate(test_idx):
                records.append({
                    data_cfg['id_field']: self.df.iloc[idx][data_cfg['id_field']],
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
        
        # Calculate spatial autocorrelation of residuals
        try:
            mi = Moran(all_residuals, self.spatial_weights)
            morans_i = float(mi.I)
            morans_p = float(mi.p_norm)
            self.logger.info(f"Moran's I on residuals: {morans_i:.4f} (p-value: {morans_p:.4f})")
        except Exception as e:
            self.logger.warning(f"Error calculating Moran's I: {e}")
            morans_i, morans_p = np.nan, np.nan
        
        # Train final model on full dataset
        self.logger.info("Training final pipeline on full dataset...")
        try:
            self.pipeline.fit(X, y)
            self._is_fitted = True
            self.logger.info("Final pipeline training complete.")
        except Exception as e:
            raise RuntimeError(f"Error training final pipeline: {e}")
        
        # Compile diagnostics
        diagnostics = {
            "morans_i": morans_i,
            "morans_i_pvalue": morans_p,
            "residual_spatial_autocorrelation": "Significant" if morans_p < 0.05 else "Not significant",
            "mean_cv_auc": np.mean(fold_aucs) if fold_aucs else np.nan,
            "std_cv_auc": np.std(fold_aucs) if fold_aucs else np.nan,
            "n_successful_folds": len(fold_aucs)
        }
        
        return pred_df, importance_df, diagnostics
    
    def _evaluate(self, pred_df: pd.DataFrame) -> ModelMetrics:
        """Evaluate model performance with comprehensive metrics."""
        if pred_df.empty:
            raise ValueError("Prediction DataFrame is empty")
        
        y_true = pred_df["y_true"].values
        y_prob = pred_df["y_prob"].values  
        y_pred = pred_df["y_pred"].values
        
        try:
            metrics = ModelMetrics(
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
            self.logger.info(f"  AUC: {metrics.overall_auc:.4f}")
            self.logger.info(f"  Accuracy: {metrics.overall_accuracy:.4f}")
            self.logger.info(f"  Precision: {metrics.precision:.4f}")
            self.logger.info(f"  Recall: {metrics.recall:.4f}")
            self.logger.info(f"  F1-Score: {metrics.f1_score:.4f}")
            
            self.metrics = metrics
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Error calculating evaluation metrics: {e}")
    
    def _predict(self, X_new: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Make predictions on new data with enhanced error handling."""
        if not self._is_fitted:
            raise ValueError("Pipeline has not been trained. Call train() first.")
        
        data_cfg = self.config['data']
        
        if X_new is None:
            pred_path = Path(data_cfg['datadir']) / data_cfg['xpred_path']
            try:
                X_new = pd.read_csv(pred_path)
                self.logger.info(f"Loaded prediction data from {pred_path}, shape: {X_new.shape}")
            except Exception as e:
                raise IOError(f"Error loading prediction data: {e}")
        
        # Validate prediction data
        required_cols = ([data_cfg['id_field']] + data_cfg['coord_cols'] + 
                        data_cfg['predictors'] + data_cfg['bbox_cols'])
        self._validate_columns(X_new, required_cols, "prediction data")
        
        # Create spatial features for prediction data (same as training)
        self.logger.info("Creating spatial features for prediction data...")
        X_pred_features = self._create_prediction_features(X_new)
        
        # Ensure column order matches training data
        missing_cols = set(self.feature_names) - set(X_pred_features.columns)
        if missing_cols:
            self.logger.warning(f"Missing features in prediction data: {missing_cols}")
            # Add missing columns with zeros
            for col in missing_cols:
                X_pred_features[col] = 0
        
        # Reorder columns to match training
        X_pred_features = X_pred_features[self.feature_names]
        
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
            'id': X_new[data_cfg['id_field']],
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
        
        # Save predictions
        output_cfg = self.config['output']
        output_dir = Path(output_cfg['outdir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / output_cfg['predictions']
        results.to_csv(output_path, index=False)
        self.logger.info(f"Predictions saved to {output_path}")
        
        return results
    
    def _create_prediction_features(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Create spatial features for prediction data."""
        data_cfg = self.config['data']
        feature_cfg = self.config['feature']
        
        # Create spatial weights for prediction data
        bbox_cols = data_cfg['bbox_cols']
        polygons = []
        for _, row in X_new.iterrows():
            xmin, ymin, xmax, ymax = [row[col] for col in bbox_cols]
            polygon = Polygon([(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
            polygons.append(polygon)
        
        temp_gdf = gpd.GeoDataFrame(X_new, geometry=polygons)
        w_pred = self._create_spatial_weights(temp_gdf)
        
        # Create feature matrix
        X_raw = X_new[data_cfg['predictors']].copy()
        
        # Apply same imputation as training
        imputer_method = feature_cfg.get('imputer', 'none')
        if imputer_method != 'none':
            for col in X_raw.columns:
                if X_raw[col].isna().any():
                    if imputer_method == 'median':
                        fill_value = X_raw[col].median()
                    elif imputer_method == 'mean':
                        fill_value = X_raw[col].mean()
                    else:
                        fill_value = 0
                    X_raw[col] = X_raw[col].fillna(fill_value)
        
        # Create spatial lags
        X_lag = pd.DataFrame({
            f"W_{col}": lag_spatial(w_pred, X_raw[col].values) 
            for col in X_raw.columns
        })
        
        # Combine features
        X_pred_all = pd.concat([X_raw, X_lag], axis=1)
        
        # Add coordinates if used in training
        if feature_cfg.get('include_coords', False):
            coord_cols = data_cfg['coord_cols']
            X_pred_all['coord_x'] = X_new[coord_cols[0]]
            X_pred_all['coord_y'] = X_new[coord_cols[1]]
        
        return X_pred_all
    
    def train(self, X_path: Optional[str] = None, y_path: Optional[str] = None) -> Dict[str, Any]:
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
            # Load data
            X, y = self._load_data(X_path, y_path)
            
            # Train model
            pred_df, importance_df, diagnostics = self._train(X, y)
            
            # Evaluate performance
            metrics = self._evaluate(pred_df)
            
            # Generate report
            self._generate_report(pred_df, importance_df, metrics, diagnostics)
            
            return {
                'predictions': pred_df,
                'feature_importances': importance_df,
                'metrics': metrics,
                'diagnostics': diagnostics,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(self, X_new: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Public method to make predictions.
        
        Parameters
        ----------
        X_new : Optional[pd.DataFrame]
            New data for prediction
            
        Returns
        -------
        pd.DataFrame
            Predictions with probabilities and confidence levels
        """
        return self._predict(X_new)
    
    def save(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save the trained pipeline and associated objects.
        
        Parameters
        ----------
        output_dir : Optional[str]
            Directory to save model artifacts
            
        Returns
        -------
        Dict[str, str]
            Paths to saved artifacts
        """
        if not self._is_fitted:
            raise ValueError("Pipeline has not been trained. Call train() first.")
        
        out_cfg = self.config['output']
        output_dir = Path(output_dir or out_cfg['outdir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        model_path = output_dir / out_cfg['model']
        try:
            joblib.dump(self.pipeline, model_path)
            self.logger.info(f"Pipeline saved to {model_path}")
        except Exception as e:
            raise IOError(f"Error saving pipeline: {e}")
        
        # Save spatial weights
        weights_path = output_dir / out_cfg.get('weights', 'spatial_weights.pkl')
        try:
            joblib.dump(self.spatial_weights, weights_path)
            self.logger.info(f"Spatial weights saved to {weights_path}")
        except Exception as e:
            self.logger.warning(f"Error saving spatial weights: {e}")
        
        # Save feature names
        features_path = output_dir / 'feature_names.pkl'
        try:
            joblib.dump(self.feature_names, features_path)
            self.logger.info(f"Feature names saved to {features_path}")
        except Exception as e:
            self.logger.warning(f"Error saving feature names: {e}")
        
        # Save config
        config_path = output_dir / 'model_config.yaml'
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.warning(f"Error saving configuration: {e}")
        
        return {
            "pipeline": str(model_path),
            "weights": str(weights_path),
            "features": str(features_path),
            "config": str(config_path)
        }
    
    def load(self, model_path: str) -> None:
        """
        Load a previously trained pipeline.
        
        Parameters
        ----------
        model_path : str
            Path to the saved pipeline
        """
        try:
            self.pipeline = joblib.load(model_path)
            self._is_fitted = True
            self.logger.info(f"Pipeline loaded from {model_path}")
        except Exception as e:
            raise IOError(f"Error loading pipeline: {e}")
        
        # Try to load associated files
        model_dir = Path(model_path).parent
        
        # Load feature names if available
        features_path = model_dir / 'feature_names.pkl'
        if features_path.exists():
            try:
                self.feature_names = joblib.load(features_path)
                self.logger.info("Feature names loaded")
            except Exception as e:
                self.logger.warning(f"Could not load feature names: {e}")
        
        # Load spatial weights if available
        weights_path = model_dir / 'spatial_weights.pkl'
        if weights_path.exists():
            try:
                self.spatial_weights = joblib.load(weights_path)
                self.logger.info("Spatial weights loaded")
            except Exception as e:
                self.logger.warning(f"Could not load spatial weights: {e}")
    
    def _generate_report(self, pred_df: pd.DataFrame, importance_df: pd.DataFrame, 
                        metrics: ModelMetrics, diagnostics: Dict[str, Any], 
                        output_dir: Optional[str] = None) -> str:
        """Generate a comprehensive model performance report."""
        out_cfg = self.config['output']
        output_dir = Path(output_dir or out_cfg['outdir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / out_cfg.get('report', 'model_report.txt')
        
        try:
            with open(report_path, "w") as f:
                f.write("=" * 70 + "\n")
                f.write(f"Enhanced Spatial Random Forest Report\n")
                f.write(f"Model: {self.config['model'].get('name', 'Unnamed')}\n")
                f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                
                # Model Configuration
                f.write("MODEL CONFIGURATION\n")
                f.write("-" * 30 + "\n")
                model_cfg = self.config['model']
                f.write(f"Model Type: {model_cfg.get('kind', 'Unknown')}\n")
                f.write(f"Parameters: {model_cfg.get('params', {})}\n")
                f.write(f"Threshold: {model_cfg.get('threshold', 0.5)}\n\n")
                
                # Data Summary
                f.write("DATA SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Training samples: {len(self.df)}\n")
                f.write(f"Features: {len(self.feature_names) if self.feature_names else 'Unknown'}\n")
                f.write(f"Target distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}\n\n")
                
                # Cross-Validation Results
                f.write("CROSS-VALIDATION RESULTS\n")
                f.write("-" * 35 + "\n")
                f.write(f"Mean CV AUC: {diagnostics.get('mean_cv_auc', 'N/A'):.4f}\n")
                f.write(f"Std CV AUC: {diagnostics.get('std_cv_auc', 'N/A'):.4f}\n")
                f.write(f"Successful folds: {diagnostics.get('n_successful_folds', 'N/A')}\n\n")
                
                # Overall Performance Metrics
                f.write("OVERALL PERFORMANCE METRICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"AUC-ROC: {metrics.overall_auc:.4f}\n")
                f.write(f"Accuracy: {metrics.overall_accuracy:.4f}\n")
                f.write(f"Precision: {metrics.precision:.4f}\n")
                f.write(f"Recall: {metrics.recall:.4f}\n")
                f.write(f"F1-Score: {metrics.f1_score:.4f}\n")
                f.write(f"Log Loss: {metrics.log_loss:.4f}\n\n")
                
                # Confusion Matrix
                f.write("CONFUSION MATRIX\n")
                f.write("-" * 25 + "\n")
                cm = np.array(metrics.confusion_matrix)
                f.write(f"              Predicted\n")
                f.write(f"           0      1\n")
                f.write(f"Actual 0  {cm[0,0]:4d}  {cm[0,1]:4d}\n")
                f.write(f"       1  {cm[1,0]:4d}  {cm[1,1]:4d}\n\n")
                
                # Spatial Diagnostics
                f.write("SPATIAL DIAGNOSTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Moran's I on Residuals: {diagnostics.get('morans_i', 'N/A'):.4f}\n")
                f.write(f"P-value: {diagnostics.get('morans_i_pvalue', 'N/A'):.4f}\n")
                f.write(f"Interpretation: {diagnostics.get('residual_spatial_autocorrelation', 'N/A')}\n\n")
                
                # Feature Importances
                if not importance_df.empty:
                    f.write("FEATURE IMPORTANCES (Top 20)\n")
                    f.write("-" * 35 + "\n")
                    avg_importances = (importance_df.groupby("feature")["importance"]
                                     .mean().sort_values(ascending=False).head(20))
                    for feature, importance in avg_importances.items():
                        f.write(f"{feature:25s}: {importance:.4f}\n")
                    f.write("\n")
                
                # Classification Report
                f.write("DETAILED CLASSIFICATION REPORT\n")
                f.write("-" * 45 + "\n")
                f.write(metrics.classification_report)
                f.write("\n")
                
                # Model Interpretation
                f.write("MODEL INTERPRETATION\n")
                f.write("-" * 30 + "\n")
                
                # Performance interpretation
                if metrics.overall_auc >= 0.9:
                    f.write("Model Performance: Excellent (AUC ≥ 0.9)\n")
                elif metrics.overall_auc >= 0.8:
                    f.write("Model Performance: Good (0.8 ≤ AUC < 0.9)\n")
                elif metrics.overall_auc >= 0.7:
                    f.write("Model Performance: Fair (0.7 ≤ AUC < 0.8)\n")
                else:
                    f.write("Model Performance: Poor (AUC < 0.7)\n")
                
                # Spatial autocorrelation interpretation
                morans_p = diagnostics.get('morans_i_pvalue', 1.0)
                if isinstance(morans_p, (int, float)) and morans_p < 0.05:
                    f.write("Spatial Residuals: Significant spatial autocorrelation detected.\n")
                    f.write("  Consider additional spatial features or different modeling approach.\n")
                else:
                    f.write("Spatial Residuals: No significant spatial autocorrelation.\n")
                    f.write("  Model adequately captures spatial patterns.\n")
                
                f.write("\n" + "=" * 70 + "\n")
            
            self.logger.info(f"Comprehensive report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise IOError(f"Could not generate report: {e}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances from the trained model.
        
        Returns
        -------
        pd.DataFrame
            Feature importances sorted by importance
        """
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet.")
        
        if not hasattr(self.pipeline.named_steps['classifier'], 'feature_importances_'):
            raise ValueError("Classifier does not support feature importances.")
        
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def validate_spatial_autocorrelation(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Validate spatial autocorrelation in residuals.
        
        Parameters
        ----------
        residuals : np.ndarray
            Model residuals
            
        Returns
        -------
        Dict[str, float]
            Moran's I statistics
        """
        if self.spatial_weights is None:
            raise ValueError("Spatial weights not available.")
        
        try:
            mi = Moran(residuals, self.spatial_weights)
            return {
                'morans_i': float(mi.I),
                'expected_i': float(mi.EI),
                'variance_i': float(mi.VI_norm),
                'z_score': float(mi.z_norm),
                'p_value': float(mi.p_norm)
            }
        except Exception as e:
            self.logger.error(f"Error calculating spatial autocorrelation: {e}")
            raise
    
    def cross_validate(self, cv_strategy: str = 'spatial', n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation with different strategies.
        
        Parameters
        ----------
        cv_strategy : str
            Cross-validation strategy ('spatial' or 'stratified')
        n_splits : int
            Number of splits
            
        Returns
        -------
        Dict[str, Any]
            Cross-validation results
        """
        if self.X_all is None or self.y is None:
            raise ValueError("No data loaded for cross-validation.")
        
        self.logger.info(f"Performing {cv_strategy} cross-validation with {n_splits} splits")
        
        if cv_strategy == 'spatial':
            # Use existing spatial CV implementation
            coords = self.df[self.config['data']['coord_cols']].values
            fold_ids = self._create_spatial_folds(coords, n_splits)
        elif cv_strategy == 'stratified':
            # Use stratified CV
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_ids = np.zeros(len(self.y))
            for fold, (_, test_idx) in enumerate(skf.split(self.X_all, self.y)):
                fold_ids[test_idx] = fold
        else:
            raise ValueError(f"Unsupported CV strategy: {cv_strategy}")
        
        # Perform cross-validation
        scores = []
        pipeline = self._create_pipeline()
        
        for fold in range(n_splits):
            train_idx = np.where(fold_ids != fold)[0]
            test_idx = np.where(fold_ids == fold)[0]
            
            X_train = self.X_all.iloc[train_idx]
            X_test = self.X_all.iloc[test_idx]
            y_train = self.y[train_idx]
            y_test = self.y[test_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, y_pred_proba)
                scores.append(auc)
        
        return {
            'strategy': cv_strategy,
            'n_splits': n_splits,
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }