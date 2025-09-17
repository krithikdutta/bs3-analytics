
import yaml
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

from libpysal.weights import Queen, KNN, Rook
from libpysal.weights import lag_spatial

@dataclass
class Dataset:
    """Data class to hold dataset information."""
    X: pd.DataFrame
    w: Any
    df: pd.DataFrame
    y: Optional[np.ndarray] = None
    raw_X: Optional[pd.DataFrame] = None
    lag_X: Optional[pd.DataFrame] = None

@dataclass
class ModelMetrics:
    """Data class to hold model evaluation metrics."""
    pr_auc: float
    brier_score: float
    log_loss: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    classification_report: str

@dataclass
class ModelDiagnostics:
    """Data class to hold model diagnostics."""
    global_moran_i: float
    global_moran_p: float
    local_moran_i: Optional[np.ndarray]
    local_moran_p: Optional[np.ndarray]

class SpatialModel:
    """
    A base class for implementing spatial models.

    This class provides a framework for building and evaluating inference models.
    It defines common attributes and methods that can be extended by specific
    model implementations.
    """
    def __init__(self,  config_path: Union[str, Path] = "config.yaml"):
        """
        Initializes the InferenceModel with placeholders for parameters, metrics, and diagnostics.
        """
        self.config: Optional[Dict] = None
        self.logger: Optional[logging.Logger] = None
        self.name: Optional[str] = None

        self.feature_names: Optional[List[str]] = None
        self.target_name: Optional[str] = None
        self.bbox_cols: Optional[List[str]] = None
        self.coord_cols: Optional[List[str]] = None
        self.id_field: Optional[str] = None

        self.train: Optional[Dataset] = None
        self.test: Optional[Dataset] = None
        self.validation: Optional[Dataset] = None
        
        self.model: Optional[Any] = None
        self._is_fitted = False

        self.metrics: Optional[ModelMetrics] = None
        self.diagnostics: Optional[ModelDiagnostics] = None
        
        # Initialize
        self._load_config(config_path)
        self._validate_config()
        self._setup_logging()
        self._finish_setup()

    def __del__(self):
        """Destructor to clean up resources."""
        self._close_logging()

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
        
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = ['data', 'model', 'training', 'feature', 'output']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_cfg = self.config['data']
        required_data_fields = ['id_field', 'target', 'predictors', 'coord_cols', 'bbox_cols']
        for field in required_data_fields:
            if field not in data_cfg:
                raise ValueError(f"Missing required data configuration field: {field}")
    
    def _setup_logging(self) -> None:
        """Set up comprehensive logging configuration."""
        log_config = self.config.get('logging', {})
        log_dir = Path(log_config.get('logdir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = log_dir / log_config.get('logpath', 'training.log')
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
        
        self.logger.info("--- Logging initialized ---")

    def _close_logging(self) -> None:
        """Close all logging handlers to release resources."""
        if hasattr(self, 'logger') and self.logger:
            self.logger.info("--- Closing logging ---")
            # Get all handlers associated with the logger
            handlers = self.logger.handlers[:]
            # Loop through and close each handler
            for handler in handlers:
                # Flushes and closes the handler
                handler.close()
                # Removes the handler from the logger
                self.logger.removeHandler(handler)
            
    def _finish_setup(self) -> None:
        """Finalize setup by extracting configuration details."""
        model_cfg = self.config['model']
        data_cfg = self.config['data']

        self.name = model_cfg.get('name', 'Inference Model')
        self.feature_names = data_cfg.get('predictors', [])
        self.target_name = data_cfg.get('target')
        self.bbox_cols = data_cfg.get('bbox_cols', [])
        self.coord_cols = data_cfg.get('coord_cols', [])
        self.id_field = data_cfg.get('id_field')
        
        if not self.feature_names or not self.target_name or not self.coord_cols or not self.bbox_cols or not self.id_field:
            raise ValueError("Feature names, target name, coordinate columns, bounding box columns, and ID field must be specified in the configuration.")
        
        self.logger.info(f"Model: {self.name} initialized")
    
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
        required_x_cols = [self.id_field] + self.coord_cols + self.feature_names + self.bbox_cols
        required_y_cols = [self.id_field, self.target_name] + self.bbox_cols
        
        self._validate_columns(X_df, required_x_cols, "X CSV")
        self._validate_columns(Y_df, required_y_cols, "Y CSV")
        
        # Merge datasets
        dframe = Y_df.merge(
            X_df[[self.id_field] + self.coord_cols + self.feature_names], 
            on=self.id_field, 
            how="inner"
        )
        
        if len(dframe) == 0:
            raise ValueError("Merge resulted in an empty DataFrame. Check ID field consistency.")
        
        self.logger.info(f"Merged dataset shape: {dframe.shape}")

        # Check for NaNs in target
        if dframe[self.target_name].isna().any():
            self.logger.warning(f"Target column '{self.target_name}' contains NaN values. These will be dropped.")
            dframe = dframe.dropna(subset=[self.target_name])
        
        # Create features and target
        _X, _y, _w, X_r, X_l = self._create_features(dframe)
        self.train = Dataset(X=_X, y=_y, w=_w, df=dframe, raw_X=X_r, lag_X=X_l)

    def _load_test_validation_data(self, test_X_path: Optional[str] = None, val_X_path: Optional[str] = None, val_y_path: Optional[str] = None) -> None:
        """Load and prepare the test and validation data with enhanced validation."""
        data_cfg = self.config['data']
        
        # Use provided paths or default from config
        test_X_path = test_X_path or Path(data_cfg['datadir']) / data_cfg['xpred_path']
        val_X_path = val_X_path or Path(data_cfg['datadir']) / data_cfg['xval_path']
        val_y_path = val_y_path or Path(data_cfg['datadir']) / data_cfg['yval_path']
        
        self.logger.info(f"Loading test data from X: {test_X_path}")
        self.logger.info(f"Loading validation data from X: {val_X_path}, y: {val_y_path}")
        
        # Load test data
        try:
            test_X_df = pd.read_csv(test_X_path)
        except Exception as e:
            raise IOError(f"Error loading test data file: {e}")
        
        self.logger.info(f"Loaded test X: {test_X_df.shape}")
        
        # Validate test columns
        required_x_cols = [self.id_field] + self.coord_cols + self.feature_names + self.bbox_cols
        self._validate_columns(test_X_df, required_x_cols, "Test X CSV")
        
        # Create test features
        test_X, _, test_w, test_X_r, test_X_l = self._create_features(test_X_df)
        self.test = Dataset(X=test_X, y=None, w=test_w, df=test_X_df, raw_X=test_X_r, lag_X=test_X_l)
        
        # Load validation data
        try:
            val_X_df = pd.read_csv(val_X_path)
            val_Y_df = pd.read_csv(val_y_path)
        except Exception as e:
            raise IOError(f"Error loading validation data files: {e}")
        
        self.logger.info(f"Loaded validation X: {val_X_df.shape}, Y: {val_Y_df.shape}")
        
        # Validate validation columns
        required_y_cols = [self.id_field, self.target_name] + self.bbox_cols
        self._validate_columns(val_X_df, required_x_cols, "Validation X CSV")
        self._validate_columns(val_Y_df, required_y_cols, "Validation Y CSV")
        
        # Merge validation datasets
        val_dframe = val_Y_df.merge(
            val_X_df[[self.id_field] + self.coord_cols + self.feature_names], 
            on=self.id_field, 
            how="inner"
        )
        
        if len(val_dframe) == 0:
            raise ValueError("Merge resulted in an empty validation DataFrame. Check ID field consistency.")
        
        self.logger.info(f"Merged validation dataset shape: {val_dframe.shape}")

        # Check for NaNs in validation target
        if val_dframe[self.target_name].isna().any():
            self.logger.warning(f"Validation target column '{self.target_name}' contains NaN values. These will be dropped.")
            val_dframe = val_dframe.dropna(subset=[self.target_name])
        
        # Create validation features and target
        val_X, val_y, val_w, val_X_r, val_X_l = self._create_features(val_dframe)
        self.validation = Dataset(X=val_X, y=val_y, w=val_w, df=val_dframe, raw_X=val_X_r, lag_X=val_X_l)
    
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
    
    def _create_features(self, dataf: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Any]:
        """Create features (optionally spatially lagged variables) and handle missing values."""
        feature_cfg = self.config['feature']
        model_cfg = self.config['model']
        self.logger.info("Creating features...")

        if self.target_name in dataf.columns:
            _y = (dataf[self.target_name] > 0).astype(int).values
            target_counts = np.bincount(_y)
            self.logger.info(f"Binary target distribution - 0s: {target_counts[0]}, 1s: {target_counts[1]}")

            # Check for class imbalance
            minority_ratio = min(target_counts) / sum(target_counts)
            if minority_ratio < 0.1:
                self.logger.warning(f"Severe class imbalance detected. Minority class ratio: {minority_ratio:.3f}")
        else:
            _y = None
            self.logger.info("No target variable found; proceeding without target.")

        self.logger.info("Building spatial weights matrix...")
        
        # Create polygons from bounding boxes
        try:
            polygons = []
            for _, row in dataf.iterrows():
                # Assuming bbox_cols = [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = [row[col] for col in self.bbox_cols]
                polygon = Polygon([(xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
                polygons.append(polygon)
            
            temp_gdf = gpd.GeoDataFrame(dataf, geometry=polygons)
        except Exception as e:
            raise ValueError(f"Error creating polygons from bounding box columns: {e}")
        
        # Create spatial weights
        spatial_weights = self._create_spatial_weights(temp_gdf)
        
        # Prepare feature matrix
        self.logger.info("Creating feature matrix...")
        X_raw = dataf[self.feature_names].copy()
        
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
        
        spatial_lag = model_cfg.get('lag_order', 1)
        if spatial_lag < 1:
            self.logger.info("No spatial lagging applied as lag_order < 1")
            self.logger.info(f"Final feature matrix shape: {X_raw.shape}")
            self.logger.info(f"Feature columns: {self.feature_names}")
            return X_raw, _y, spatial_weights, X_raw, None
        
        # Create spatially lagged features
        self.logger.info("Creating spatially lagged features...")
        try:
            X_lag = pd.DataFrame({
                f"W_{col}": lag_spatial(spatial_weights, X_raw[col].values) 
                for col in X_raw.columns
            })
        except Exception as e:
            raise ValueError(f"Error creating spatial lag features: {e}")
        
        # Combine original and lagged features
        X_all = pd.concat([X_raw, X_lag], axis=1)
        
        self.logger.info(f"Final feature matrix shape: {X_all.shape}")
        self.logger.info(f"Feature columns: {self.feature_names}")
        
        return X_all, _y, spatial_weights, X_raw, X_lag
        
    def _train(self):
        """
        Trains the model using the loaded data.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement model training")
    
    def _evaluate(self):
        """
        Evaluates the model's performance on the test data.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Must implement model evaluation")