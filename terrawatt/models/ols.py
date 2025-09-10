"""
OLS Model Module
Ordinary Least Squares (OLS) Regression implementation with spatial cross-validation using statsmodels.
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

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import acf

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from libpysal.weights import Queen, KNN
from libpysal.weights import lag_spatial
from esda.moran import Moran

import terrawatt.models.terrawatt as twm

warnings.filterwarnings("ignore")


class OLSModel(twm.InferenceModel):
    """OLS Regression Model using statsmodels."""

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the OLS model.
        
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
        self.logger.info("OLS Model initialized")

    def _check_compatibility(self):
        """Check if the model kind is compatible."""
        model_cfg = self.config['model']
        if model_cfg['kind'] != self.kind:
            raise ValueError(f"Model kind '{model_cfg['kind']}' is not compatible with OLSModel.")
    
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
        
        # Use continuous target variable for OLS regression
        self.y = self.df[data_cfg['target']].values
        self.logger.info(f"Target variable statistics: mean={np.mean(self.y):.3f}, std={np.std(self.y):.3f}")
        
        return self._create_features()
    
    def _create_features(self):
        """
        Create features for OLS Regression.
        
        Returns
        -------
        tuple
            (X_all, y) feature matrix and target vector
        """
        data_cfg = self.config['data']
        feature_cfg = self.config['feature']
        
        # Create feature matrix (X)
        self.logger.info("Creating features...")
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
        
        self.X_all = X_raw.copy()
        self.logger.info(f"Final feature matrix shape: {self.X_all.shape}")
        
        return self.X_all, self.y
    
    def _perform_diagnostics(self):
        """
        Perform diagnostic tests on the OLS results.
            
        Returns
        -------
        dict
            Dictionary containing diagnostic test results
        """
        self.diagnostics = {}
        results = self.results
        
        try:
            # Heteroscedasticity tests
            # bp_test = het_breuschpagan(results.resid, results.model.exog)
            # self.diagnostics['breusch_pagan'] = {
            #     'statistic': bp_test[0],
            #     'p_value': bp_test[1],
            #     'f_statistic': bp_test[2],
            #     'f_p_value': bp_test[3]
            # }
            
            white_test = het_white(results.resid, results.model.exog)
            self.diagnostics['white_test'] = {
                'statistic': white_test[0],
                'p_value': white_test[1],
                'f_statistic': white_test[2],
                'f_p_value': white_test[3]
            }
            
            # Durbin-Watson test for autocorrelation
            dw_stat = durbin_watson(results.resid)
            self.diagnostics['durbin_watson'] = dw_stat
            
            # Jarque-Bera test for normality of residuals
            jb_test = jarque_bera(results.resid)
            self.diagnostics['jarque_bera'] = {
                'statistic': jb_test[0],
                'p_value': jb_test[1],
                'skewness': jb_test[2],
                'kurtosis': jb_test[3]
            }
            
        except Exception as e:
            self.logger.warning(f"Error in diagnostic tests: {e}")
            self.diagnostics['error'] = str(e)
        
        return self.diagnostics
    
    def _train(self, X=None, y=None):
        """
        Train the OLS model using statsmodels.
        
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

        # Choose scaler
        feature_cfg = self.config['feature']
        if feature_cfg['scaling'] == 'standard':
            scaler = StandardScaler()
        elif feature_cfg['scaling'] == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = None

        # Apply scaling outside pipeline
        if scaler:
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        else:
            X_scaled = X.copy()

        self.logger.info("Training OLS model using statsmodels...")

        records, cv_scores = [], []
        all_residuals = np.zeros(len(self.df))

        
        for fold in range(train_cfg['scv_folds']):
            self.logger.info(f"Processing fold {fold + 1}/{train_cfg['scv_folds']}...")
            train_idx, test_idx = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y[train_idx].ravel(), y[test_idx].ravel()

            # X_train_const = sm.add_constant(X_train)
            # X_test_const = sm.add_constant(X_test)
            
            cv_model = sm.OLS(y_train, X_train).fit()
            cv_pred = cv_model.predict(X_test)
            cv_score = r2_score(y_test, cv_pred)
            cv_scores.append(cv_score)
            
            self.logger.info(f"  Fold {fold + 1} R²: {cv_score:.4f}")
            
            # Store results and residuals
            all_residuals[test_idx] = y_test - cv_pred
            for i, idx in enumerate(test_idx):
                records.append({
                    data_cfg['id_field']: self.df.iloc[idx][data_cfg['id_field']], 
                    "fold": fold,
                    "y_true": int(y_test[i]),
                    "y_pred": cv_pred[idx]
                })
            
            # Store coefficients
            coef_df = pd.DataFrame({
                'fold': fold,
                'Feature': list(X.columns),
                'Coefficient': cv_model.params,
                'Std_Error': cv_model.bse,
                't_value': cv_model.tvalues,
                'P_value': cv_model.pvalues,
                'CI_Lower': cv_model.conf_int()[0],
                'CI_Upper': cv_model.conf_int()[1]
            })
            coef_df.to_csv(os.path.join(self.config['logging']['logdir'], f"fold_{fold+1}_ols_coefficients.csv"), index=False)

        self.logger.info(f"Cross-validation R² scores: {cv_scores}")
        self.logger.info(f"Mean CV R² score: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        # Fit OLS model using statsmodels
        self.model = sm.OLS(y, X_scaled)
        self.results = self.model.fit()
        self.parameters = self.results.params
        
        self.logger.info("Model training complete.")
        self.logger.info(f"\nModel Summary:\n{self.results.summary()}")
        
        # Extract coefficients with statistical significance
        coefficients_df = pd.DataFrame({
            'Feature': list(X.columns),
            'Coefficient': self.results.params,
            'Std_Error': self.results.bse,
            't_value': self.results.tvalues,
            'P_value': self.results.pvalues,
            'CI_Lower': self.results.conf_int()[0],
            'CI_Upper': self.results.conf_int()[1]
        })
        
        # Add significance indicators
        coefficients_df['Significant'] = coefficients_df['P_value'] < 0.05
        coefficients_df['Significance'] = pd.cut(
            coefficients_df['P_value'], 
            bins=[0, 0.001, 0.01, 0.05, 1], 
            labels=['***', '**', '*', ''],
            include_lowest=True
        )
        
        self.logger.info(f"\nModel Coefficients:\n{coefficients_df}")
        
        return self.results, coefficients_df