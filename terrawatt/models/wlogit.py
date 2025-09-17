"""
Logit Model Module
Logistic Regression implementation with spatial cross-validation using statsmodels.
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
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report

from libpysal.weights import Queen, KNN
from libpysal.weights import lag_spatial
from esda.moran import Moran

import terrawatt.models.terrawatt as twm

warnings.filterwarnings("ignore")

def pearson_residuals(y, y_hat):
    """Calculate Pearson residuals."""
    return (y - y_hat) / np.sqrt(y_hat * (1 - y_hat))


class WLogitModel(twm.SpatialModel):
    """Weighted Logistic Regression Model using statsmodels."""

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the Logit model.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration YAML file (default is "config.yaml")
        """
        super().__init__(config_path)
        self.residuals = None
        self.predictions = None
        
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
        self.logger.info("Weighted Logit Model initialized")

    def _check_compatibility(self):
        """Check if the model kind is compatible."""
        model_cfg = self.config['model']
        if model_cfg['kind'] != self.kind:
            raise ValueError(f"Model kind '{model_cfg['kind']}' is not compatible with WLogitModel.")
    
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
    
    def _calculate_weights(self):
        """Calculate class weights for weighted regression."""
        self.logger.info("Calculating class weights...")
        
        # Calculate the number of samples in each class
        n_samples = len(self.y)
        n_class_0 = np.sum(self.y == 0)
        n_class_1 = np.sum(self.y == 1)
        
        # Calculate weights
        weight_for_0 = n_samples / (2 * n_class_0)
        weight_for_1 = n_samples / (2 * n_class_1)
        
        # Assign weight to each sample
        self.weights = np.where(self.y.ravel() == 1, weight_for_1, weight_for_0)
        self.logger.info(f"Weight for class 0: {weight_for_0:.2f}, Weight for class 1: {weight_for_1:.2f}")

    def _create_features(self):
        """
        Create features for Weighted Logit Regression.
        
        Returns
        -------
        tuple
            (X_all, y) feature matrix and target vector
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

        # Calculate weights after y has been created
        self._calculate_weights()
        
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
        
        try:
            self.logger.info("Performing diagnostic tests...")

            # Moran's I for spatial autocorrelation of residuals (Spatial CV)
            moran_test = Moran(self.residuals, self.w, permutations=9999)
            self.diagnostics['moran_cv'] = {
                'I': moran_test.I,
                'EI': moran_test.EI,
                'p_value': moran_test.p_sim,
                'z_score': moran_test.z_sim
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
            plt.hist(moran_test.sim, bins=50, color='lightblue')
            plt.axvline(moran_test.I, color='red', linestyle='dashed')
            plt.title("Reference Distribution of Moran's I under Null Hypothesis")
            plt.xlabel("Moran's I Value")
            plt.savefig(plot_path)
            plt.close()
            self.logger.info(f"Moran's I distribution plot saved to {plot_path}")

            # Classification Report
            class_report = classification_report(
                self.predictions['y_true'], 
                self.predictions['y_pred'], 
                output_dict=True
            )
            self.diagnostics['classification_report'] = class_report
            report_path = os.path.join(out_cfg['outdir'], out_cfg['report'])
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            pd.DataFrame(class_report).transpose().to_csv(report_path, index=True)
            self.logger.info(f"Classification report saved to {report_path}")

        except Exception as e:
            self.logger.warning(f"Error in diagnostic tests: {e}")
            self.diagnostics['error'] = str(e)
        
        return self.diagnostics
    
    def _train(self, X=None, y=None):
        """
        Train the Weighted Logit model using statsmodels.
        
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

        if not hasattr(self, 'weights'):
            self._calculate_weights()

        train_cfg = self.config['training']
        model_cfg = self.config['model']
        data_cfg = self.config['data']
        out_cfg = self.config['output']

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

        self.logger.info("Training Weighted Logit model using statsmodels...")

        records, cv_scores = [], []
        all_residuals = np.zeros(len(self.df))
        
        for fold in range(train_cfg['scv_folds']):
            self.logger.info(f"Processing fold {fold + 1}/{train_cfg['scv_folds']}...")
            train_idx, test_idx = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y[train_idx].ravel(), y[test_idx].ravel()

            # Get the weights for the current training fold
            weights_train = self.weights[train_idx]

            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)
            
            cv_model = sm.Logit(y_train, X_train_const, freq_weights=weights_train).fit()
            cv_prob = cv_model.predict(X_test_const)
            cv_pred = (cv_prob >= model_cfg['threshold']).astype(int)

            cv_accuracy = accuracy_score(y_test, cv_pred)
            cv_logloss = log_loss(y_test, cv_prob)
            cv_scores.append({"Accuracy": cv_accuracy, "LogLoss": cv_logloss})
            
            self.logger.info(f"  Fold {fold + 1} LogLoss: {cv_logloss:.4f} Accuracy: {cv_accuracy:.4f}")
            
            # Store results and residuals
            all_residuals[test_idx] = pearson_residuals(y_test, cv_prob)
            for i, idx in enumerate(test_idx):
                records.append({
                    data_cfg['id_field']: self.df.iloc[idx][data_cfg['id_field']], 
                    "fold": fold,
                    "y_true": int(y_test[i]),
                    "y_prob": cv_prob[idx],
                    "y_pred": int(cv_prob[idx] >= model_cfg['threshold']),
                })
            
            # Store coefficients
            coef_df = pd.DataFrame({
                'fold': fold,
                'Feature': ["intercept"] + list(X.columns),
                'Coefficient': cv_model.params,
                'Std_Error': cv_model.bse,
                't_value': cv_model.tvalues,
                'P_value': cv_model.pvalues,
                'CI_Lower': cv_model.conf_int()[0],
                'CI_Upper': cv_model.conf_int()[1]
            })
            coef_df.to_csv(os.path.join(self.config['logging']['logdir'], f"fold_{fold+1}_coefficients.csv"), index=False)

        self.logger.info(f"Cross-validation scores: {cv_scores}")
        
        # Fit Logit model using statsmodels
        X_scaled_const = sm.add_constant(X_scaled)

        self.model = sm.Logit(y, X_scaled_const, freq_weights=self.weights)
        self.results = self.model.fit()
        self.residuals = all_residuals
        
        self.logger.info("Model training complete.")
        self.logger.info(f"\nModel Summary:\n{self.results.summary()}")
        self.logger.info(f"Model BIC: {self.results.bic}")
        self.logger.info(f"Model AfIC: {self.results.aic}")
        self.logger.info(f"Model Log-Likelihood: {self.results.llf}")
        
        # Extract coefficients with statistical significance
        coefficients_df = pd.DataFrame({
            'Feature': ["intercept"] + list(X.columns),
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
        
        # Save coefficients
        self.parameters = coefficients_df
        coef_path = os.path.join(out_cfg['outdir'], out_cfg['coefficients'])
        os.makedirs(out_cfg['outdir'], exist_ok=True)
        coefficients_df.to_csv(coef_path, index=False)
        self.logger.info(f"Model coefficients saved to {coef_path}")

        # Save Predictions
        self.predictions = pd.DataFrame(records)
        pred_path = os.path.join(out_cfg['outdir'], out_cfg['predictions'])
        self.predictions.to_csv(pred_path, index=False)
        self.logger.info(f"Predictions saved to {pred_path}")
        
        return self.results