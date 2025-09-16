"""
OLS Model Module
Ordinary Least Squares (OLS) Regression implementation with spatial cross-validation using statsmodels.
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from esda.moran import Moran

import terrawatt.models.terrawatt as twm

warnings.filterwarnings("ignore")


class OLSModel(twm.SpatialModel):
    """OLS Regression Model using statsmodels."""

    def __init__(self, config_path="config.yaml"):
        """
        Initialize the OLS model.
        
        Parameters
        ----------
        config_path : str, optional
            Path to the configuration YAML file (default is "config.yaml")
        """
        super().__init__(config_path)
        self.residuals = None
        
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
            if self.train.X is None or self.train.y is None:
                raise ValueError("No data available. Call load_data() first or provide X and y.")
            X = self.train.X
            y = self.train.y
        
        train_cfg = self.config['training']

        # Get coordinates for spatial CV
        coords = self.train.df[self.coord_cols].values
        
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
        all_residuals = np.zeros(len(self.train.df))

        
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
                    self.id_field: self.train.df.iloc[idx][self.id_field], 
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
        self.residuals = all_residuals
        
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

    def _perform_diagnostics(self):
        """
        Perform diagnostic tests on the regression results.
            
        Returns
        -------
        dict
            Dictionary containing diagnostic test results
        """
        out_cfg = self.config['output']
        diagnostics = {}
        
        try:
            self.logger.info("Performing diagnostic tests...")

            # Moran's I for spatial autocorrelation of residuals (Spatial CV)
            moran_test = Moran(self.residuals, self.train.w, permutations=9999)
            diagnostics['moran_cv'] = {
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

        except Exception as e:
            self.logger.warning(f"Error in diagnostic tests: {e}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def run(self) -> None:
        """
        Execute the full modeling pipeline: load data, preprocess, train, evaluate, and save results.
        """
        self._load_data()
        results, coefficients = self._train()
        diagnostics = self._perform_diagnostics()