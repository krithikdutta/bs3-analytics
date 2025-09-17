"""
Bayesian SEM Model Module
Bayesian Spatial Error Model (SEM) implementation using PyMC with spatial cross-validation.
Inherits from SpatialModel base class for consistency with the terrawatt framework.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from esda.moran import Moran


from terrawatt.models.terrawatt import SpatialModel, ModelMetrics, ModelDiagnostics

import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, eye
import pymc as pm
import arviz as az

warnings.filterwarnings("ignore")


class SparseSolverOp(Op):
    """
    PyTensor Op: solve (I - λW) u = eps for u using sparse solver.
    """
    itypes = [pt.dscalar, pt.dvector]  # lambda, eps
    otypes = [pt.dvector]              # u

    def __init__(self, W, I):
        self.W = W  # scipy sparse csr_matrix
        self.I = I  # scipy sparse csr_matrix

    def make_node(self, lam, eps):
        lam = pt.as_tensor_variable(lam)
        eps = pt.as_tensor_variable(eps)
        return Apply(self, [lam, eps], [eps.type()])

    def perform(self, node, inputs, outputs):
        lam, eps = inputs
        A = self.I - lam * self.W
        u = spsolve(A, eps)
        outputs[0][0] = np.asarray(u, dtype=np.float64)

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[1]]


class BayesianSEMModel(SpatialModel):
    """
    Bayesian Spatial Error Model (SEM) using PyMC.
    
    This implementation uses a Bayesian framework with spatial error autocorrelation,
    following the specification: y = Xβ + u, where u = λWu + ε
    """
    
    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        """
        Initialize the Bayesian SEM model.
        
        Parameters
        ----------
        config_path : Union[str, Path]
            Path to the configuration YAML file
        """
        # Initialize parent class first
        super().__init__(config_path)
        
        # Initialize SEM-specific attributes
        self.trace: Optional[Any] = None
        self.model: Optional[pm.Model] = None
        self._training_predictions: Optional[pd.DataFrame] = None
        self._posterior_samples: Optional[pd.DataFrame] = None
        self._spatial_lambda: Optional[float] = None
        self._convergence_diagnostics: Optional[Dict[str, Any]] = None
    
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
    
    def _prepare_spatial_weights_matrix(self, weights, indices: np.ndarray) -> sparse.csr_matrix:
        """
        Prepare a sparse spatial weights matrix for a subset of data.
        
        Parameters
        ----------
        weights : libpysal.weights.W
            Spatial weights object.
        indices : np.ndarray
            Indices to subset the weights matrix.
            
        Returns
        -------
        scipy.sparse.csr_matrix
            Row-standardized sparse spatial weights matrix for the subset.
        """
        # Get the sparse representation of the full weights matrix
        w_sparse = weights.sparse
        
        # Subset the sparse matrix using the provided indices. This is more memory-efficient
        # than creating and subsetting a full dense matrix.
        w_subset = w_sparse[indices, :][:, indices]
        
        # Calculate row sums for standardization
        row_sums = np.array(w_subset.sum(axis=1)).flatten()
        
        # To avoid division by zero for islands (locations with no neighbors in the subset),
        # we replace zero sums with 1. The resulting row will correctly remain all zeros.
        row_sums[row_sums == 0] = 1
        
        # Create a diagonal matrix of the inverse row sums for efficient scaling
        inv_row_sums = 1.0 / row_sums
        scaler = sparse.diags(inv_row_sums, format='csr')
        
        # Perform row standardization via sparse matrix multiplication
        w_std_sparse = scaler @ w_subset
        
        return w_std_sparse
    
    def _fit_bayesian_sem(self, X_train: np.ndarray, y_train: np.ndarray, 
                         W_train: np.ndarray) -> Tuple[pm.Model, Any]:
        """
        Fit Bayesian SEM model using PyMC.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        W_train : np.ndarray
            Training spatial weights matrix
            
        Returns
        -------
        Tuple[pm.Model, trace]
            PyMC model and MCMC trace
        """
        train_cfg = self.config['training']
        model_cfg = self.config['model']
        
        n_train = len(y_train)
        n_features = X_train.shape[1]
        
        # MCMC parameters
        n_samples = train_cfg.get('n_samples', 1000)
        n_tune = train_cfg.get('n_tune', 1000)
        n_chains = train_cfg.get('n_chains', 2)
        target_accept = train_cfg.get('target_accept', 0.9)

        I = eye(n_train, format="csr")

        sparse_solver_op = SparseSolverOp(W_train, I)
        
        self.logger.info(f"Fitting Bayesian SEM with {n_samples} samples, {n_tune} tune, {n_chains} chains")
        
        with pm.Model() as sem_model:
            # Prior for regression coefficients
            beta_prior_std = model_cfg.get('beta_prior_std', 2.0)
            beta = pm.Normal("beta", mu=0, sigma=beta_prior_std, shape=n_features)
            
            # Prior for spatial parameter lambda
            lambda_prior_std = model_cfg.get('lambda_prior_std', 0.5)
            lam = pm.Normal("lambda", mu=0, sigma=lambda_prior_std)
            
            # Error term
            sigma_prior = model_cfg.get('sigma_prior', 1.0)
            eps = pm.Normal("eps", mu=0, sigma=sigma_prior, shape=n_train)
            
            # Spatial error: u = (I - λW)^(-1)ε
            try:
                u = sparse_solver_op(lam, eps)
            except Exception as e:
                self.logger.warning(f"Matrix inversion failed, using approximation: {e}")
                # Fallback: direct spatial lag approximation
                u = eps + lam * pm.math.dot(W_train, eps)
            
            # Linear predictor: Xβ + u
            mu = pm.math.dot(X_train, beta) + u
            
            # Likelihood
            p = pm.math.sigmoid(mu)
            y_obs = pm.Bernoulli("y_obs", p=p, observed=y_train)
            
            # Sample from posterior
            try:
                trace = pm.sample(
                    draws=n_samples,
                    tune=n_tune,
                    chains=n_chains,
                    target_accept=target_accept,
                    random_seed=42,
                    return_inferencedata=True
                )
            except Exception as e:
                self.logger.error(f"MCMC sampling failed: {e}")
                raise RuntimeError(f"Bayesian model fitting failed: {e}")
        
        return sem_model, trace
    
    def _predict_bayesian_sem(self, model: pm.Model, trace: Any, 
                             X_test: np.ndarray, W_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using fitted Bayesian SEM model.
        
        Parameters
        ----------
        model : pm.Model
            Fitted PyMC model
        trace : arviz.InferenceData
            MCMC trace
        X_test : np.ndarray
            Test features
        W_test : np.ndarray
            Test spatial weights matrix
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted probabilities and binary predictions
        """
        n_test = len(X_test)
        # I_test = np.eye(n_test)

        I = eye(n_test, format="csr")

        sparse_solver_op = SparseSolverOp(W_test, I)
        
        # Extract posterior means for prediction
        beta_mean = trace.posterior["beta"].mean(dim=("chain", "draw")).values
        lambda_mean = trace.posterior["lambda"].mean(dim=("chain", "draw")).values
        
        self.logger.info(f"Using posterior means - Lambda: {lambda_mean:.4f}")
        
        try:
            with pm.Model() as pred_model:
                # Use posterior means as point estimates
                beta = pm.ConstantData("beta", beta_mean)
                lam = pm.ConstantData("lambda", lambda_mean)
                
                # Generate new error terms for prediction
                eps = pm.Normal("eps", mu=0, sigma=1, shape=n_test)
                
                # Spatial error structure
                try:
                    u = sparse_solver_op(lam, eps)
                except:
                    # Fallback approximation
                    u = eps + lam * pm.math.dot(W_test, eps)
                
                # Linear predictor
                mu = pm.math.dot(X_test, beta) + u
                p = pm.math.sigmoid(mu)
                
                # Sample posterior predictive
                ppc = pm.sample_posterior_predictive(
                    trace, 
                    var_names=["eps"], 
                    predictions=True,
                    random_seed=42
                )
                
                # Calculate predicted probabilities
                y_pred_prob = ppc.predictions["p"].mean(dim=("chain", "draw")).values
                
        except Exception as e:
            self.logger.warning(f"Full posterior predictive failed, using point estimates: {e}")
            
            # Fallback: direct calculation with posterior means
            try:
                A_inv = np.linalg.inv(I_test - lambda_mean * W_test)
                # Use zero mean for error in point prediction
                u_mean = np.zeros(n_test)
                mu_pred = X_test @ beta_mean + u_mean
                y_pred_prob = 1 / (1 + np.exp(-mu_pred))  # sigmoid
            except np.linalg.LinAlgError:
                self.logger.warning("Matrix inversion failed, using no spatial error")
                mu_pred = X_test @ beta_mean
                y_pred_prob = 1 / (1 + np.exp(-mu_pred))
        
        # Convert to binary predictions
        threshold = self.config['model'].get('threshold', 0.5)
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        return y_pred_prob, y_pred
    
    def _train(self) -> None:
        """Train the Bayesian SEM model using spatial cross-validation."""
        if self.train is None:
            self._load_data()
        
        train_cfg = self.config['training']
        feature_cfg = self.config['feature']
        
        # Get data from parent class Dataset structure
        X_raw, y = self.train.raw_X, self.train.y
        coords = self.train.df[self.coord_cols].values
        
        # Apply scaling if specified
        if feature_cfg.get('scaling') == 'standard':
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_raw), 
                columns=X_raw.columns, 
                index=X_raw.index
            )
            self.logger.info("Applied StandardScaler to features")
        elif feature_cfg.get('scaling') == 'minmax':
            scaler = MinMaxScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_raw), 
                columns=X_raw.columns, 
                index=X_raw.index
            )
            self.logger.info("Applied MinMaxScaler to features")
        else:
            X_scaled = X_raw.copy()
        
        X = X_scaled.values  # Convert to numpy array for PyMC
        
        # Create spatial folds
        n_folds = train_cfg.get('scv_folds', 3)  # Fewer folds for Bayesian models
        fold_ids = self._create_spatial_folds(coords, n_folds)
        
        # Cross-validation
        records = []
        posterior_records = []
        all_residuals = np.zeros(len(self.train.df))
        fold_aucs = []
        
        for fold in range(n_folds):
            self.logger.info(f"Processing fold {fold + 1}/{n_folds}...")
            
            train_idx = np.where(fold_ids != fold)[0]
            test_idx = np.where(fold_ids == fold)[0]
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            self.logger.info(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Prepare spatial weights matrices
            try:
                W_train = self._prepare_spatial_weights_matrix(self.train.w, train_idx)
                W_test = self._prepare_spatial_weights_matrix(self.train.w, test_idx)
            except Exception as e:
                self.logger.error(f"Error preparing weights for fold {fold}: {e}")
                continue
            
            # Fit Bayesian SEM model
            try:
                fold_model, fold_trace = self._fit_bayesian_sem(X_train, y_train, W_train)
                
                # Store convergence diagnostics
                if fold == 0:  # Store diagnostics from first fold
                    try:
                        rhat = az.rhat(fold_trace)
                        ess = az.ess(fold_trace)
                        self._convergence_diagnostics = {
                            'rhat_max': float(rhat.max().values),
                            'ess_min': float(ess.min().values),
                            'n_divergent': int(fold_trace.sample_stats.diverging.sum().values)
                        }
                        self.logger.info(f"Convergence: R-hat max = {self._convergence_diagnostics['rhat_max']:.4f}")
                        self.logger.info(f"Convergence: ESS min = {self._convergence_diagnostics['ess_min']:.0f}")
                    except Exception as e:
                        self.logger.warning(f"Could not compute convergence diagnostics: {e}")
                
            except Exception as e:
                self.logger.error(f"Error training fold {fold}: {e}")
                continue
            
            # Make predictions
            try:
                prob, pred = self._predict_bayesian_sem(fold_model, fold_trace, X_test, W_test)
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
            
            # Store posterior summaries
            try:
                posterior_summary = az.summary(fold_trace)
                lambda_mean = fold_trace.posterior["lambda"].mean().values
                lambda_hdi = az.hdi(fold_trace, var_names=["lambda"])["lambda"].values
                
                posterior_records.append({
                    "fold": fold,
                    "parameter": "lambda",
                    "mean": float(lambda_mean),
                    "hdi_lower": float(lambda_hdi[0]),
                    "hdi_upper": float(lambda_hdi[1])
                })
                
                # Store beta coefficients
                for i, feature in enumerate(X_scaled.columns):
                    beta_mean = fold_trace.posterior["beta"][..., i].mean().values
                    beta_hdi = az.hdi(fold_trace, var_names=["beta"])["beta"][i].values
                    
                    posterior_records.append({
                        "fold": fold,
                        "parameter": f"beta_{feature}",
                        "mean": float(beta_mean),
                        "hdi_lower": float(beta_hdi[0]),
                        "hdi_upper": float(beta_hdi[1])
                    })
            except Exception as e:
                self.logger.warning(f"Error extracting posterior summaries for fold {fold}: {e}")
        
        # Create result DataFrames
        pred_df = pd.DataFrame.from_records(records)
        posterior_df = pd.DataFrame.from_records(posterior_records)
        
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
            W_full = self._prepare_spatial_weights_matrix(self.train.w, np.arange(len(X)))
            final_model, final_trace = self._fit_bayesian_sem(X, y, W_full)
            
            # Store final results
            self.model = final_model
            self.trace = final_trace
            self._is_fitted = True
            
            # Extract spatial parameter
            self._spatial_lambda = float(final_trace.posterior["lambda"].mean().values)
            
            self.logger.info("Final model training complete.")
            self.logger.info(f"Final spatial parameter λ = {self._spatial_lambda:.4f}")
            
        except Exception as e:
            raise RuntimeError(f"Error training final model: {e}")
        
        # Store training results
        self._training_predictions = pred_df
        self._posterior_samples = posterior_df
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
                pr_auc=float(roc_auc_score(y_true, y_prob)),
                brier_score=float(np.mean((y_prob - y_true) ** 2)),  # Brier score
                log_loss=float(log_loss(y_true, y_prob)),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
                confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
                classification_report=classification_report(y_true, y_pred)
            )
            
            self.logger.info("Evaluation Metrics:")
            self.logger.info(f"  AUC: {self.metrics.pr_auc:.4f}")
            self.logger.info(f"  Brier Score: {self.metrics.brier_score:.4f}")
            self.logger.info(f"  Log Loss: {self.metrics.log_loss:.4f}")
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
            if self.test is None:
                self._load_test_validation_data()
            X_new = self.test.raw_X.copy()
            test_weights = self.test.w
            test_df = self.test.df
        else:
            # For custom data, spatial weights would need to be provided
            raise NotImplementedError("Prediction on custom data requires spatial weights implementation")
        
        self.logger.info("Making Bayesian predictions...")
        
        # Prepare test data
        X_test = X_new.values
        W_test = self._prepare_spatial_weights_matrix(test_weights, np.arange(len(X_test)))
        
        try:
            probabilities, predictions = self._predict_bayesian_sem(
                self.model, self.trace, X_test, W_test
            )
        except Exception as e:
            raise RuntimeError(f"Error making predictions: {e}")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'id': test_df[self.id_field],
            'probability': probabilities,
            'prediction': predictions
        })
        
        # Add confidence categories based on prediction uncertainty
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
                'posterior_samples': self._posterior_samples,
                'metrics': self.metrics,
                'diagnostics': self.diagnostics,
                'cv_metrics': getattr(self, '_cv_metrics', {}),
                'convergence_diagnostics': getattr(self, '_convergence_diagnostics', {}),
                'spatial_lambda': self._spatial_lambda,
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
                f.write(f"Bayesian Spatial Error Model Report\n")
                f.write(f"Model: {self.name}\n")
                f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                
                # Model Configuration
                f.write("MODEL CONFIGURATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Model Type: Bayesian Spatial Error Model (SEM)\n")
                f.write(f"Features: {len(self.feature_names)}\n")
                f.write(f"Spatial Parameter λ: {self._spatial_lambda:.4f}\n\n")
                
                # Performance Metrics
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"AUC-ROC: {self.metrics.pr_auc:.4f}\n")
                f.write(f"Brier Score: {self.metrics.brier_score:.4f}\n")
                f.write(f"Log Loss: {self.metrics.log_loss:.4f}\n")
                f.write(f"Precision: {self.metrics.precision:.4f}\n")
                f.write(f"Recall: {self.metrics.recall:.4f}\n")
                f.write(f"F1-Score: {self.metrics.f1_score:.4f}\n\n")
                
                # Convergence Diagnostics
                if self._convergence_diagnostics:
                    f.write("MCMC CONVERGENCE DIAGNOSTICS\n")
                    f.write("-" * 40 + "\n")
                    conv = self._convergence_diagnostics
                    f.write(f"Max R-hat: {conv.get('rhat_max', 'N/A'):.4f}\n")
                    f.write(f"Min Effective Sample Size: {conv.get('ess_min', 'N/A'):.0f}\n")
                    f.write(f"Divergent Transitions: {conv.get('n_divergent', 'N/A')}\n")
                    
                    # Convergence assessment
                    rhat_ok = conv.get('rhat_max', 2.0) < 1.1
                    ess_ok = conv.get('ess_min', 0) > 100
                    div_ok = conv.get('n_divergent', 100) < 10
                    
                    f.write(f"Convergence Status: ")
                    if rhat_ok and ess_ok and div_ok:
                        f.write("GOOD\n")
                    else:
                        f.write("POOR - Consider increasing sampling parameters\n")
                    f.write("\n")
                
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
                
                # Posterior Parameter Summaries
                if hasattr(self, '_posterior_samples') and not self._posterior_samples.empty:
                    f.write("POSTERIOR PARAMETER SUMMARIES\n")
                    f.write("-" * 45 + "\n")
                    
                    # Group by parameter and compute cross-validation statistics
                    param_stats = self._posterior_samples.groupby('parameter').agg({
                        'mean': ['mean', 'std'],
                        'hdi_lower': 'mean',
                        'hdi_upper': 'mean'
                    }).round(4)
                    
                    # Spatial parameter
                    lambda_stats = self._posterior_samples[
                        self._posterior_samples['parameter'] == 'lambda'
                    ]
                    if not lambda_stats.empty:
                        f.write(f"Spatial Parameter (λ):\n")
                        f.write(f"  Mean: {lambda_stats['mean'].mean():.4f} ± {lambda_stats['mean'].std():.4f}\n")
                        f.write(f"  95% HDI: [{lambda_stats['hdi_lower'].mean():.4f}, {lambda_stats['hdi_upper'].mean():.4f}]\n\n")
                    
                    # Beta coefficients
                    beta_stats = self._posterior_samples[
                        self._posterior_samples['parameter'].str.startswith('beta_')
                    ]
                    if not beta_stats.empty:
                        f.write("Regression Coefficients (β):\n")
                        for param in beta_stats['parameter'].unique():
                            param_data = beta_stats[beta_stats['parameter'] == param]
                            feature_name = param.replace('beta_', '')
                            mean_est = param_data['mean'].mean()
                            std_est = param_data['mean'].std()
                            hdi_lower = param_data['hdi_lower'].mean()
                            hdi_upper = param_data['hdi_upper'].mean()
                            
                            # Check if HDI excludes zero (significant effect)
                            significant = (hdi_lower > 0 and hdi_upper > 0) or (hdi_lower < 0 and hdi_upper < 0)
                            sig_marker = "*" if significant else ""
                            
                            f.write(f"  {feature_name:20s}: {mean_est:8.4f} ± {std_est:.4f} "
                                   f"[{hdi_lower:.4f}, {hdi_upper:.4f}] {sig_marker}\n")
                        f.write("\n")
                
                # Model Interpretation
                f.write("MODEL INTERPRETATION\n")
                f.write("-" * 30 + "\n")
                
                # Spatial parameter interpretation
                if self._spatial_lambda is not None:
                    if abs(self._spatial_lambda) > 0.1:
                        f.write(f"Spatial Effects: Strong spatial autocorrelation (λ = {self._spatial_lambda:.4f})\n")
                        if self._spatial_lambda > 0:
                            f.write("  Positive spatial spillovers detected in error term\n")
                        else:
                            f.write("  Negative spatial spillovers detected in error term\n")
                    else:
                        f.write(f"Spatial Effects: Weak spatial autocorrelation (λ = {self._spatial_lambda:.4f})\n")
                        f.write("  Limited spatial spillover effects\n")
                
                # Residual diagnostics interpretation
                if self.diagnostics.global_moran_p < 0.05:
                    f.write("Residual Diagnostics: Remaining spatial autocorrelation detected\n")
                    f.write("  Consider additional spatial features or alternative model specification\n")
                else:
                    f.write("Residual Diagnostics: No significant spatial autocorrelation in residuals\n")
                    f.write("  Model adequately captures spatial dependencies\n")
                
                f.write("\n" + "=" * 70 + "\n")
            
            self.logger.info(f"Model report saved to {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            raise IOError(f"Could not generate report: {e}")
    
    def get_posterior_summary(self) -> pd.DataFrame:
        """Get posterior parameter summaries from cross-validation."""
        if not hasattr(self, '_posterior_samples') or self._posterior_samples.empty:
            raise ValueError("No posterior samples available. Model may not be trained.")
        
        # Compute summary statistics across folds
        summary = self._posterior_samples.groupby('parameter').agg({
            'mean': ['mean', 'std', 'min', 'max'],
            'hdi_lower': ['mean', 'std'],
            'hdi_upper': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        
        return summary
    
    def get_spatial_effects(self) -> Dict[str, Any]:
        """Get spatial effects analysis."""
        if not self._is_fitted:
            raise ValueError("Model has not been trained yet.")
        
        # Extract spatial parameter information
        spatial_effects = {
            'lambda_estimate': self._spatial_lambda,
            'lambda_interpretation': self._interpret_lambda(),
            'residual_autocorrelation': {
                'moran_i': self.diagnostics.global_moran_i,
                'p_value': self.diagnostics.global_moran_p,
                'significant': self.diagnostics.global_moran_p < 0.05
            }
        }
        
        # Add convergence information if available
        if self._convergence_diagnostics:
            spatial_effects['convergence'] = self._convergence_diagnostics
        
        return spatial_effects
    
    def _interpret_lambda(self) -> str:
        """Interpret the spatial parameter lambda."""
        if self._spatial_lambda is None:
            return "No spatial parameter estimated"
        
        lambda_val = self._spatial_lambda
        
        if abs(lambda_val) < 0.05:
            return "Very weak spatial autocorrelation in errors"
        elif abs(lambda_val) < 0.2:
            return "Weak spatial autocorrelation in errors"
        elif abs(lambda_val) < 0.5:
            return "Moderate spatial autocorrelation in errors"
        elif abs(lambda_val) < 0.8:
            return "Strong spatial autocorrelation in errors"
        else:
            return "Very strong spatial autocorrelation in errors"
    
    def plot_posterior_diagnostics(self, save_plots: bool = True) -> Optional[List[str]]:
        """
        Generate posterior diagnostic plots.
        
        Parameters
        ----------
        save_plots : bool
            Whether to save plots to files
            
        Returns
        -------
        Optional[List[str]]
            List of plot file paths if save_plots=True
        """
        if not self._is_fitted or self.trace is None:
            raise ValueError("Model has not been trained yet.")
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("Matplotlib not available, cannot generate plots")
            return None
        
        plot_paths = []
        output_cfg = self.config['output']
        output_dir = Path(output_cfg['outdir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Trace plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            az.plot_trace(self.trace, var_names=["lambda"], axes=axes[:, 0])
            az.plot_trace(self.trace, var_names=["beta"], axes=axes[:, 1], coords={"beta_dim_0": 0})
            plt.suptitle("MCMC Trace Plots")
            plt.tight_layout()
            
            if save_plots:
                trace_path = output_dir / 'trace_plots.png'
                plt.savefig(trace_path, dpi=300, bbox_inches='tight')
                plot_paths.append(str(trace_path))
                self.logger.info(f"Trace plots saved to {trace_path}")
            
            plt.close()
            
            # Posterior distributions
            fig = plt.figure(figsize=(12, 6))
            az.plot_posterior(self.trace, var_names=["lambda", "beta"], 
                            coords={"beta_dim_0": slice(None, min(5, len(self.feature_names)))})
            plt.suptitle("Posterior Distributions")
            
            if save_plots:
                posterior_path = output_dir / 'posterior_distributions.png'
                plt.savefig(posterior_path, dpi=300, bbox_inches='tight')
                plot_paths.append(str(posterior_path))
                self.logger.info(f"Posterior distribution plots saved to {posterior_path}")
            
            plt.close()
            
            # R-hat plot
            fig = plt.figure(figsize=(10, 6))
            az.plot_rank(self.trace, var_names=["lambda"])
            plt.suptitle("Rank Plot for Convergence Assessment")
            
            if save_plots:
                rank_path = output_dir / 'rank_plot.png'
                plt.savefig(rank_path, dpi=300, bbox_inches='tight')
                plot_paths.append(str(rank_path))
                self.logger.info(f"Rank plots saved to {rank_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error generating diagnostic plots: {e}")
            return None
        
        return plot_paths if save_plots else None
    
    def get_model_comparison_metrics(self) -> Dict[str, float]:
        """
        Get model comparison metrics (WAIC, LOO).
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing model comparison metrics
        """
        if not self._is_fitted or self.trace is None:
            raise ValueError("Model has not been trained yet.")
        
        try:
            # Calculate WAIC and LOO
            waic = az.waic(self.trace)
            loo = az.loo(self.trace)
            
            comparison_metrics = {
                'waic': float(waic.waic),
                'waic_se': float(waic.se),
                'loo': float(loo.loo),
                'loo_se': float(loo.se),
                'p_waic': float(waic.p_waic),
                'p_loo': float(loo.p_loo)
            }
            
            self.logger.info("Model Comparison Metrics:")
            self.logger.info(f"  WAIC: {comparison_metrics['waic']:.2f} ± {comparison_metrics['waic_se']:.2f}")
            self.logger.info(f"  LOO: {comparison_metrics['loo']:.2f} ± {comparison_metrics['loo_se']:.2f}")
            
            return comparison_metrics
            
        except Exception as e:
            self.logger.warning(f"Error computing model comparison metrics: {e}")
            return {}
    
    def save_trace(self, filepath: Optional[str] = None) -> str:
        """
        Save MCMC trace to NetCDF file.
        
        Parameters
        ----------
        filepath : Optional[str]
            Path to save trace. If None, uses output directory from config
            
        Returns
        -------
        str
            Path where trace was saved
        """
        if not self._is_fitted or self.trace is None:
            raise ValueError("Model has not been trained yet.")
        
        if filepath is None:
            output_cfg = self.config['output']
            output_dir = Path(output_cfg['outdir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / 'mcmc_trace.nc'
        
        try:
            self.trace.to_netcdf(filepath)
            self.logger.info(f"MCMC trace saved to {filepath}")
            return str(filepath)
        except Exception as e:
            raise IOError(f"Error saving trace: {e}")
    
    def load_trace(self, filepath: str) -> None:
        """
        Load MCMC trace from NetCDF file.
        
        Parameters
        ----------
        filepath : str
            Path to load trace from
        """
        try:
            self.trace = az.from_netcdf(filepath)
            self._is_fitted = True
            self.logger.info(f"MCMC trace loaded from {filepath}")
        except Exception as e:
            raise IOError(f"Error loading trace: {e}")