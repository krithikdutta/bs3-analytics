"""
Spatial Variational Autoencoder (VAE) for Anomaly Detection
Implements a VAE-based anomaly detection model that inherits from the SpatialModel
base class, suitable for spatial data.
"""
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from terrawatt.models.terrawatt import SpatialModel, ModelMetrics, ModelDiagnostics

warnings.filterwarnings("ignore", category=FutureWarning)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) network architecture.
    """
    def __init__(self, input_dim, latent_dim=8, num_classes=2):
        """
        Initializes the VAE network.

        Parameters
        ----------
        input_dim : int
            The number of input features.
        latent_dim : int, optional
            The dimensionality of the latent space. Defaults to 8.
        num_classes : int, optional
            The number of classes for the classification head. Defaults to 2.
        """
        super(VAE, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.mu = nn.Linear(16, latent_dim)
        self.logvar = nn.Linear(16, latent_dim)
        self.classifier = nn.Linear(16, num_classes)

        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def encode(self, x):
        """Encodes the input into the latent space."""
        h = self.encoder(x)
        return self.mu(h), self.logvar(h), self.classifier(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass of the VAE."""
        mu, logvar, logits = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, logits

def vae_loss(recon_x, x, mu, logvar, logits, y, classification_loss_weight=1.0):
    """
    Calculates the VAE loss, which is a combination of reconstruction loss,
    Kullback-Leibler (KL) divergence, and a classification loss.

    Parameters
    ----------
    recon_x : torch.Tensor
        The reconstructed input.
    x : torch.Tensor
        The original input.
    mu : torch.Tensor
        The mean of the latent distribution.
    logvar : torch.Tensor
        The log-variance of the latent distribution.
    logits : torch.Tensor
        The output logits from the classification head.
    y : torch.Tensor
        The true class labels.
    classification_loss_weight : float, optional
        Weight for the classification loss. Defaults to 1.0.

    Returns
    -------
    torch.Tensor
        The total VAE loss.
    """
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    classification_loss = nn.CrossEntropyLoss()(logits, y.long())
    return recon_loss + kld + classification_loss_weight * classification_loss


class SpatialVAEModel(SpatialModel):
    """
    Spatial VAE model for anomaly detection inheriting from SpatialModel.
    """
    def __init__(self, config_path: Path = Path("config.yaml")):
        """
        Initializes the Spatial VAE model.

        Parameters
        ----------
        config_path : Path
            Path to the configuration YAML file.
        """
        super().__init__(config_path)
        self._is_fitted = False
        self._anomaly_scores: Optional[np.ndarray] = None
        self._scaled_anomaly_scores: Optional[np.ndarray] = None
        
        # Configure model parameters
        model_cfg = self.config['model']
        self.latent_dim = model_cfg.get('latent_dim', 8)
        self.epochs = model_cfg.get('epochs', 30)
        self.learning_rate = model_cfg.get('learning_rate', 1e-3)
        self.batch_size = model_cfg.get('batch_size', 32)
        self.classification_loss_weight = model_cfg.get('classification_loss_weight', 1.0)
        
    def _train(self) -> None:
        """
        Trains the VAE model.
        """
        if self.train is None:
            self._load_data()

        self.logger.info("Starting VAE model training...")
        
        # Prepare data for PyTorch
        X_tensor = torch.tensor(self.train.X.values, dtype=torch.float32)
        y_tensor = torch.tensor(self.train.y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize VAE model, optimizer, and loss
        input_dim = self.train.X.shape[1]
        self.model = VAE(input_dim=input_dim, latent_dim=self.latent_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            for data in dataloader:
                batch_X, batch_y = data
                optimizer.zero_grad()
                recon, mu, logvar, logits = self.model(batch_X)
                loss = vae_loss(recon, batch_X, mu, logvar, logits, batch_y, self.classification_loss_weight)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        self._is_fitted = True
        self.logger.info("VAE model training complete.")

    def _evaluate(self) -> None:
        """
        Evaluates the VAE model by calculating anomaly scores and metrics
        on the validation set.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
        
        if self.validation is None:
            self._load_test_validation_data()

        self.logger.info("Evaluating VAE model performance on validation data...")

        # Calculate anomaly scores for the validation data
        with torch.no_grad():
            X_tensor = torch.tensor(self.validation.X.values, dtype=torch.float32)
            recon, _, _, _ = self.model(X_tensor)
            # Use Mean Squared Error (MSE) as the anomaly score
            anomaly_scores = ((recon - X_tensor)**2).mean(axis=1).numpy()
            
        # Reshape for scaling
        anomaly_scores_reshaped = anomaly_scores.reshape(-1, 1)
        
        # Scale anomaly scores to a range of [0, 1]
        scaler = MinMaxScaler()
        scaled_scores = scaler.fit_transform(anomaly_scores_reshaped)
        self._scaled_anomaly_scores = scaled_scores.flatten()
            
        y_true = (self.validation.df[self.target_name] > 0).astype(int).values
        
        # Calculate evaluation metrics
        try:
            auc_score = float(roc_auc_score(y_true, anomaly_scores))
            pr_auc = float(average_precision_score(y_true, anomaly_scores))
            
            # Determine a threshold for classification (e.g., 95th percentile)
            threshold = np.percentile(anomaly_scores, 95)
            y_pred = (anomaly_scores > threshold).astype(int)
            
            # Calculate other classification metrics
            conf_matrix = confusion_matrix(y_true, y_pred).tolist()
            report = classification_report(y_true, y_pred, output_dict=True)
            
            precision = float(report['1']['precision'])
            recall = float(report['1']['recall'])
            f1 = float(report['1']['f1-score'])

            self.metrics = ModelMetrics(
                pr_auc=pr_auc,
                brier_score=np.nan,
                log_loss=np.nan,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=conf_matrix,
                classification_report=str(report)
            )
            self.metrics.overall_auc = auc_score

            self.logger.info(f"Evaluation Metrics:")
            self.logger.info(f"  ROC-AUC: {self.metrics.overall_auc:.4f}")
            self.logger.info(f"  PR-AUC: {self.metrics.pr_auc:.4f}")
            self.logger.info(f"  Precision (Validation): {self.metrics.precision:.4f}")
            self.logger.info(f"  Recall (Validation): {self.metrics.recall:.4f}")
            self.logger.info(f"  F1-Score (Validation): {self.metrics.f1_score:.4f}")
            self.logger.info(f"  Confusion Matrix:\n{conf_matrix}")

        except Exception as e:
            self.logger.error(f"Error calculating evaluation metrics: {e}")
            raise RuntimeError(f"Error calculating evaluation metrics: {e}")
            
    def predict(self, X_new: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Makes anomaly predictions on new data.

        Parameters
        ----------
        X_new : pd.DataFrame, optional
            DataFrame containing new data to predict on. If None, uses test data.

        Returns
        -------
        pd.DataFrame
            DataFrame with anomaly scores and predictions.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been trained. Call train() first.")
            
        if X_new is None:
            if self.test is None:
                self._load_test_validation_data()
            _X = self.test.X
            _df = self.test.df
        else:
            # Create features for the new data
            _X, _, _, _, _ = self._create_features(X_new)
            _df = X_new
            
        self.logger.info("Making anomaly predictions...")
        
        with torch.no_grad():
            X_tensor = torch.tensor(_X.values, dtype=torch.float32)
            recon, _, _, _ = self.model(X_tensor)
            # Use MSE as anomaly score
            anomaly_scores = ((recon - X_tensor)**2).mean(axis=1).numpy()

        # Reshape for scaling
        anomaly_scores_reshaped = anomaly_scores.reshape(-1, 1)
        
        # Scale anomaly scores to a range of [0, 1]
        scaler = MinMaxScaler()
        scaled_scores = scaler.fit_transform(anomaly_scores_reshaped)

        results = pd.DataFrame({
            self.id_field: _df[self.id_field],
            'anomaly_score': anomaly_scores,
            'scaled_anomaly_score': scaled_scores.flatten()
        })

        # Save predictions
        out_cfg = self.config['output']
        output_dir = Path(out_cfg.get('outdir', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / out_cfg.get('predictions', 'vae_predictions.csv')
        results.to_csv(output_path, index=False)
        self.logger.info(f"Predictions saved to {output_path}")

        return results
        
    def run(self, X_path: Optional[str] = None, y_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Public method to train, evaluate, and predict with the model.
        
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
            self._load_data(X_path=X_path, y_path=y_path)
            self._train()
            self._evaluate()
            
            # Since VAE is an unsupervised model, diagnostics on residuals are not directly applicable
            # as it doesn't predict a class. We'll skip Moran's I on residuals here.
            
            return {
                'anomaly_scores': self._anomaly_scores,
                'scaled_anomaly_scores': self._scaled_anomaly_scores,
                'metrics': self.metrics,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
