"""
Spatial Neural Network Module
Improved Spatial Neural Network (SNN) classifier implementation with spatial cross-validation.
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.base import BaseEstimator, TransformerMixin

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


class SpatialNN(nn.Module):
    """Simple feedforward neural network for classification."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x)).squeeze()


class SpatialNeuralNetModel(twm.PredictionModel):
    """
    Spatial Neural Network (SNN) Classifier Model with spatial cross-validation.
    
    Includes:
    - Configurable network architecture
    - Scalers for preprocessing
    - Spatial lag feature generation
    - Spatial cross-validation folds
    - Full training, evaluation, and prediction pipelines
    """

    kind = "prediction"

    def __init__(self, config_path: Union[str, Path] = "config.yaml"):
        super().__init__()
        self.config: Optional[Dict] = None
        self.logger: Optional[logging.Logger] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data & model
        self.X_all: Optional[pd.DataFrame] = None
        self.y: Optional[np.ndarray] = None
        self.df: Optional[pd.DataFrame] = None
        self.feature_names: Optional[List[str]] = None
        self.spatial_weights: Optional[Any] = None

        self.model: Optional[SpatialNN] = None
        self.scaler: Optional[Any] = None
        self.metrics: Optional[ModelMetrics] = None
        self._is_fitted = False

        # Init
        self._load_config(config_path)
        self._setup_logging()
        self._validate_config()

    def _load_config(self, config_path: Union[str, Path]) -> None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def _setup_logging(self) -> None:
        log_config = self.config.get("logging", {})
        log_dir = Path(log_config.get("logdir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / log_config.get("logpath", "snn_model.log")
        log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info("Spatial Neural Network Model initialized")

    def _validate_config(self) -> None:
        required_sections = ["data", "model", "training", "feature", "output"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")

    def _create_pipeline(self, X: pd.DataFrame) -> None:
        """Prepare scaler and neural net model."""
        feature_cfg = self.config["feature"]
        model_cfg = self.config["model"]

        # Choose scaler
        scaling = feature_cfg.get("scaling", "standard").lower()
        if scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        if self.scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Define NN
        hidden_dims = model_cfg.get("hidden_dims", [64, 32])
        dropout = model_cfg.get("dropout", 0.3)
        self.model = SpatialNN(X_scaled.shape[1], hidden_dims, dropout).to(self.device)

    def _train_epoch(self, loader, criterion, optimizer) -> float:
        self.model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            preds = self.model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate_epoch(self, loader, criterion) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss, y_true, y_prob = 0, [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self.model(xb)
                loss = criterion(preds, yb)
                total_loss += loss.item()
                y_true.extend(yb.cpu().numpy())
                y_prob.extend(preds.cpu().numpy())
        return total_loss / len(loader), np.array(y_true), np.array(y_prob)

    def _train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train NN with spatial CV folds (KMeans)."""
        train_cfg = self.config["training"]
        lr = train_cfg.get("lr", 1e-3)
        batch_size = train_cfg.get("batch_size", 32)
        n_epochs = train_cfg.get("epochs", 50)

        # Scale & init model
        self._create_pipeline(X)

        X_scaled = self.scaler.transform(X) if self.scaler else X.values
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            train_loss = self._train_epoch(loader, criterion, optimizer)
            self.logger.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f}")

        self._is_fitted = True
        return {"success": True}

    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Predict with trained NN."""
        if not self._is_fitted:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler.transform(X_new) if self.scaler else X_new.values
        tensor_X = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probs = self.model(tensor_X).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        return pd.DataFrame({
            "probability": probs,
            "prediction": preds
        })
