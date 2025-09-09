# --- Import Statements ---
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
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from libpysal.weights import Queen, KNN
from libpysal.weights import lag_spatial
from esda.moran import Moran

warnings.filterwarnings("ignore")

# --- Function Definitions ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_config):
    log_dir = log_config['logdir']
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_config['logpath'])
    log_level = getattr(logging, log_config['level'].upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

# --- Main Script ---
CONFIG_PATH = "config.yaml"
config = load_config(CONFIG_PATH)
setup_logging(config['logging'])

logging.info("Starting SLX model training script...")
logging.info(f"Configuration loaded from {CONFIG_PATH}")

# --- 3) LOAD AND VALIDATE DATA ---
data_cfg = config['data']
out_cfg = config['output']

os.makedirs(out_cfg['outdir'], exist_ok=True)

logging.info("Loading data...")
X_df = pd.read_csv(os.path.join(data_cfg['datadir'], data_cfg['xtrain_path']))
Y_df = pd.read_csv(os.path.join(data_cfg['datadir'], data_cfg['ytrain_path']))
logging.info(f"X data shape: {X_df.shape}, Y data shape: {Y_df.shape}")

# --- Data Validation ---
def validate_columns(df, cols, df_name):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} not found in {df_name}.")

validate_columns(X_df, [data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors'] + data_cfg['bbox_cols'], "X CSV")
validate_columns(Y_df, [data_cfg['id_field'], data_cfg['target']] + data_cfg['bbox_cols'], "Y CSV")

# Merge data
df = Y_df.merge(X_df[[data_cfg['id_field']] + data_cfg['coord_cols'] + data_cfg['predictors']], on=data_cfg['id_field'], how="inner")
logging.info(f"Merged data shape: {df.shape}")
if len(df) == 0:
    raise ValueError("Merge resulted in an empty DataFrame. Check ID fields.")

# Create binary target variable
y = (df[data_cfg['target']] > 0).astype(int).values.reshape(-1, 1)
logging.info(f"Binary target distribution (0s, 1s): {np.bincount(y.ravel())}")

# --- 4) FEATURE ENGINEERING & SPATIAL WEIGHTS ---
feature_cfg = config['feature']

logging.info("Building spatial weights matrix...")
polygons = [Polygon([(r[data_cfg['bbox_cols'][0]], r[data_cfg['bbox_cols'][3]]),
                     (r[data_cfg['bbox_cols'][2]], r[data_cfg['bbox_cols'][3]]),
                     (r[data_cfg['bbox_cols'][2]], r[data_cfg['bbox_cols'][1]]),
                     (r[data_cfg['bbox_cols'][0]], r[data_cfg['bbox_cols'][1]])])
            for _, r in df.iterrows()]
temp_gdf = gpd.GeoDataFrame(df, geometry=polygons)

if feature_cfg['contiguity'] == 'queen':
    w = Queen.from_dataframe(temp_gdf)
elif feature_cfg['contiguity'] == 'knn':
    centroids = temp_gdf.geometry.centroid
    coords_knn = np.vstack([centroids.x.values, centroids.y.values]).T
    w = KNN.from_array(coords_knn, k=feature_cfg['n_neighbors'])
else:
    raise ValueError(f"Unsupported contiguity type: {feature_cfg['contiguity']}")

w.transform = "r"
logging.info(f"Spatial weights created ({feature_cfg['contiguity']}): {w.n} obs, {w.s0:.2f} total weights")
if w.islands:
    logging.warning(f"{len(w.islands)} observations have no neighbors (islands).")

# Create feature matrix (X) and spatially lagged features (WX)
logging.info("Creating spatially lagged features...")
X_raw = df[data_cfg['predictors']].copy()

# Impute missing values
if feature_cfg['imputer'] != 'none':
    for col in X_raw.columns:
        if X_raw[col].isna().any():
            if feature_cfg['imputer'] == 'median':
                fill_value = X_raw[col].median()
            elif feature_cfg['imputer'] == 'mean':
                fill_value = X_raw[col].mean()
            X_raw[col] = X_raw[col].fillna(fill_value)
            logging.info(f"Filled {X_raw[col].isna().sum()} NaNs in '{col}' using {feature_cfg['imputer']}.")

# Create spatially lagged features
X_lag = pd.DataFrame({f"W_{col}": lag_spatial(w, X_raw[col].values) for col in X_raw.columns})
X_all = pd.concat([X_raw, X_lag], axis=1)
logging.info(f"Final feature matrix shape: {X_all.shape}")

# --- 5) SPATIAL CROSS-VALIDATION SETUP ---
train_cfg = config['training']
coords = df[data_cfg['coord_cols']].values

logging.info(f"Creating {train_cfg['scv_folds']} spatial folds using KMeans clustering...")
kmeans = KMeans(n_clusters=train_cfg['scv_folds'], random_state=42, n_init='auto')
fold_ids = kmeans.fit_predict(coords)
logging.info(f"Fold sizes: {np.bincount(fold_ids)}")

# --- 6) MODEL TRAINING AND EVALUATION ---
model_cfg = config['model']

# Define scaling step from config
if feature_cfg['scaling'] == 'standard':
    scaler = StandardScaler()
elif feature_cfg['scaling'] == 'minmax':
    scaler = MinMaxScaler()
else: # 'none'
    scaler = None

# Build pipeline
pipeline_steps = []
if scaler:
    pipeline_steps.append(("scaler", scaler))
pipeline_steps.append(("logit", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")))
pipe = Pipeline(pipeline_steps)

records, coef_records = [], []
all_residuals = np.zeros(len(df))

for fold in range(train_cfg['scv_folds']):
    logging.info(f"Processing fold {fold + 1}/{train_cfg['scv_folds']}...")
    train_idx, test_idx = np.where(fold_ids != fold)[0], np.where(fold_ids == fold)[0]
    X_train, X_test = X_all.iloc[train_idx], X_all.iloc[test_idx]
    y_train, y_test = y[train_idx].ravel(), y[test_idx].ravel()

    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= model_cfg['threshold']).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else np.nan
    acc = accuracy_score(y_test, pred)
    logging.info(f"  Fold {fold + 1}: AUC={auc:.4f}, Accuracy={acc:.4f}")
    
    # Store results and residuals
    all_residuals[test_idx] = y_test - prob
    for i, idx in enumerate(test_idx):
        records.append({data_cfg['id_field']: df.iloc[idx][data_cfg['id_field']], "fold": fold,
                        "y_true": int(y_test[i]), "y_prob": float(prob[i]), "y_pred": int(pred[i])})

    # Store coefficients
    coef = pipe.named_steps["logit"].coef_.ravel()
    for f, c in zip(X_all.columns, coef):
        coef_records.append({"fold": fold, "feature": f, "coef": c})

# --- 7) POST-PROCESSING AND REPORTING ---
pred_df = pd.DataFrame.from_records(records)
coef_df = pd.DataFrame.from_records(coef_records)

logging.info("Computing spatial autocorrelation of residuals...")
mi = Moran(all_residuals, w)
logging.info(f"Moran's I on residuals: {mi.I:.4f} (p-value: {mi.p_norm:.4f})")

logging.info("Training final model on full dataset...")
pipe.fit(X_all, y.ravel())

# --- 8) SAVE OUTPUTS ---
logging.info("Saving results to output directory...")
joblib.dump(pipe, os.path.join(out_cfg['outdir'], out_cfg['model']))
joblib.dump(w, os.path.join(out_cfg['outdir'], out_cfg['weights']))

pred_df.to_csv(os.path.join(out_cfg['outdir'], out_cfg['predictions']), index=False)
coef_df.to_csv(os.path.join(out_cfg['outdir'], out_cfg['coefficients']), index=False)

# Write final report
with open(os.path.join(out_cfg['outdir'], out_cfg['report']), "w") as f:
    overall_auc = roc_auc_score(pred_df["y_true"], pred_df["y_prob"])
    overall_acc = accuracy_score(pred_df["y_true"], pred_df["y_pred"])
    
    f.write(f"Spatial Logit Model Report: {model_cfg['name']}\n")
    f.write("="*40 + "\n\n")
    f.write(f"Overall AUC: {overall_auc:.4f}\n")
    f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
    f.write(f"Spatial Autocorrelation of Residuals (Moran's I):\n")
    f.write(f"  I-value: {mi.I:.4f}\n")
    f.write(f"  p-value: {mi.p_norm:.4f}\n")
    if mi.p_norm < 0.05:
        f.write("  *** Significant spatial clustering detected in residuals. Model may be misspecified.\n")
    else:
        f.write("  No significant spatial autocorrelation in residuals.\n")
    
    avg_coefs = coef_df.groupby("feature")["coef"].mean().sort_values(key=abs, ascending=False)
    f.write("\nAverage Feature Coefficients:\n")
    f.write(avg_coefs.to_string())

logging.info("Script finished successfully.")