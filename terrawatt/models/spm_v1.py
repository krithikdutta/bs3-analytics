import geopandas as gpd
import pandas as pd
import libpysal
from libpysal.weights import Queen, KNN
from spreg import Probit
from shapely.geometry import Point
from sklearn.impute import SimpleImputer

def fit_spatial_probit(x_file_path, y_file_path, knn_k=16):
    """
    Fits a spatial probit model using the provided data files.
    
    Args:
        x_file_path (str): Path to the features CSV file
        y_file_path (str): Path to the target CSV file
        knn_k (int): Number of nearest neighbors for KNN weights (default: 16)
    
    Returns:
        model: Fitted Probit model
    """
    # 1. Load X and Y files
    X_df = pd.read_csv(x_file_path)
    Y_df = pd.read_csv(y_file_path)
    
    # Merge on grid_id
    data = pd.merge(X_df, Y_df, on="id")
    
    # 2. Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(data["Longitude"], data["Lattitude"])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")
    
    # 3. Build KNN weights
    W = KNN.from_dataframe(gdf, k=knn_k)
    W.transform = "r"
    
    # 4. Prepare features and target
    drop_cols = ["id", "Lattitude", "Longitude", "left_y", "top_y", "right_y", "bottom_y", 
                 "left_x", "top_x", "right_x", "bottom_x", "ndbi_median", "ntl_median", 
                 "ntl_sd", "ghi_median", "ghi_sd", "roadnet_len", "roadnet_count",
                 "row_index", "col_index", "road_density", "poi_count", "BSS", "num_bss"]
    
    X_cols = [c for c in data.columns if c not in drop_cols]
    X = data[X_cols].values
    y = data["num_bss"].values.reshape(-1, 1)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 5. Fit Spatial Probit
    model = Probit(y, X_imputed, w=W, name_y="BSS", name_x=X_cols, name_w="queen")
    
    return model

def predict_spatial_probit(model, imputer, X_cols, new_data_path):
    """
    Make predictions using the fitted spatial probit model.
    
    Args:
        model: Fitted Probit model
        imputer: Fitted SimpleImputer
        X_cols: List of feature column names
        new_data_path (str): Path to new data CSV file
    
    Returns:
        DataFrame: Predictions with original data
    """
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    # Create geometry points
    geometry = [Point(xy) for xy in zip(new_data["Longitude"], new_data["Lattitude"])]
    new_gdf = gpd.GeoDataFrame(new_data, geometry=geometry, crs="EPSG:4326")
    
    # Prepare features
    X_new = pd.DataFrame()
    for col in X_cols:
        if col in new_data.columns:
            X_new[col] = new_data[col]
        else:
            X_new[col] = 0
            print(f"Warning: Column {col} not found. Using 0.")
    
    # Impute missing values
    X_new_imputed = imputer.transform(X_new)
    
    # Create spatial weights
    W_new = KNN.from_dataframe(new_gdf, k=min(16, len(new_gdf)-1))
    W_new.transform = "r"
    
    # Make predictions
    predictions = model.predict(X_new_imputed)
    
    # Add predictions to results
    results_df = new_data.copy()
    results_df['predicted_prob'] = predictions
    results_df['predicted_class'] = (predictions > 0.5).astype(int)
    
    return results_df

# Example usage:
if __name__ == "__main__":
    x_file = "data/BLR_X_normalized.csv"
    y_file = "data/BLR_Y.csv"
    
    # Fit model
    model = fit_spatial_probit(x_file, y_file)
    print("Model Summary:")
    print(model.summary)
    
    # Make predictions
    try:
        predictions = predict_spatial_probit(
            model=model,
            imputer=imputer,
            X_cols=X_cols,
            new_data_path="data/RPR_X.csv"
        )
        
        predictions.to_csv("predictions.csv", index=False)
        print("\nPrediction Results:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Class distribution: {predictions['predicted_class'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")