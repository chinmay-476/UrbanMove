import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

# Configuration - resolve paths from project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
INPUT_FILE = os.path.join(PROJECT_ROOT, 'Cleaned_House_Rent_Dataset.csv')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'model_artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def perform_eda(df):
    print("Performing EDA...")
    stats = {
        "total_listings": len(df),
        "avg_rent": float(df['Rent'].mean()),
        "med_rent": float(df['Rent'].median()),
        "min_rent": float(df['Rent'].min()),
        "max_rent": float(df['Rent'].max()),
        "std_rent": float(df['Rent'].std()),
        "cities": sorted(df['City'].unique().tolist()) if 'City' in df.columns else [],
        "rent_by_city": df.groupby('City')['Rent'].mean().to_dict() if 'City' in df.columns else {},
        "rent_by_bhk": df.groupby('BHK')['Rent'].mean().to_dict() if 'BHK' in df.columns else {},
        "rent_by_furnishing": df.groupby('Furnishing Status')['Rent'].mean().to_dict() if 'Furnishing Status' in df.columns else {},
        "rent_by_area_type": df.groupby('Area Type')['Rent'].mean().to_dict() if 'Area Type' in df.columns else {},
        "total_cities": len(df['City'].unique()) if 'City' in df.columns else 0
    }
    
    # Save stats for Web App to consume
    with open(os.path.join(ARTIFACTS_DIR, 'eda_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    print(" EDA stats saved.")

def train_clustering_model(df):
    print("Training Spatial Clustering Model (Affordability Zones)...")
    # Features for clustering: Location + Price + Livability
    features = ['Latitude', 'Longitude', 'Rent', 'Neighborhood_Livability_Score']
    X = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df['Cluster_ID'] = clusters
    
    # Save clustering model and scaler
    joblib.dump(kmeans, os.path.join(ARTIFACTS_DIR, 'kmeans_model.pkl'))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'clustering_scaler.pkl'))
    print(" Clustering model saved.")
    return df

def train_price_prediction_model(df):
    print("Training Rental Price Prediction Model...")
    
    # Features - Include Area Type
    numeric_features = ['BHK', 'Size', 'Bathroom', 'Latitude', 'Longitude']
    categorical_features = ['City', 'Furnishing Status', 'Tenant Preferred', 'Bathroom_Type']
    
    # Add Area Type if it exists
    if 'Area Type' in df.columns:
        categorical_features.append('Area Type')
        print(f" Including 'Area Type' in features: {df['Area Type'].unique()}")
    
    # Ensure all required columns exist
    required_cols = numeric_features + categorical_features + ['Rent']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f" Warning: Missing columns {missing_cols}, attempting to handle...")
        for col in missing_cols:
            if col == 'Rent':
                raise ValueError("'Rent' column is required but missing!")
            elif col in categorical_features:
                df[col] = 'Unknown'
            else:
                df[col] = 0
    
    X = df[numeric_features + categorical_features]
    y = df['Rent']
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Model Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f" Model Performance -> MAE: {mae:.2f}, R2 Score: {r2:.2f}")
    
    # Save Model
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'price_prediction_model.pkl'))
    print(" Price prediction model saved.")

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error: {e}")
        return

    perform_eda(df)
    
    df = train_clustering_model(df)
    
    train_price_prediction_model(df)
    
    # Save DataFrame with clusters for the Web App
    output_file = os.path.join(ARTIFACTS_DIR, 'clustered_data.csv')
    df.to_csv(output_file, index=False)
    print(f"Final data with clusters saved to {output_file}")

if __name__ == "__main__":
    main()
