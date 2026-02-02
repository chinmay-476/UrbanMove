from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import json
import numpy as np
import random

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configuration - Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')
CLUSTERED_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'clustered_data.csv')
DATASET_PATH = os.path.join(BASE_DIR, 'Cleaned_House_Rent_Dataset.csv')

# Load Artifacts
print("Loading models...")
model = None
clustering_model = None
clustering_scaler = None

try:
    model_path = os.path.join(ARTIFACTS_DIR, 'price_prediction_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("[OK] Price prediction model loaded successfully.")
    else:
        print("[WARN] Warning: Price prediction model not found. Run scripts/train_models.py first.")
except Exception as e:
    print(f"[FAIL] Error loading price prediction model: {e}")

try:
    clustering_model_path = os.path.join(ARTIFACTS_DIR, 'kmeans_model.pkl')
    clustering_scaler_path = os.path.join(ARTIFACTS_DIR, 'clustering_scaler.pkl')
    if os.path.exists(clustering_model_path) and os.path.exists(clustering_scaler_path):
        clustering_model = joblib.load(clustering_model_path)
        clustering_scaler = joblib.load(clustering_scaler_path)
        print("[OK] Clustering models loaded successfully.")
except Exception as e:
    print(f"[WARN] Warning: Clustering models not found: {e}")

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        if not os.path.exists(os.path.join(ARTIFACTS_DIR, 'eda_stats.json')):
            return jsonify({'error': 'Stats file not found'}), 404
        with open(os.path.join(ARTIFACTS_DIR, 'eda_stats.json'), 'r') as f:
            stats = json.load(f)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/properties', methods=['GET'])
def get_properties():
    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({'error': 'Dataset file not found. Please run scripts/preprocess_data.py first.'}), 404
        
        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        
        # Get query parameters for filtering
        city = request.args.get('city')
        max_rent = request.args.get('max_rent', type=float)
        min_rent = request.args.get('min_rent', type=float)
        bhk = request.args.get('bhk', type=int)
        
        # Apply filters
        if city:
            df = df[df['City'] == city]
        if max_rent:
            df = df[df['Rent'] <= max_rent]
        if min_rent:
            df = df[df['Rent'] >= min_rent]
        if bhk:
            df = df[df['BHK'] == bhk]
        
        # Simple pagination or limit for performance
        limit = min(int(request.args.get('limit', 50)), 100)  # Max 100 records
        properties = df.sample(n=min(limit, len(df))).fillna('').to_dict(orient='records')
        
        # Add an ID field if not present (using index equivalent)
        for i, prop in enumerate(properties):
            if 'id' not in prop:
                prop['id'] = i 
                
        return jsonify({
            'properties': properties,
            'total': len(df),
            'returned': len(properties)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/map_data', methods=['GET'])
def get_map_data():
    try:
        # Load a subset of data for the map to avoid huge payload
        if not os.path.exists(CLUSTERED_DATA_PATH):
            # Fallback to main dataset if clustered data doesn't exist
            if not os.path.exists(DATASET_PATH):
                return jsonify({'error': 'Dataset file not found. Please run scripts/preprocess_data.py first.'}), 404
            df = pd.read_csv(DATASET_PATH)
            if 'Cluster_ID' not in df.columns:
                df['Cluster_ID'] = 0
            if 'Neighborhood_Livability_Score' not in df.columns:
                df['Neighborhood_Livability_Score'] = 5.0
        else:
            df = pd.read_csv(CLUSTERED_DATA_PATH)
        
        # Ensure required columns exist
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return jsonify({'error': 'Latitude/Longitude columns not found. Please run scripts/preprocess_data.py first.'}), 500
        
        # Select relevant columns (handle missing columns gracefully)
        required_cols = ['Latitude', 'Longitude', 'Rent', 'City']
        available_cols = [col for col in required_cols if col in df.columns]
        
        # Add optional columns for better info display
        optional_cols = ['Area Locality', 'BHK', 'Size', 'Furnishing Status', 'Area Type']
        for col in optional_cols:
            if col in df.columns:
                available_cols.append(col)
        
        if 'Cluster_ID' in df.columns:
            available_cols.append('Cluster_ID')
        if 'Neighborhood_Livability_Score' in df.columns:
            available_cols.append('Neighborhood_Livability_Score')
        
        # Sample data for performance (30% or max 1000 records)
        sample_size = min(int(len(df) * 0.3), 1000)
        if len(df) == 0:
            return jsonify([])
        map_data = df[available_cols].sample(n=min(sample_size, len(df))).fillna(0).to_dict(orient='records')
        return jsonify(map_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/properties/<int:prop_id>', methods=['GET'])
def get_property(prop_id):
    try:
        df = pd.read_csv(DATASET_PATH)
        # Assuming ID is index if not explicit
        if 0 <= prop_id < len(df):
            prop = df.iloc[prop_id].fillna('').to_dict()
            prop['id'] = prop_id
            return jsonify(prop)
        else:
            return jsonify({'error': 'Property not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """Get list of available cities and their localities"""
    try:
        if not os.path.exists(DATASET_PATH):
            return jsonify({'error': 'Dataset file not found. Please run scripts/preprocess_data.py first.'}), 404
        
        df = pd.read_csv(DATASET_PATH)
        cities = df['City'].unique().tolist() if 'City' in df.columns else []
        
        # Get localities by city
        localities_by_city = {}
        if 'Area Locality' in df.columns and 'City' in df.columns:
            for city in cities:
                localities = df[df['City'] == city]['Area Locality'].dropna().unique().tolist()
                localities_by_city[city] = sorted(localities[:50])  # Limit to 50 per city
        
        return jsonify({
            'cities': sorted(cities),
            'localities_by_city': localities_by_city
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_rent():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    
    try:
        # Helper function to get coordinates from city/locality
        def get_coordinates(city, locality=None):
            city_coords = {
                'Mumbai': (19.0760, 72.8777),
                'Bangalore': (12.9716, 77.5946),
                'Chennai': (13.0827, 80.2707),
                'Delhi': (28.7041, 77.1025),
                'Hyderabad': (17.3850, 78.4867),
                'Kolkata': (22.5726, 88.3639)
            }
            base_lat, base_lon = city_coords.get(city, (20.5937, 78.9629))
            # Add small jitter if locality is provided
            if locality:
                base_lat += random.uniform(-0.05, 0.05)
                base_lon += random.uniform(-0.05, 0.05)
            return base_lat, base_lon
        
        # Get coordinates if not provided
        city = data.get('city', 'Mumbai')
        locality = data.get('locality', '')
        if 'latitude' in data and 'longitude' in data:
            lat, lon = float(data.get('latitude')), float(data.get('longitude'))
        else:
            lat, lon = get_coordinates(city, locality)
        
        # Expected features: 'BHK', 'Size', 'Bathroom', 'Latitude', 'Longitude', 'City', 
        # 'Furnishing Status', 'Tenant Preferred', 'Bathroom_Type', 'Area Type'
        # Build input data with all required features
        input_data = pd.DataFrame([{
            'BHK': int(data.get('bhk', 1)),
            'Size': float(data.get('size', 500)),
            'Bathroom': int(data.get('bathroom', 1)),
            'Latitude': lat,
            'Longitude': lon,
            'City': city,
            'Furnishing Status': data.get('furnishing', 'Unfurnished'),
            'Tenant Preferred': data.get('tenant', 'Bachelors/Family'),
            'Bathroom_Type': data.get('bathroom_type', 'Standard'),
            'Area Type': data.get('area_type', 'Super Area')
        }])
        
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'predicted_rent': round(prediction, 2),
            'currency': 'INR',
            'location': {
                'city': city,
                'locality': locality,
                'coordinates': {'lat': lat, 'lon': lon}
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
