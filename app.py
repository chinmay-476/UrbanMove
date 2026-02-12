from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import json
import numpy as np
import random
from datetime import datetime

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configuration - Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')
CLUSTERED_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'clustered_data.csv')
DATASET_PATH = os.path.join(BASE_DIR, 'Cleaned_House_Rent_Dataset.csv')

# In-memory cache to avoid reading CSV from disk on every request.
_DATASET_CACHE = {
    'mtime': None,
    'df': None
}

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

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError('Dataset file not found. Please run scripts/preprocess_data.py first.')

    mtime = os.path.getmtime(DATASET_PATH)
    if _DATASET_CACHE['df'] is None or _DATASET_CACHE['mtime'] != mtime:
        _DATASET_CACHE['df'] = pd.read_csv(DATASET_PATH)
        _DATASET_CACHE['mtime'] = mtime

    return _DATASET_CACHE['df'].copy()

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
        # Load dataset
        df = load_dataset()
        
        # Get query parameters for filtering
        city = request.args.get('city')
        max_rent = request.args.get('max_rent', type=float)
        min_rent = request.args.get('min_rent', type=float)
        bhk = request.args.get('bhk', type=int)
        
        # Apply filters
        if city:
            df = df[df['City'].fillna('').str.lower() == city.lower()]
        if max_rent:
            df = df[df['Rent'] <= max_rent]
        if min_rent:
            df = df[df['Rent'] >= min_rent]
        if bhk:
            df = df[df['BHK'] == bhk]
        
        # Simple pagination or limit for performance
        limit = request.args.get('limit', default=50, type=int)
        if limit is None:
            limit = 50
        limit = max(1, min(limit, 100))
        properties_df = df.sample(n=min(limit, len(df))) if len(df) > 0 else df
        properties = properties_df.fillna('').to_dict(orient='records')
        
        # Preserve dataset row index as ID when available.
        for idx, prop in zip(properties_df.index.tolist(), properties):
            if 'id' not in prop:
                prop['id'] = int(idx)
                
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
        df = load_dataset()
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
        df = load_dataset()
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

@app.route('/api/budget_advisor', methods=['GET'])
def get_budget_advisor():
    """
    Rank localities by budget-fit + livability to help users shortlist areas.
    Query params:
      - budget (required): user budget in INR
      - city (optional)
      - bhk (optional)
      - limit (optional, default 5, max 20)
    """
    try:
        budget = request.args.get('budget', type=float)
        if budget is None or budget <= 0:
            return jsonify({'error': "A positive 'budget' query parameter is required."}), 400

        city = request.args.get('city')
        bhk = request.args.get('bhk', type=int)
        limit = request.args.get('limit', default=5, type=int)
        if limit is None:
            limit = 5
        limit = max(1, min(limit, 20))

        df = load_dataset()
        if city:
            df = df[df['City'].fillna('').str.lower() == city.lower()]
        if bhk:
            df = df[df['BHK'] == bhk]

        if df.empty:
            return jsonify({
                'budget': budget,
                'city': city,
                'bhk': bhk,
                'recommendations': [],
                'total_candidates': 0
            })

        df = df.copy()
        if 'Area Locality' not in df.columns:
            df['Area Locality'] = df['City'].fillna('Unknown locality')
        else:
            df['Area Locality'] = df['Area Locality'].fillna('').replace('', 'Unknown locality')

        if 'Neighborhood_Livability_Score' not in df.columns:
            df['Neighborhood_Livability_Score'] = 5.0
        df['Neighborhood_Livability_Score'] = pd.to_numeric(
            df['Neighborhood_Livability_Score'],
            errors='coerce'
        ).fillna(5.0)

        grouped = (
            df.dropna(subset=['Rent'])
              .groupby('Area Locality', as_index=False)
              .agg(
                  city=('City', lambda s: s.mode().iloc[0] if not s.mode().empty else (s.iloc[0] if len(s) else 'N/A')),
                  avg_rent=('Rent', 'mean'),
                  median_rent=('Rent', 'median'),
                  livability=('Neighborhood_Livability_Score', 'mean'),
                  listings=('Rent', 'count')
              )
        )

        if grouped.empty:
            return jsonify({
                'budget': budget,
                'city': city,
                'bhk': bhk,
                'recommendations': [],
                'total_candidates': 0
            })

        grouped['affordability_score'] = (1 - (grouped['avg_rent'] - budget).abs() / max(budget, 1)).clip(0, 1)
        grouped['livability_score'] = (grouped['livability'] / 10.0).clip(0, 1)
        grouped['listing_score'] = (grouped['listings'] / max(grouped['listings'].max(), 1)).clip(0, 1)
        grouped['match_score'] = (
            0.55 * grouped['affordability_score'] +
            0.35 * grouped['livability_score'] +
            0.10 * grouped['listing_score']
        )

        total_candidates = len(grouped)
        grouped = grouped.sort_values(by=['match_score', 'listings'], ascending=[False, False]).head(limit)

        recommendations = []
        for _, row in grouped.iterrows():
            avg_rent = float(row['avg_rent'])
            budget_gap = avg_rent - budget
            recommendations.append({
                'locality': row['Area Locality'],
                'city': row['city'],
                'avg_rent': round(avg_rent, 2),
                'median_rent': round(float(row['median_rent']), 2),
                'livability': round(float(row['livability']), 2),
                'listings': int(row['listings']),
                'match_score': round(float(row['match_score']) * 100, 1),
                'budget_gap': round(float(budget_gap), 2),
                'savings_if_under_budget': round(float(max(0, -budget_gap)), 2),
                'over_budget_by': round(float(max(0, budget_gap)), 2)
            })

        return jsonify({
            'budget': budget,
            'city': city,
            'bhk': bhk,
            'recommendations': recommendations,
            'total_candidates': total_candidates
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_insights', methods=['GET'])
def get_market_insights():
    """
    Compare a predicted rent against similar market listings.
    Query params:
      - predicted_rent (required)
      - city (optional)
      - bhk (optional)
    """
    try:
        predicted_rent = request.args.get('predicted_rent', type=float)
        if predicted_rent is None or predicted_rent <= 0:
            return jsonify({'error': "A positive 'predicted_rent' query parameter is required."}), 400

        city = request.args.get('city')
        bhk = request.args.get('bhk', type=int)

        df = load_dataset()
        if city:
            df = df[df['City'].fillna('').str.lower() == city.lower()]
        if bhk:
            df = df[df['BHK'] == bhk]

        if 'Rent' not in df.columns:
            return jsonify({'error': "'Rent' column is missing in dataset."}), 500

        rents = pd.to_numeric(df['Rent'], errors='coerce').dropna()
        if rents.empty:
            return jsonify({
                'predicted_rent': predicted_rent,
                'city': city,
                'bhk': bhk,
                'market_size': 0
            })

        avg_rent = float(rents.mean())
        median_rent = float(rents.median())
        p25, p75 = np.percentile(rents, [25, 75])
        iqr = float(p75 - p25)
        percentile = float((rents <= predicted_rent).mean() * 100.0)

        below_count = int((rents < predicted_rent).sum())
        above_count = int((rents > predicted_rent).sum())
        equal_count = int((rents == predicted_rent).sum())

        if predicted_rent < p25:
            position_label = 'Below market'
        elif predicted_rent > p75:
            position_label = 'Above market'
        else:
            position_label = 'Within market range'

        if percentile <= 30:
            recommendation = 'Competitive for tenants (lower than most comparable listings).'
        elif percentile >= 70:
            recommendation = 'Premium segment (higher than most comparable listings).'
        else:
            recommendation = 'Balanced segment (close to typical market pricing).'

        return jsonify({
            'predicted_rent': round(predicted_rent, 2),
            'city': city,
            'bhk': bhk,
            'market_size': int(len(rents)),
            'avg_rent': round(avg_rent, 2),
            'median_rent': round(median_rent, 2),
            'p25_rent': round(float(p25), 2),
            'p75_rent': round(float(p75), 2),
            'iqr_rent': round(iqr, 2),
            'percentile': round(percentile, 1),
            'below_count': below_count,
            'above_count': above_count,
            'equal_count': equal_count,
            'position_label': position_label,
            'recommendation': recommendation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rent_trends', methods=['GET'])
def get_rent_trends():
    """
    Return historical monthly rent trend and a simple linear forecast.
    Query params:
      - city (optional)
      - bhk (optional)
      - locality (optional, partial match)
      - months_history (optional, default 12, max 36)
      - months_forecast (optional, default 6, max 12)
      - predicted_rent (optional, used for comparison text)
    """
    try:
        city = request.args.get('city')
        bhk = request.args.get('bhk', type=int)
        locality = request.args.get('locality')
        months_history = request.args.get('months_history', default=12, type=int)
        months_forecast = request.args.get('months_forecast', default=6, type=int)
        predicted_rent = request.args.get('predicted_rent', type=float)

        if months_history is None:
            months_history = 12
        if months_forecast is None:
            months_forecast = 6
        months_history = max(3, min(months_history, 36))
        months_forecast = max(1, min(months_forecast, 12))

        df = load_dataset()
        if city:
            df = df[df['City'].fillna('').str.lower() == city.lower()]
        if bhk:
            df = df[df['BHK'] == bhk]
        if locality and 'Area Locality' in df.columns:
            df = df[df['Area Locality'].fillna('').str.contains(locality, case=False, regex=False)]

        if 'Rent' not in df.columns:
            return jsonify({'error': "Required column 'Rent' not found."}), 500

        trend_df = df.copy()
        trend_df['Rent'] = pd.to_numeric(trend_df['Rent'], errors='coerce')

        # Try to infer a posted-date column; fallback to synthetic timeline if unavailable.
        date_series = None
        date_source = 'synthetic'
        date_candidates = [col for col in trend_df.columns if 'posted' in str(col).strip().lower()]
        for candidate in date_candidates:
            parsed = pd.to_datetime(trend_df[candidate], errors='coerce')
            if parsed.notna().sum() >= max(5, int(0.03 * len(trend_df))):
                date_series = parsed
                date_source = 'posted_on'
                break

        if date_series is not None:
            trend_df['_trend_date'] = date_series
        else:
            n_rows = len(trend_df)
            if n_rows == 0:
                trend_df['_trend_date'] = pd.NaT
            else:
                span_months = min(max(months_history + months_forecast, 12), 24)
                start_month = pd.Timestamp.today().to_period('M').to_timestamp() - pd.DateOffset(months=span_months - 1)
                month_idx = (np.arange(n_rows) * span_months // max(n_rows, 1)).astype(int)
                trend_df['_trend_date'] = [start_month + pd.DateOffset(months=int(v)) for v in month_idx]

        trend_df = trend_df.dropna(subset=['Rent', '_trend_date'])

        if trend_df.empty:
            return jsonify({
                'city': city,
                'bhk': bhk,
                'locality': locality,
                'market_size': 0,
                'historical': [],
                'forecast': [],
                'trend_direction': 'unknown',
                'avg_monthly_change': 0.0
            })

        trend_df['Month'] = trend_df['_trend_date'].dt.to_period('M').dt.to_timestamp()
        monthly = (
            trend_df.groupby('Month', as_index=False)
                    .agg(
                        avg_rent=('Rent', 'mean'),
                        median_rent=('Rent', 'median'),
                        listings=('Rent', 'count')
                    )
                    .sort_values('Month')
        )

        if monthly.empty:
            return jsonify({
                'city': city,
                'bhk': bhk,
                'locality': locality,
                'market_size': 0,
                'historical': [],
                'forecast': [],
                'trend_direction': 'unknown',
                'avg_monthly_change': 0.0
            })

        monthly = monthly.tail(months_history).reset_index(drop=True)
        y = monthly['median_rent'].values.astype(float)
        x = np.arange(len(y), dtype=float)

        if len(y) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope = 0.0
            intercept = float(y[0])

        slope = float(slope)
        intercept = float(intercept)

        last_month = monthly['Month'].iloc[-1]
        forecast = []
        for i in range(1, months_forecast + 1):
            x_future = len(y) - 1 + i
            forecast_rent = max(0.0, slope * x_future + intercept)
            forecast_month = (last_month + pd.DateOffset(months=i)).strftime('%Y-%m')
            forecast.append({
                'month': forecast_month,
                'forecast_rent': round(float(forecast_rent), 2)
            })

        historical = [{
            'month': row['Month'].strftime('%Y-%m'),
            'avg_rent': round(float(row['avg_rent']), 2),
            'median_rent': round(float(row['median_rent']), 2),
            'listings': int(row['listings'])
        } for _, row in monthly.iterrows()]

        avg_monthly_change = round(slope, 2)
        if avg_monthly_change > 500:
            trend_direction = 'upward'
        elif avg_monthly_change < -500:
            trend_direction = 'downward'
        else:
            trend_direction = 'stable'

        next_month_forecast = forecast[0]['forecast_rent'] if forecast else float(y[-1])
        predicted_vs_next_month = None
        if predicted_rent is not None:
            predicted_vs_next_month = round(float(predicted_rent - next_month_forecast), 2)

        return jsonify({
            'city': city,
            'bhk': bhk,
            'locality': locality,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'timeline_source': date_source,
            'market_size': int(len(trend_df)),
            'months_history': months_history,
            'months_forecast': months_forecast,
            'trend_direction': trend_direction,
            'avg_monthly_change': avg_monthly_change,
            'next_month_forecast': round(float(next_month_forecast), 2),
            'predicted_vs_next_month': predicted_vs_next_month,
            'historical': historical,
            'forecast': forecast
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
