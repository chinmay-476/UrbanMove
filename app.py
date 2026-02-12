from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
import json
import numpy as np
import random
from datetime import datetime
import subprocess
import shutil
import sys
from math import radians, sin, cos, sqrt, atan2

# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configuration - Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')
CLUSTERED_DATA_PATH = os.path.join(ARTIFACTS_DIR, 'clustered_data.csv')
DATASET_PATH = os.path.join(BASE_DIR, 'Cleaned_House_Rent_Dataset.csv')
ALERTS_PATH = os.path.join(ARTIFACTS_DIR, 'alerts.json')
MODEL_METRICS_HISTORY_PATH = os.path.join(ARTIFACTS_DIR, 'model_metrics_history.json')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'price_prediction_model.pkl')

# In-memory cache to avoid reading CSV from disk on every request.
_DATASET_CACHE = {
    'mtime': None,
    'df': None
}

CITY_COORDS = {
    'Mumbai': (19.0760, 72.8777),
    'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Delhi': (28.7041, 77.1025),
    'Hyderabad': (17.3850, 78.4867),
    'Kolkata': (22.5726, 88.3639)
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

def _safe_json_read(path, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def _safe_json_write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

def _get_city_coords(city):
    return CITY_COORDS.get(city, (20.5937, 78.9629))

def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c

def _build_model_input(data):
    city = data.get('city', 'Mumbai')
    locality = data.get('locality', '')

    if 'latitude' in data and 'longitude' in data:
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))
    else:
        base_lat, base_lon = _get_city_coords(city)
        # Keep deterministic jitter for same locality so repeated calls are stable.
        if locality:
            seed = abs(hash(locality)) % 10_000
            rng = random.Random(seed)
            lat = base_lat + rng.uniform(-0.05, 0.05)
            lon = base_lon + rng.uniform(-0.05, 0.05)
        else:
            lat, lon = base_lat, base_lon

    input_df = pd.DataFrame([{
        'BHK': int(data.get('bhk', 1)),
        'Size': float(data.get('size', 500)),
        'Bathroom': int(data.get('bathroom', 1)),
        'Latitude': float(lat),
        'Longitude': float(lon),
        'City': city,
        'Furnishing Status': data.get('furnishing', 'Unfurnished'),
        'Tenant Preferred': data.get('tenant', 'Bachelors/Family'),
        'Bathroom_Type': data.get('bathroom_type', 'Standard'),
        'Area Type': data.get('area_type', 'Super Area')
    }])

    return input_df, city, locality, lat, lon

def _evaluate_model_mae(model_obj, sample_size=800):
    try:
        df = load_dataset()
        req_cols = [
            'BHK', 'Size', 'Bathroom', 'Latitude', 'Longitude', 'City',
            'Furnishing Status', 'Tenant Preferred', 'Bathroom_Type', 'Area Type', 'Rent'
        ]
        available_cols = [c for c in req_cols if c in df.columns]
        if len(available_cols) < len(req_cols):
            return None

        eval_df = df[req_cols].copy()
        eval_df['Rent'] = pd.to_numeric(eval_df['Rent'], errors='coerce')
        eval_df = eval_df.dropna(subset=req_cols)
        if eval_df.empty:
            return None

        if len(eval_df) > sample_size:
            eval_df = eval_df.sample(n=sample_size, random_state=42)

        y_true = eval_df['Rent'].values
        X = eval_df.drop(columns=['Rent'])
        y_pred = model_obj.predict(X)
        return float(np.mean(np.abs(y_true - y_pred)))
    except Exception:
        return None

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

@app.route('/api/what_if', methods=['POST'])
def what_if_simulator():
    """Run multi-scenario predictions for what-if analysis."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        payload = request.json or {}
        base = payload.get('base', payload)
        base_bhk = int(base.get('bhk', 2))
        base_size = float(base.get('size', 850))
        base_bath = int(base.get('bathroom', 2))

        bhk_opts = payload.get('bhk_options') or [max(1, base_bhk - 1), base_bhk, base_bhk + 1]
        size_opts = payload.get('size_options') or [max(150, base_size - 200), base_size, base_size + 200]
        bath_opts = payload.get('bathroom_options') or [max(1, base_bath - 1), base_bath, base_bath + 1]
        furnishing_opts = payload.get('furnishing_options') or [
            base.get('furnishing', 'Semi-Furnished'), 'Unfurnished', 'Furnished'
        ]
        area_opts = payload.get('area_type_options') or [
            base.get('area_type', 'Super Area'), 'Carpet Area', 'Built Area'
        ]

        scenario_rows = []

        def add_scenario(label, modified):
            input_df, city, locality, lat, lon = _build_model_input(modified)
            pred = float(model.predict(input_df)[0])
            scenario_rows.append({
                'scenario': label,
                'predicted_rent': round(pred, 2),
                'city': city,
                'locality': locality,
                'bhk': int(modified.get('bhk', base_bhk)),
                'size': float(modified.get('size', base_size)),
                'bathroom': int(modified.get('bathroom', base_bath)),
                'furnishing': modified.get('furnishing', base.get('furnishing', 'Semi-Furnished')),
                'area_type': modified.get('area_type', base.get('area_type', 'Super Area')),
                'latitude': round(float(lat), 5),
                'longitude': round(float(lon), 5)
            })

        # Baseline
        add_scenario('Baseline', dict(base))

        # One-dimensional sweeps
        for v in sorted(set(int(x) for x in bhk_opts if str(x).strip() != '')):
            modified = dict(base)
            modified['bhk'] = v
            add_scenario(f'BHK={v}', modified)

        for v in sorted(set(float(x) for x in size_opts if str(x).strip() != '')):
            modified = dict(base)
            modified['size'] = v
            add_scenario(f'Size={int(v)} sqft', modified)

        for v in sorted(set(int(x) for x in bath_opts if str(x).strip() != '')):
            modified = dict(base)
            modified['bathroom'] = v
            add_scenario(f'Bathrooms={v}', modified)

        for v in sorted(set(str(x) for x in furnishing_opts if str(x).strip() != '')):
            modified = dict(base)
            modified['furnishing'] = v
            add_scenario(f'Furnishing={v}', modified)

        for v in sorted(set(str(x) for x in area_opts if str(x).strip() != '')):
            modified = dict(base)
            modified['area_type'] = v
            add_scenario(f'Area Type={v}', modified)

        # Deduplicate scenarios by label
        seen = set()
        deduped = []
        for item in scenario_rows:
            key = item['scenario']
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        deduped_sorted = sorted(deduped, key=lambda x: x['predicted_rent'])
        baseline_rent = next((x['predicted_rent'] for x in deduped if x['scenario'] == 'Baseline'), None)
        for row in deduped_sorted:
            if baseline_rent is not None:
                row['delta_vs_baseline'] = round(row['predicted_rent'] - baseline_rent, 2)

        return jsonify({
            'baseline_rent': baseline_rent,
            'scenario_count': len(deduped_sorted),
            'scenarios': deduped_sorted
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/similar_listings', methods=['POST'])
def get_similar_listings():
    """Find top similar listings for a given input profile."""
    try:
        payload = request.json or {}
        limit = int(payload.get('limit', 5))
        limit = max(1, min(limit, 10))
        base_bhk = int(payload.get('bhk', 2))
        base_size = float(payload.get('size', 850))
        base_bath = int(payload.get('bathroom', 2))
        city = payload.get('city')
        furnishing = payload.get('furnishing', '')
        area_type = payload.get('area_type', '')
        tenant = payload.get('tenant', '')

        df = load_dataset()
        if city:
            df = df[df['City'].fillna('').str.lower() == city.lower()]
        if df.empty:
            return jsonify({'similar': [], 'total_candidates': 0})

        df = df.copy()
        df['Rent'] = pd.to_numeric(df.get('Rent'), errors='coerce')
        df['BHK'] = pd.to_numeric(df.get('BHK'), errors='coerce')
        df['Size'] = pd.to_numeric(df.get('Size'), errors='coerce')
        df['Bathroom'] = pd.to_numeric(df.get('Bathroom'), errors='coerce')
        df = df.dropna(subset=['Rent', 'BHK', 'Size', 'Bathroom'])
        if df.empty:
            return jsonify({'similar': [], 'total_candidates': 0})

        size_scale = max(base_size, 1.0)
        score_rows = []
        for idx, row in df.iterrows():
            numeric_distance = (
                abs(float(row['BHK']) - base_bhk) / max(base_bhk, 1.0) +
                abs(float(row['Bathroom']) - base_bath) / max(base_bath, 1.0) +
                abs(float(row['Size']) - base_size) / size_scale
            )
            cat_match = 0
            if furnishing and str(row.get('Furnishing Status', '')).lower() == furnishing.lower():
                cat_match += 1
            if area_type and str(row.get('Area Type', '')).lower() == area_type.lower():
                cat_match += 1
            if tenant and str(row.get('Tenant Preferred', '')).lower() == tenant.lower():
                cat_match += 1

            raw_score = 100 - (numeric_distance * 28) + (cat_match * 6)
            match_score = max(0.0, min(100.0, raw_score))
            score_rows.append((idx, match_score))

        score_rows.sort(key=lambda x: x[1], reverse=True)
        top_ids = [i for i, _ in score_rows[:limit]]

        top = []
        for rank, (idx, score) in enumerate(score_rows[:limit], start=1):
            row = df.loc[idx]
            top.append({
                'rank': rank,
                'id': int(idx),
                'match_score': round(float(score), 1),
                'city': row.get('City', ''),
                'locality': row.get('Area Locality', ''),
                'rent': round(float(row.get('Rent', 0)), 2),
                'bhk': int(row.get('BHK', 0)),
                'size': float(row.get('Size', 0)),
                'bathroom': int(row.get('Bathroom', 0)),
                'furnishing': row.get('Furnishing Status', ''),
                'area_type': row.get('Area Type', '')
            })

        return jsonify({
            'total_candidates': int(len(df)),
            'selected_ids': top_ids,
            'similar': top
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/price_intelligence', methods=['GET'])
def get_price_intelligence():
    """Tag listings as underpriced/fair/overpriced based on model residuals."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        city = request.args.get('city')
        bhk = request.args.get('bhk', type=int)
        limit = request.args.get('limit', default=50, type=int)
        if limit is None:
            limit = 50
        limit = max(10, min(limit, 150))

        df = load_dataset()
        if city:
            df = df[df['City'].fillna('').str.lower() == city.lower()]
        if bhk:
            df = df[df['BHK'] == bhk]

        req_cols = [
            'BHK', 'Size', 'Bathroom', 'Latitude', 'Longitude', 'City',
            'Furnishing Status', 'Tenant Preferred', 'Bathroom_Type', 'Area Type', 'Rent'
        ]
        missing_cols = [c for c in req_cols if c not in df.columns]
        if missing_cols:
            return jsonify({'error': f"Missing columns: {missing_cols}"}), 500

        work = df[req_cols].copy()
        work['Rent'] = pd.to_numeric(work['Rent'], errors='coerce')
        work = work.dropna(subset=req_cols)
        if work.empty:
            return jsonify({'summary': {}, 'records': []})

        if len(work) > limit:
            work = work.sample(n=limit, random_state=42)

        preds = model.predict(work.drop(columns=['Rent']))
        work['Predicted_Rent'] = preds
        work['Residual'] = work['Rent'] - work['Predicted_Rent']
        work['Residual_Pct'] = np.where(
            work['Predicted_Rent'] == 0,
            0.0,
            (work['Residual'] / work['Predicted_Rent']) * 100.0
        )

        def tag_row(pct):
            if pct <= -15:
                return 'Underpriced'
            if pct >= 15:
                return 'Overpriced'
            return 'Fair'

        work['Price_Tag'] = work['Residual_Pct'].apply(tag_row)
        counts = work['Price_Tag'].value_counts().to_dict()

        work = work.assign(id=work.index.astype(int))
        records = []
        for _, row in work.sort_values(by='Residual_Pct').iterrows():
            records.append({
                'id': int(row['id']),
                'city': row.get('City', ''),
                'rent': round(float(row['Rent']), 2),
                'predicted_rent': round(float(row['Predicted_Rent']), 2),
                'residual': round(float(row['Residual']), 2),
                'residual_pct': round(float(row['Residual_Pct']), 2),
                'tag': row['Price_Tag'],
                'bhk': int(row['BHK']),
                'size': float(row['Size']),
                'locality': df.loc[row['id']].get('Area Locality', '') if row['id'] in df.index else ''
            })

        return jsonify({
            'summary': {
                'total': int(len(records)),
                'underpriced': int(counts.get('Underpriced', 0)),
                'fair': int(counts.get('Fair', 0)),
                'overpriced': int(counts.get('Overpriced', 0))
            },
            'records': records
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/locality_scorecard', methods=['GET'])
def get_locality_scorecard():
    """Rank localities with a composite score."""
    try:
        city = request.args.get('city')
        bhk = request.args.get('bhk', type=int)
        limit = request.args.get('limit', default=15, type=int)
        if limit is None:
            limit = 15
        limit = max(5, min(limit, 30))

        df = load_dataset()
        if city:
            df = df[df['City'].fillna('').str.lower() == city.lower()]
        if bhk:
            df = df[df['BHK'] == bhk]

        required = ['Rent', 'Area Locality', 'City']
        if any(c not in df.columns for c in required):
            return jsonify({'error': 'Required columns missing for scorecard.'}), 500

        work = df.copy()
        work['Rent'] = pd.to_numeric(work['Rent'], errors='coerce')
        work['Area Locality'] = work['Area Locality'].fillna('').replace('', 'Unknown locality')
        if 'Neighborhood_Livability_Score' not in work.columns:
            work['Neighborhood_Livability_Score'] = 5.0
        work['Neighborhood_Livability_Score'] = pd.to_numeric(
            work['Neighborhood_Livability_Score'],
            errors='coerce'
        ).fillna(5.0)

        work = work.dropna(subset=['Rent'])
        if work.empty:
            return jsonify({'scorecard': []})

        grouped = (
            work.groupby('Area Locality', as_index=False)
                .agg(
                    city=('City', lambda s: s.mode().iloc[0] if not s.mode().empty else (s.iloc[0] if len(s) else 'N/A')),
                    avg_rent=('Rent', 'mean'),
                    median_rent=('Rent', 'median'),
                    livability=('Neighborhood_Livability_Score', 'mean'),
                    listings=('Rent', 'count')
                )
        )

        if grouped.empty:
            return jsonify({'scorecard': []})

        # Normalized sub-scores
        rent_min = float(grouped['avg_rent'].min())
        rent_max = float(grouped['avg_rent'].max())
        rent_span = max(rent_max - rent_min, 1.0)
        grouped['affordability_score'] = 1 - ((grouped['avg_rent'] - rent_min) / rent_span)
        grouped['livability_score'] = (grouped['livability'] / 10.0).clip(0, 1)
        grouped['density_score'] = (grouped['listings'] / max(grouped['listings'].max(), 1)).clip(0, 1)

        # Trend proxy from avg-vs-median skew (simple robust signal when timeline is weak).
        grouped['trend_proxy_score'] = (
            1 - ((grouped['avg_rent'] - grouped['median_rent']).abs() / grouped['median_rent'].replace(0, np.nan))
        ).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0, 1)

        grouped['locality_score'] = (
            0.35 * grouped['affordability_score'] +
            0.30 * grouped['livability_score'] +
            0.20 * grouped['density_score'] +
            0.15 * grouped['trend_proxy_score']
        )

        top = grouped.sort_values(by='locality_score', ascending=False).head(limit)
        scorecard = []
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            scorecard.append({
                'rank': rank,
                'locality': row['Area Locality'],
                'city': row['city'],
                'locality_score': round(float(row['locality_score']) * 100, 1),
                'avg_rent': round(float(row['avg_rent']), 2),
                'median_rent': round(float(row['median_rent']), 2),
                'livability': round(float(row['livability']), 2),
                'listings': int(row['listings'])
            })

        return jsonify({'scorecard': scorecard, 'total_candidates': int(len(grouped))})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/commute_advisor', methods=['GET'])
def get_commute_advisor():
    """Recommend localities using budget + livability + commute distance."""
    try:
        city = request.args.get('city')
        bhk = request.args.get('bhk', type=int)
        budget = request.args.get('budget', type=float)
        limit = request.args.get('limit', default=8, type=int)
        work_city = request.args.get('work_city')
        work_lat = request.args.get('work_lat', type=float)
        work_lon = request.args.get('work_lon', type=float)

        if budget is None or budget <= 0:
            return jsonify({'error': "A positive 'budget' query parameter is required."}), 400
        limit = max(3, min(limit if limit is not None else 8, 20))

        if work_lat is None or work_lon is None:
            work_ref_city = work_city or city or 'Mumbai'
            work_lat, work_lon = _get_city_coords(work_ref_city)

        df = load_dataset()
        if city:
            df = df[df['City'].fillna('').str.lower() == city.lower()]
        if bhk:
            df = df[df['BHK'] == bhk]

        needed = ['Rent', 'Area Locality', 'City', 'Latitude', 'Longitude']
        if any(c not in df.columns for c in needed):
            return jsonify({'error': 'Required columns missing for commute advisor.'}), 500

        work_df = df.copy()
        work_df['Rent'] = pd.to_numeric(work_df['Rent'], errors='coerce')
        work_df['Latitude'] = pd.to_numeric(work_df['Latitude'], errors='coerce')
        work_df['Longitude'] = pd.to_numeric(work_df['Longitude'], errors='coerce')
        if 'Neighborhood_Livability_Score' not in work_df.columns:
            work_df['Neighborhood_Livability_Score'] = 5.0
        work_df['Neighborhood_Livability_Score'] = pd.to_numeric(
            work_df['Neighborhood_Livability_Score'], errors='coerce'
        ).fillna(5.0)
        work_df['Area Locality'] = work_df['Area Locality'].fillna('').replace('', 'Unknown locality')
        work_df = work_df.dropna(subset=['Rent', 'Latitude', 'Longitude'])

        if work_df.empty:
            return jsonify({'recommendations': []})

        grouped = (
            work_df.groupby('Area Locality', as_index=False)
                   .agg(
                       city=('City', lambda s: s.mode().iloc[0] if not s.mode().empty else (s.iloc[0] if len(s) else 'N/A')),
                       avg_rent=('Rent', 'mean'),
                       livability=('Neighborhood_Livability_Score', 'mean'),
                       listings=('Rent', 'count'),
                       lat=('Latitude', 'mean'),
                       lon=('Longitude', 'mean')
                   )
        )

        grouped['budget_fit'] = (1 - (grouped['avg_rent'] - budget).abs() / max(budget, 1)).clip(0, 1)
        grouped['livability_score'] = (grouped['livability'] / 10.0).clip(0, 1)
        grouped['density_score'] = (grouped['listings'] / max(grouped['listings'].max(), 1)).clip(0, 1)
        grouped['commute_km'] = grouped.apply(
            lambda r: _haversine_km(float(r['lat']), float(r['lon']), float(work_lat), float(work_lon)), axis=1
        )
        grouped['commute_score'] = np.exp(-grouped['commute_km'] / 20.0).clip(0, 1)

        grouped['final_score'] = (
            0.45 * grouped['budget_fit'] +
            0.30 * grouped['livability_score'] +
            0.20 * grouped['commute_score'] +
            0.05 * grouped['density_score']
        )

        top = grouped.sort_values(by='final_score', ascending=False).head(limit)
        recommendations = []
        for _, row in top.iterrows():
            recommendations.append({
                'locality': row['Area Locality'],
                'city': row['city'],
                'avg_rent': round(float(row['avg_rent']), 2),
                'livability': round(float(row['livability']), 2),
                'commute_km': round(float(row['commute_km']), 2),
                'score': round(float(row['final_score']) * 100, 1),
                'coordinates': {'lat': round(float(row['lat']), 5), 'lon': round(float(row['lon']), 5)}
            })

        return jsonify({
            'work_coordinates': {'lat': float(work_lat), 'lon': float(work_lon)},
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/explain_prediction', methods=['POST'])
def explain_prediction():
    """Return heuristic feature impacts for a prediction."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        payload = request.json or {}
        input_df, city, locality, lat, lon = _build_model_input(payload)
        prediction = float(model.predict(input_df)[0])

        df = load_dataset()
        city_df = df[df['City'].fillna('').str.lower() == city.lower()] if 'City' in df.columns else df
        if city_df.empty:
            city_df = df

        rent_series = pd.to_numeric(city_df.get('Rent'), errors='coerce').dropna()
        median_rent = float(rent_series.median()) if not rent_series.empty else prediction
        median_size = float(pd.to_numeric(city_df.get('Size'), errors='coerce').dropna().median()) if 'Size' in city_df.columns else float(input_df['Size'].iloc[0])
        median_bhk = float(pd.to_numeric(city_df.get('BHK'), errors='coerce').dropna().median()) if 'BHK' in city_df.columns else float(input_df['BHK'].iloc[0])
        median_bath = float(pd.to_numeric(city_df.get('Bathroom'), errors='coerce').dropna().median()) if 'Bathroom' in city_df.columns else float(input_df['Bathroom'].iloc[0])

        size = float(input_df['Size'].iloc[0])
        bhk = float(input_df['BHK'].iloc[0])
        bathroom = float(input_df['Bathroom'].iloc[0])
        furnishing = str(input_df['Furnishing Status'].iloc[0])
        area_type = str(input_df['Area Type'].iloc[0])

        rent_per_sqft = median_rent / max(median_size, 1)
        impacts = {
            'Size': (size - median_size) * rent_per_sqft * 0.65,
            'BHK': (bhk - median_bhk) * median_rent * 0.11,
            'Bathroom': (bathroom - median_bath) * median_rent * 0.05,
            'Furnishing Status': {'Furnished': 2500, 'Semi-Furnished': 1200, 'Unfurnished': -800}.get(furnishing, 0),
            'Area Type': {'Carpet Area': 1800, 'Built Area': 500, 'Super Area': 0}.get(area_type, 0),
            'Local Market Level': median_rent - prediction
        }

        sorted_impacts = sorted(impacts.items(), key=lambda x: abs(float(x[1])), reverse=True)
        top_impacts = [{
            'feature': k,
            'impact': round(float(v), 2),
            'direction': 'up' if float(v) >= 0 else 'down'
        } for k, v in sorted_impacts[:6]]

        return jsonify({
            'predicted_rent': round(prediction, 2),
            'city': city,
            'locality': locality,
            'summary': f"Prediction is benchmarked against {city} market medians.",
            'top_impacts': top_impacts,
            'location': {'lat': round(float(lat), 5), 'lon': round(float(lon), 5)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model_monitoring', methods=['GET'])
def model_monitoring():
    """Model monitoring and data quality snapshot."""
    try:
        df = load_dataset()
        rows, cols = df.shape
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        top_missing = [{
            'column': col,
            'missing_pct': round(float(pct), 2)
        } for col, pct in missing_pct.head(8).items()]

        duplicate_rows = int(df.duplicated().sum())
        eda_stats = _safe_json_read(os.path.join(ARTIFACTS_DIR, 'eda_stats.json'), {})

        current_avg = float(pd.to_numeric(df.get('Rent'), errors='coerce').dropna().mean()) if 'Rent' in df.columns else None
        baseline_avg = float(eda_stats.get('avg_rent')) if eda_stats.get('avg_rent') is not None else None
        rent_drift_pct = None
        if current_avg is not None and baseline_avg not in (None, 0):
            rent_drift_pct = round(((current_avg - baseline_avg) / baseline_avg) * 100.0, 2)

        model_file = MODEL_PATH
        model_mtime = datetime.utcfromtimestamp(os.path.getmtime(model_file)).isoformat() + 'Z' if os.path.exists(model_file) else None
        mae_sample = _evaluate_model_mae(model) if model else None

        history = _safe_json_read(MODEL_METRICS_HISTORY_PATH, [])
        return jsonify({
            'dataset': {
                'rows': int(rows),
                'columns': int(cols),
                'duplicate_rows': duplicate_rows,
                'top_missing': top_missing
            },
            'drift': {
                'baseline_avg_rent': baseline_avg,
                'current_avg_rent': round(current_avg, 2) if current_avg is not None else None,
                'rent_drift_pct': rent_drift_pct
            },
            'model': {
                'loaded': bool(model is not None),
                'model_path': model_file,
                'last_updated_utc': model_mtime,
                'sample_mae': round(mae_sample, 2) if mae_sample is not None else None,
                'version_history': history[-12:]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain_pipeline', methods=['POST'])
def retrain_pipeline():
    """Run retraining and promote only if quality is not degraded."""
    global model
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        old_model = model
        if old_model is None and os.path.exists(MODEL_PATH):
            try:
                old_model = joblib.load(MODEL_PATH)
            except Exception:
                old_model = None
        old_mae = _evaluate_model_mae(old_model) if old_model is not None else None

        backup_path = MODEL_PATH + '.backup'
        if os.path.exists(MODEL_PATH):
            shutil.copy2(MODEL_PATH, backup_path)

        cmd = [sys.executable, os.path.join('scripts', 'train_models.py')]
        proc = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, MODEL_PATH)
            return jsonify({
                'status': 'failed',
                'stdout_tail': proc.stdout[-1000:],
                'stderr_tail': proc.stderr[-1000:]
            }), 500

        try:
            new_model = joblib.load(MODEL_PATH)
        except Exception as load_err:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, MODEL_PATH)
            return jsonify({
                'status': 'failed',
                'error': f'New model artifact incompatible: {load_err}',
                'stdout_tail': proc.stdout[-1000:],
                'stderr_tail': proc.stderr[-1000:]
            }), 500
        new_mae = _evaluate_model_mae(new_model)

        promoted = True
        reason = 'new model promoted'
        if old_mae is not None and new_mae is not None and new_mae > old_mae * 1.05:
            promoted = False
            reason = 'new model reverted (MAE degraded >5%)'
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, MODEL_PATH)
            try:
                model = joblib.load(MODEL_PATH)
            except Exception:
                model = None
        else:
            model = new_model

        history = _safe_json_read(MODEL_METRICS_HISTORY_PATH, [])
        history.append({
            'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
            'old_mae': round(float(old_mae), 2) if old_mae is not None else None,
            'new_mae': round(float(new_mae), 2) if new_mae is not None else None,
            'promoted': promoted,
            'reason': reason
        })
        _safe_json_write(MODEL_METRICS_HISTORY_PATH, history[-200:])

        return jsonify({
            'status': 'ok',
            'promoted': promoted,
            'reason': reason,
            'old_mae': round(float(old_mae), 2) if old_mae is not None else None,
            'new_mae': round(float(new_mae), 2) if new_mae is not None else None,
            'stdout_tail': proc.stdout[-600:]
        })
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Retraining timed out'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET', 'POST'])
def alerts():
    if request.method == 'GET':
        return jsonify({'alerts': _safe_json_read(ALERTS_PATH, [])})

    try:
        payload = request.json or {}
        city = payload.get('city')
        budget = payload.get('budget')
        if not city or budget is None:
            return jsonify({'error': 'city and budget are required'}), 400

        alerts_list = _safe_json_read(ALERTS_PATH, [])
        next_id = max([int(a.get('id', 0)) for a in alerts_list] + [0]) + 1
        new_alert = {
            'id': next_id,
            'name': payload.get('name', f'Alert {next_id}'),
            'city': city,
            'bhk': payload.get('bhk'),
            'budget': float(budget),
            'active': bool(payload.get('active', True)),
            'created_at': datetime.utcnow().isoformat() + 'Z'
        }
        alerts_list.append(new_alert)
        _safe_json_write(ALERTS_PATH, alerts_list)
        return jsonify({'alert': new_alert, 'total': len(alerts_list)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/alerts/<int:alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    try:
        alerts_list = _safe_json_read(ALERTS_PATH, [])
        remaining = [a for a in alerts_list if int(a.get('id', -1)) != int(alert_id)]
        _safe_json_write(ALERTS_PATH, remaining)
        return jsonify({'deleted_id': alert_id, 'remaining': len(remaining)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    try:
        payload = request.json or {}
        alert_id = payload.get('alert_id')
        alerts_list = _safe_json_read(ALERTS_PATH, [])
        if alert_id is not None:
            alerts_list = [a for a in alerts_list if int(a.get('id', -1)) == int(alert_id)]

        df = load_dataset()
        df['Rent'] = pd.to_numeric(df.get('Rent'), errors='coerce')
        results = []
        for alert in alerts_list:
            if not alert.get('active', True):
                continue
            sub = df[df['City'].fillna('').str.lower() == str(alert.get('city', '')).lower()]
            if alert.get('bhk') is not None:
                sub = sub[sub['BHK'] == int(alert.get('bhk'))]
            sub = sub[sub['Rent'] <= float(alert.get('budget', 0))]
            samples = []
            if not sub.empty:
                for idx, row in sub.head(5).iterrows():
                    samples.append({
                        'id': int(idx),
                        'city': row.get('City', ''),
                        'locality': row.get('Area Locality', ''),
                        'rent': round(float(row.get('Rent', 0)), 2),
                        'bhk': int(row.get('BHK', 0)) if pd.notna(row.get('BHK', np.nan)) else None
                    })
            results.append({
                'alert_id': int(alert.get('id')),
                'name': alert.get('name', ''),
                'matches': int(len(sub)),
                'samples': samples
            })

        return jsonify({'results': results, 'checked_alerts': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_listings', methods=['GET'])
def compare_listings():
    """Compare selected listing ids side by side."""
    try:
        ids_raw = request.args.get('ids', '')
        ids = []
        for token in ids_raw.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                ids.append(int(token))
            except ValueError:
                continue
        ids = ids[:6]
        if not ids:
            return jsonify({'error': 'Provide ids query param, e.g., ids=1,2,3'}), 400

        df = load_dataset()
        rows = []
        for idx in ids:
            if 0 <= idx < len(df):
                row = df.iloc[idx].to_dict()
                row['id'] = idx
                rows.append(row)

        if not rows:
            return jsonify({'comparisons': [], 'summary': {}})

        rents = [float(r.get('Rent', 0)) for r in rows if str(r.get('Rent', '')).strip() != '']
        sizes = [float(r.get('Size', 0)) for r in rows if str(r.get('Size', '')).strip() != '']
        summary = {
            'count': len(rows),
            'min_rent': round(min(rents), 2) if rents else None,
            'max_rent': round(max(rents), 2) if rents else None,
            'avg_rent': round(sum(rents) / len(rents), 2) if rents else None,
            'min_size': round(min(sizes), 2) if sizes else None,
            'max_size': round(max(sizes), 2) if sizes else None
        }
        return jsonify({'comparisons': rows, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_rent():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json or {}

    try:
        input_data, city, locality, lat, lon = _build_model_input(data)
        prediction = float(model.predict(input_data)[0])

        # Confidence range using local market dispersion where possible.
        market_df = load_dataset()
        if 'City' in market_df.columns:
            market_df = market_df[market_df['City'].fillna('').str.lower() == city.lower()]
        if 'BHK' in market_df.columns:
            market_df = market_df[market_df['BHK'] == int(input_data['BHK'].iloc[0])]

        rents = pd.to_numeric(market_df.get('Rent'), errors='coerce').dropna()
        if len(rents) >= 20:
            p15 = float(np.percentile(rents, 15))
            p85 = float(np.percentile(rents, 85))
            low = 0.55 * prediction + 0.45 * p15
            high = 0.55 * prediction + 0.45 * p85
            method = 'city_bhk_percentile_blend'
        else:
            low = prediction * 0.85
            high = prediction * 1.15
            method = 'fallback_margin'

        low = round(float(max(0.0, min(low, prediction))), 2)
        high = round(float(max(prediction, high)), 2)

        return jsonify({
            'predicted_rent': round(prediction, 2),
            'currency': 'INR',
            'confidence_range': {
                'low': low,
                'expected': round(prediction, 2),
                'high': high,
                'method': method,
                'market_samples': int(len(rents))
            },
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
