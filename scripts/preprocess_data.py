import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def random_date(start_year=2025, end_year=2026):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randrange(delta.days)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')

def generate_coords(city):
    # Approximate centers for common Indian cities to generate clusters
    city_coords = {
        'Mumbai': (19.0760, 72.8777),
        'Bangalore': (12.9716, 77.5946),
        'Chennai': (13.0827, 80.2707),
        'Delhi': (28.7041, 77.1025),
        'Hyderabad': (17.3850, 78.4867),
        'Kolkata': (22.5726, 88.3639)
    }
    
    base_lat, base_lon = city_coords.get(city, (20.5937, 78.9629)) # Default to India center
    
    # Add random jitter (approx +/- 5-10km)
    lat_jitter = np.random.uniform(-0.1, 0.1)
    lon_jitter = np.random.uniform(-0.1, 0.1)
    
    return base_lat + lat_jitter, base_lon + lon_jitter

def process_data(input_file, output_file):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 1. Update 'Posted On' to 2025-2026
    print("Updating 'Posted On' dates...")
    df['Posted On'] = [random_date() for _ in range(len(df))]

    # 2. Add 'Bathroom_Type' (User Request)
    # Assuming 'Bathroom' column exists generally, but user asked to add 'Bathroom Type'
    # We will generate a categorical type
    print("Adding 'Bathroom_Type'...")
    bathroom_types = ['Attached', 'Common', 'En-suite', 'Standard']
    df['Bathroom_Type'] = np.random.choice(bathroom_types, size=len(df), p=[0.6, 0.2, 0.1, 0.1])
    
    # 3. Ensure 'Bathroom' count exists implies valid integer
    # If Bathroom is null or 0, fill based on BHK
    if 'Bathroom' not in df.columns:
        print(" Generating 'Bathroom' counts (missing in source)...")
        df['Bathroom'] = df['BHK'].apply(lambda x: random.randint(1, x + 1))
    else:
        print(" Validating 'Bathroom' counts...")
        df['Bathroom'] = df.apply(
            lambda row: random.randint(1, row['BHK'] + 1) if pd.isnull(row['Bathroom']) or row['Bathroom'] == 0 else row['Bathroom'], 
            axis=1
        )
        df['Bathroom'] = df['Bathroom'].astype(int)

    # 4. Synthetic Geospatial Data (for Abstract: 'Spatial clustering')
    print("Generating synthetic geospatial data...")
    # Check if City column exists
    if 'City' in df.columns:
        coords = df['City'].apply(generate_coords)
        df['Latitude'] = coords.apply(lambda x: x[0])
        df['Longitude'] = coords.apply(lambda x: x[1])
    else:
        # Fallback if City is missing
        print(" 'City' column not found, using generic coordinates.")
        df['Latitude'] = np.random.uniform(8.4, 37.6, len(df))
        df['Longitude'] = np.random.uniform(68.7, 97.2, len(df))

    # 5. Neighborhood Livability Score (for decision support)
    print("Generating Neighborhood Livability Scores...")
    df['Neighborhood_Livability_Score'] = np.round(np.random.uniform(1.0, 10.0, size=len(df)), 1)
    
    # Ensure Area Type column exists and handle missing values
    if 'Area Type' not in df.columns:
        print(" 'Area Type' column not found, generating default values...")
        df['Area Type'] = np.random.choice(['Super Area', 'Carpet Area', 'Built Area'], 
                                           size=len(df), p=[0.5, 0.3, 0.2])
    else:
        # Fill missing Area Type values
        df['Area Type'] = df['Area Type'].fillna('Super Area')
    
    # Ensure Area Locality exists
    if 'Area Locality' not in df.columns:
        print(" 'Area Locality' column not found, generating default values...")
        df['Area Locality'] = df['City'].apply(lambda x: f"Locality in {x}")
    
    # Clean up any empty columns if they exist (datasets sometimes have trailing commas)
    df = df.dropna(axis=1, how='all')
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    input_csv = os.path.join(PROJECT_ROOT, "House_Rent_Dataset.csv")
    output_csv = os.path.join(PROJECT_ROOT, "Cleaned_House_Rent_Dataset.csv")
    process_data(input_csv, output_csv)
