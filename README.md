# Urban Rental Decision Support System

## Problem Statement
Decisions related to rental housing in metropolitan cities have become increasingly complex due to rapid urban migration, fluctuating rental prices, and varying neighborhood livability. Potential tenants lack a unified, data-driven platform to assess affordability, compare neighborhood amenities, and predict fair rental prices, leading to information asymmetry and suboptimal housing choices.

## Project Objectives
1.  **Data Integration**: Aggregating and processing rental housing data to clean, normalize, and enrich it with geospatial and amenity attributes.
2.  **Predictive Modeling**: Developing machine learning models to estimate fair rental prices based on property features and location.
3.  **Neighborhood Analysis**: identifying "Affordability Zones" and evaluating neighborhood livability using spatial clustering techniques.
4.  **Decision Support Interface**: Providing an interactive web-based dashboard for users to visualize market trends, explore location data, and receive personalized rent estimates.

## High-Level Scope

### In-Scope Features
*   **Data Processing Pipeline**:
    *   Cleaning and preprocessing of raw CSV dataset (handling missing values, date normalization).
    *   Feature engineering (e.g., `Bathroom_Type`, generic `Livability_Score`).
    *   Synthetic data generation for missing geospatial coordinates (Lat/Lon).
*   **Machine Learning Module**:
    *   **Price Prediction**: Random Forest Regressor to predict rent based on BHK, Size, Area Type, City, etc.
    *   **Clustering**: K-Means clustering to categorize neighborhoods into affordability/livability zones.
    *   **EDA**: Automated generation of key statistical insights (e.g., Rent distribution by City).
*   **Web Application**:
    *   **Backend**: Flask-based REST API to serve model predictions and analytics data.
    *   **Frontend**: Responsive HTML/CSS/JS interface with a "Premium/Dark Mode" design.
    *   **Interactive Map**: Leaflet.js map visualizing property clusters and zones.
    *   **Dashboard**: Chart.js visualizations for market insights.
    *   **Prediction Tool**: User-friendly form for real-time rent estimation.

### Out-of-Scope Features
*   Real-time data scraping from live real estate websites.
*   User authentication, user profiles, or save functionality.
*   Direct integration with payment gateways or landlord contact systems.
*   Mobile application (iOS/Android native).
*   Advanced NLP analysis of listing descriptions.

## Repository Structure
```text
.
|-- app.py
|-- requirements.txt
|-- Cleaned_House_Rent_Dataset.csv
|-- model_artifacts/
|-- static/
|-- scripts/
|   |-- preprocess_data.py
|   |-- train_models.py
|   `-- version_debug.py
`-- docs/
    |-- ARCHITECTURE.md
    |-- DIAGRAM_EXPORT.md
    |-- PPT_OUTLINE.md
    |-- SCHEMA.md
    |-- SPEAKER_NOTES.md
    `-- USER_STORIES.md
```

## Run Locally
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Rebuild processed data and models:
   ```bash
   python scripts/preprocess_data.py
   python scripts/train_models.py
   ```
4. Start the app:
   ```bash
   python app.py
   ```

## How To Use (Page by Page)
1. `📊 Dashboard`
   - Opens by default and shows market KPIs.
   - Hover bars in the city chart to see city-wise average rent in the stat card.

2. `🗺️ Map Explore`
   - Explore listing clusters and heat layer.
   - Click anywhere on the map or a marker to open that location in Google Maps.

3. `💡 Rent Analytics`
   - Fill `City`, `BHK`, `Size`, `Bathrooms`, furnishing details, then click `Predict Rent`.
   - After prediction, the right panel auto-loads:
   - confidence range (low/expected/high),
   - market position,
   - trend forecast,
   - suggested locations,
   - budget advisor.
   - On desktop, form stays fixed and result cards are independently scrollable.

4. `📈 Trends & Forecast`
   - Enter city/locality filters and forecast horizon (`3M/6M/12M`).
   - Click `Load Trend` to view historical median trend and projected curve.

5. `📋 Compare Listings`
   - Enter comma-separated listing IDs (example: `1,5,12`).
   - Click `Compare` for side-by-side summary.

6. `🏙️ Locality Scorecard`
   - Set city and BHK.
   - Click `Load Scorecard` to rank localities by affordability, livability, trend proxy, and demand.

7. `🔔 Alerts`
   - Create alerts using city/BHK/budget.
   - Click `Save Alert`, then `Check Alerts` to find matching listings.

8. `🏠 Similar Homes`
   - Use manual inputs and click `Find Similar`, or click `Use Last Prediction`.
   - Shows top similar listings with match score.

9. `🧠 Price Intelligence`
   - Choose city, BHK, and sample limit.
   - Click `Analyze Prices` to label records as `Underpriced`, `Fair`, or `Overpriced`.

10. `🚇 Commute Planner`
    - Set city, budget, BHK, and work city.
    - Click `Find Commute-Friendly` to get locality recommendations with commute distance.

11. `🔬 Model Lab`
    - `Run What-If`: scenario simulation from latest prediction.
    - `Explain Last Prediction`: feature-impact style explanation.
    - `Load Monitoring`: data quality, drift snapshot, model metrics.
    - `Run Retrain Pipeline`: retrain and safe-promote/revert based on MAE.

## Functionality Smoke Test
Run this to verify all APIs after changes:
```bash
python -m py_compile app.py
node --check static/script.js
```

For endpoint checks, use Flask `test_client` to call:
`/api/stats`, `/api/properties`, `/api/map_data`, `/api/properties/<id>`, `/api/cities`,
`/api/budget_advisor`, `/api/market_insights`, `/api/rent_trends`, `/api/predict`,
`/api/what_if`, `/api/similar_listings`, `/api/price_intelligence`,
`/api/locality_scorecard`, `/api/commute_advisor`, `/api/explain_prediction`,
`/api/model_monitoring`, `/api/retrain_pipeline`, `/api/alerts`,
`/api/alerts/check`, `/api/compare_listings`.
