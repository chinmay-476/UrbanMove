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
