# Urban Rental Decision Support System — PPT Outline

Copy each slide block into PowerPoint or Google Slides. Use slide titles as heading and bullets as content.

---

## Slide 1: Title

**Urban Rental Decision Support System**

- AI-Powered Rent Estimation & Neighborhood Analysis
- [Your name]
- [Course / 6th Semester]
- [Date]

---

## Slide 2: Problem Statement

**Problem Statement**

- Rental housing decisions in metropolitan cities are increasingly complex (rapid urban migration, fluctuating prices, varying livability).
- Tenants lack a unified, data-driven platform to assess affordability, compare neighborhoods, and predict fair rent.
- This leads to information asymmetry and suboptimal housing choices.

---

## Slide 3: Project Objectives

**Project Objectives (4 Pillars)**

1. **Data Integration** — Clean, normalize, and enrich rental data with geospatial and amenity attributes.
2. **Predictive Modeling** — Build ML models to estimate fair rent from property features and location.
3. **Neighborhood Analysis** — Identify "Affordability Zones" and evaluate livability using spatial clustering.
4. **Decision Support Interface** — Provide an interactive web dashboard for trends, map exploration, and rent estimates.

---

## Slide 4: High-Level Architecture

**High-Level Architecture**

- **Caption:** Service-Oriented Architecture: Frontend (SPA) ↔ Flask API ↔ ML models & file-based data.
- **Visual:** Insert the architecture diagram (see DIAGRAM_EXPORT.md for Mermaid code and export instructions).
- User → Frontend → API → ML Layer + Data Pipeline.

---

## Slide 5: Tech Stack

**Tech Stack**

- **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JS, Chart.js, Leaflet.js
- **Backend:** Python, Flask, Flask-CORS
- **ML:** Scikit-Learn (Random Forest, K-Means), Joblib, Pandas, NumPy
- **Data:** File-based (CSV, JSON, PKL); OpenStreetMap tiles for map

---

## Slide 6: Data Pipeline (ETL)

**Data Pipeline (ETL)**

- **Flow:** Raw CSV → Clean (missing values, date normalization) → Augment (synthetic Lat/Lon, Livability_Score) → Export Cleaned CSV + eda_stats.json
- **Script:** preprocess_data.py
- Ingest → Clean → Augment → Export

---

## Slide 7: Machine Learning Module

**Machine Learning Module**

- **Price Prediction:** Random Forest Regressor (BHK, Size, Area Type, City, etc.) → rent estimate.
- **Clustering:** K-Means on neighborhoods → Affordability / Livability zones.
- **EDA:** Automated stats (e.g., rent distribution by city) in eda_stats.json.
- **Artifacts:** price_prediction_model.pkl, kmeans_model.pkl, clustering_scaler.pkl, eda_stats.json, clustered_data.csv.

---

## Slide 8: Backend API (Flask)

**Backend API (Flask)**

- GET / — Serve SPA
- GET /api/stats — EDA statistics (for charts)
- GET /api/properties — Filtered property list (city, rent, BHK)
- GET /api/map_data — Clustered points for Leaflet map
- POST /api/predict — Rent prediction from form inputs

---

## Slide 9: Frontend — Dashboard

**Frontend: Dashboard**

- Premium/Dark theme with glassmorphism cards
- Chart.js: Rent Distribution by City
- Stat cards: Avg Rent, Total Listings, Cities, Median Rent
- AI-driven insights for urban rentals

---

## Slide 10: Frontend — Map & Predictor

**Frontend: Map & Predictor**

- **Map:** Leaflet.js with cluster/heatmap; markers open in Google Maps.
- **Rent Predictor:** Form (BHK, Size, City, Area Type, etc.) → real-time estimate via /api/predict.

---

## Slide 11: User Personas & Use Cases

**User Personas & Use Cases**

- **Urban Migrants** — Rent estimate, affordability zones, livability assessment, budget planning.
- **Real Estate Analysts** — Market trends, clustering analysis, model performance, feature impact.
- **System Admins** — Run preprocess_data.py, train_models.py, verify API endpoints, manage artifacts.

---

## Slide 12: In-Scope vs Out-of-Scope

**In-Scope vs Out-of-Scope**

- **In-Scope:** Data pipeline, Random Forest + K-Means, Flask API, responsive UI, map, prediction form.
- **Out-of-Scope:** Live scraping, user auth/profiles, payments/landlord contact, native mobile app, NLP on descriptions.

---

## Slide 13: Future Scalability

**Future Scalability**

- Migrate to PostgreSQL + PostGIS for spatial queries
- Implement full schema: users, saved_searches, neighborhoods, properties, rental_listings, ml_predictions
- Model versioning and prediction logging

---

## Slide 14: Demo / Screenshots

**Demo / Screenshots**

- Screenshot 1: Dashboard (City Overview, charts, stat cards)
- Screenshot 2: Map Explore (clustering, heatmap)
- Screenshot 3: Rent Predictor (form + result)

---

## Slide 15: Conclusion & Q&A

**Conclusion & Q&A**

- Data-driven DSS for rental decisions: ETL → ML (RF + K-Means) → Flask API → interactive dashboard and map.
- Thank you — Questions?
