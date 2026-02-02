# Urban Rental Decision Support System — Speaker Notes

Short notes for what to say on each slide. Adjust timing to your audience (e.g., 1–2 minutes per slide).

---

## Slide 1: Title

- Introduce the project name and subtitle.
- Optionally: "This is my 6th semester project on an AI-powered system to help people make better rental decisions in cities."

---

## Slide 2: Problem Statement

- "In big cities, finding the right rental is hard: prices change, areas differ in livability, and people don't have one place to compare and get fair rent estimates."
- "We're solving information asymmetry—tenants often don't know if a rent is fair or which areas fit their budget."

---

## Slide 3: Project Objectives

- "We have four main goals: first, integrate and clean rental data; second, build models to predict fair rent; third, analyze neighborhoods with clustering; fourth, give users a web dashboard to explore and get estimates."

---

## Slide 4: High-Level Architecture

- "The system uses a service-oriented design: the user interacts with a frontend SPA, which talks to a Flask API. The API uses ML models and reads from cleaned data. Separately, an ETL pipeline and training scripts produce that data and the model files."

---

## Slide 5: Tech Stack

- "Frontend is HTML, CSS with a glassmorphism style, and vanilla JavaScript with Chart.js and Leaflet. Backend is Flask. ML is Scikit-Learn—Random Forest and K-Means—with Joblib for saving models. Data is file-based for now: CSV, JSON, and PKL."

---

## Slide 6: Data Pipeline (ETL)

- "We start from raw house rent CSV. We clean missing values and dates, then add synthetic latitude/longitude and livability scores. The output is a cleaned CSV and an EDA stats JSON. All of this is done by preprocess_data.py."

---

## Slide 7: Machine Learning Module

- "We use a Random Forest regressor for rent prediction from BHK, size, city, area type, and similar features. K-Means clusters neighborhoods into affordability and livability zones. EDA stats are precomputed and stored in JSON for the dashboard."

---

## Slide 8: Backend API (Flask)

- "The Flask app serves the single-page app and exposes REST endpoints: stats for charts, properties with filters, map_data for the map, and predict for the rent estimator. Models are loaded at startup."

---

## Slide 9: Frontend — Dashboard

- "The dashboard has a dark, premium look with glass-style cards. It shows key stats—average rent, total listings, cities, median rent—and a Chart.js chart for rent distribution by city."

---

## Slide 10: Frontend — Map & Predictor

- "The map uses Leaflet with clusters and a heatmap; clicking a marker can open the location in Google Maps. The predictor is a form where you enter BHK, size, city, area type, etc., and get a rent estimate in real time from the API."

---

## Slide 11: User Personas & Use Cases

- "We target three groups: urban migrants who want estimates and affordability zones; analysts who want market trends and clustering; and admins who run the data and training pipelines and check that APIs work."

---

## Slide 12: In-Scope vs Out-of-Scope

- "In scope we have the full pipeline, both ML models, the API, and the UI with map and predictor. Out of scope for this project are live scraping, user accounts, payments, native mobile apps, and NLP on descriptions."

---

## Slide 13: Future Scalability

- "For scaling, we could move to PostgreSQL with PostGIS, add the full schema from our design doc—users, saved searches, neighborhoods, properties, listings, and prediction logs—and add model versioning."

---

## Slide 14: Demo / Screenshots

- "Here are screenshots of the dashboard, the map with clusters, and the rent predictor. [If doing live demo: I can show a quick demo of the app.]"

---

## Slide 15: Conclusion & Q&A

- "To wrap up: we built a data-driven decision support system—ETL, Random Forest and K-Means, Flask API, and an interactive dashboard and map—to help users with rental decisions. Thank you; I'm happy to take questions."
