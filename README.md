# UrbanMove: Rental Analytics and Decision Support System

UrbanMove is a Flask-based web application that helps renters make practical housing decisions using market data, machine learning, and a simple decision-support interface.

## Problem Statement

Rental decisions in metropolitan cities are difficult because users must balance budget, property features, commute, and neighborhood quality without a single reliable reference point. This project addresses that gap by combining rent prediction, locality analysis, trend context, and listing comparison in one workflow.

## Project Objectives

1. Clean and enrich rental housing data for analysis and prediction.
2. Predict fair rent values from property and location attributes.
3. Recommend localities using affordability, livability, and commute context.
4. Provide a web interface that remains usable for non-technical users.

## Current Feature Set

- Dashboard with city-level market statistics and charts.
- Spatial Analysis map with livability heat zones, rent-band marker colors, popup listing details, and a category legend below the map.
- Rent Analytics flow with Classic and Glass Cards modes, resizable input/result panes, prediction, confidence range, market position, budget guidance, and locality recommendations.
- Trends and forecast view for short-term rent direction.
- Compare Listings for side-by-side property review.
- Locality Scorecard for ranked locality summaries.
- Saved searches, shortlist storage, and prediction feedback using SQLite.
- Listing freshness and contact-type indicators.
- Light and dark theme toggle.
- Workplace map URL input that extracts coordinates instead of asking users for raw latitude and longitude.
- Recommendation cards can jump directly to the exact in-app heat-map location before opening Google Maps.

## Architecture Snapshot

- Frontend: HTML, CSS, Vanilla JavaScript, Chart.js, Leaflet.
- Backend: Python, Flask, Flask-CORS.
- ML layer: scikit-learn Random Forest for rent prediction and K-Means for clustering.
- Data layer: CSV-based rental dataset, model artifacts in `model_artifacts/`, and SQLite for local user actions.

## Latest UX Updates

- Added a dual-mode Rent Analytics workspace: `Classic` for a normal split layout and `Glass Cards` for large floating input cards.
- Added a draggable divider between analytics inputs and prediction results so users can resize both sides.
- Simplified recommendation-to-map flow so location actions focus the internal heat map first.
- Reworked the map into clear livability categories with visible color meaning for both zones and rent markers.

## Repository Structure

```text
.
|-- app.py
|-- Cleaned_House_Rent_Dataset.csv
|-- data/
|-- docs/
|-- instance/
|-- model_artifacts/
|-- scripts/
|-- static/
|-- tests/
`-- requirements.txt
```

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Regenerate model artifacts if needed:

   ```bash
   python scripts/preprocess_data.py
   python scripts/train_models.py
   ```

4. Start the app:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000`.

## Verification

Run these checks before submission or demo:

```bash
python -m py_compile app.py
node --check static/script.js
python -m unittest discover -s tests -p "test_*.py"
```

## Presentation and Submission Docs

- PPT outline: `docs/PPT_OUTLINE.md`
- Speaker notes: `docs/SPEAKER_NOTES.md`
- Architecture summary: `docs/ARCHITECTURE.md`
- Final submission and demo checklist: `docs/FINAL_SUBMISSION_CHECKLIST.md`

## Notes

- Model artifacts have been retrained in the current environment to avoid scikit-learn version mismatch warnings during normal verification.
- The app is intentionally focused on a guided rental decision workflow rather than account management or live web scraping.
