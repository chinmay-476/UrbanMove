# Final Submission Checklist

Use this checklist before final report submission, review, or demo.

## 1. Report Formatting

- Title matches the project exactly: `UrbanMove: Rental Analytics and Decision Support System`
- Abstract and problem statement explain the renter decision problem clearly.
- Objectives are listed in measurable form.
- Architecture diagram is added and readable.
- Dataset description includes the cleaned rental dataset and locality profile enrichment.
- Model section covers Random Forest prediction and K-Means clustering.
- UI section includes screenshots for Dashboard, Map, Rent Analytics, Compare Listings, and Locality Scorecard.
- Testing section includes code validation, API smoke testing, and user-flow verification.
- Future scope is realistic and separate from implemented features.
- Team member names, roll numbers, guide name, and submission date are consistent everywhere.

## 2. Technical Verification

Run these commands and keep the results ready:

```bash
python -m py_compile app.py
node --check static/script.js
python -m unittest discover -s tests -p "test_*.py"
```

## 3. Demo Readiness

- Start the app locally before the review begins.
- Keep one browser tab open on the Dashboard and one on Rent Analytics.
- Make sure the dataset and model artifacts are present.
- Confirm that saved searches and shortlist actions work in the local SQLite database.
- Keep at least 3 screenshots ready in case live internet is unstable for map tiles.

## 4. Suggested 5-Minute Live Demo Flow

1. Open Dashboard and explain city-wide market summary.
2. Open Map Explore and show clustered listings.
3. Open Rent Analytics and enter a realistic example.
4. Click `Predict Rent` and explain confidence range, budget gap, and locality matches.
5. Save the search, open Compare Listings, and compare top matches.
6. Open Locality Scorecard and explain how rankings support final decision making.
7. Show theme toggle and shortlist or feedback storage as finishing touches.

## 5. Sample Demo Input

Use this if you need a stable example:

- City: `Mumbai`
- Locality: `Borivali East`
- BHK: `2`
- Size: `750`
- Bathroom: `2`
- Furnishing: `Semi-Furnished`
- Tenant: `Bachelors/Family`
- Area Type: `Carpet Area`
- Budget Target: `35000`
- Workplace Map URL: `https://maps.google.com/?q=19.0760,72.8777`

## 6. Backup Talking Points

- The system helps non-technical users make rental decisions without reading raw data tables.
- Prediction is not the only output; the app also gives locality ranking, comparison, cost planning, and market context.
- Saved searches, shortlist, and feedback make the app more practical than a one-time rent predictor.
- The project combines data science and web application development in one end-to-end workflow.
