# UrbanMove: Rental Analytics and Decision Support System - PPT Outline

Copy each slide title and bullet list into PowerPoint or Google Slides.

## Slide 1: Title

- UrbanMove: Rental Analytics and Decision Support System
- Final Year Minor Project
- Team members
- Guide name
- Department / Semester / Date

## Slide 2: Problem Statement

- Rental decisions in metro cities are affected by price variation, locality quality, and commute constraints.
- Most users do not have one system that combines fair rent estimation with locality comparison.
- This creates information asymmetry and poor decision making.

## Slide 3: Project Objectives

- Build a cleaned rental data pipeline.
- Predict fair rent from property features.
- Rank localities using affordability and livability factors.
- Provide a simple web interface for non-technical users.

## Slide 4: Dataset and Preprocessing

- Source rental dataset cleaned and normalized.
- Missing values handled and useful derived columns created.
- Listing dates refreshed for freshness analysis.
- Locality profile data added for richer recommendation logic.

## Slide 5: Machine Learning Approach

- Random Forest Regressor for rent prediction.
- K-Means clustering for area grouping and map visualization.
- Supporting analytics for market range, trend context, and locality scoring.

## Slide 6: System Architecture

- Frontend: HTML, CSS, JavaScript, Chart.js, Leaflet.
- Backend: Flask REST API.
- ML and analytics layer: Pandas, scikit-learn, Joblib.
- Persistence: CSV data, model artifacts, SQLite for saved searches and shortlist.

## Slide 7: Main User Flow

- User enters rental requirements in Rent Analytics.
- System predicts rent and confidence range.
- Budget and market context are shown.
- Recommended localities and next actions are generated.

## Slide 8: Key Features Implemented

- Dashboard and map exploration.
- Rent Analytics with prediction and recommendations.
- Trends and forecast.
- Compare Listings.
- Locality Scorecard.
- Saved searches, shortlist, and prediction feedback.

## Slide 9: UI and UX Improvements

- Simplified analytics form for non-technical users.
- Theme toggle for light and dark mode.
- Compact card-based layouts for lower sections.
- Workplace map URL support instead of manual latitude and longitude entry.

## Slide 10: Testing and Validation

- Python syntax validation for backend.
- JavaScript syntax validation for frontend logic.
- Automated unit and smoke tests for main APIs.
- Local retraining completed to align model artifacts with the current environment.

## Slide 11: Results and Practical Value

- Users can estimate fair rent quickly.
- Users can compare multiple listings side by side.
- Users can evaluate localities using affordability, livability, and commute context.
- Users can save searches and shortlist options for later review.

## Slide 12: Future Scope

- Live listing refresh from external sources.
- Real travel time instead of approximate distance only.
- Visit scheduling and negotiation workflow.
- User login and cloud sync.

## Slide 13: Demo Screens

- Dashboard
- Rent Analytics prediction screen
- Compare Listings
- Locality Scorecard

## Slide 14: Conclusion

- UrbanMove combines data science and web development to support real rental decisions.
- The project delivers prediction, comparison, recommendation, and persistence in one workflow.
- The platform can be extended into a production-grade rental intelligence system.

## Slide 15: Questions

- Thank you
- Questions and feedback
