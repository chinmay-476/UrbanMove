# UrbanMove: Rental Analytics and Decision Support System - Speaker Notes

Use these as short presentation notes. Keep each slide explanation to 30-60 seconds unless the panel asks for detail.

## Slide 1: Title

- Introduce the project name, team, guide, and the fact that this is a rental decision support web application.

## Slide 2: Problem Statement

- Explain that renters usually compare budget, locality, and property quality manually.
- Point out that there is no single simple tool for fair rent estimation plus decision support.

## Slide 3: Project Objectives

- Mention the four goals: clean data, predict rent, analyze localities, and build a usable interface.

## Slide 4: Dataset and Preprocessing

- Explain that the raw rental data was cleaned, normalized, and enriched.
- Mention refreshed listing dates and locality profile support.

## Slide 5: Machine Learning Approach

- State that Random Forest is used for rent prediction.
- State that K-Means is used for locality clustering and map grouping.
- Mention that the system also produces supporting analytics beyond the prediction itself.

## Slide 6: System Architecture

- Describe the flow as frontend to Flask API to ML/data layer.
- Mention SQLite for saved searches, shortlist items, and feedback.

## Slide 7: Main User Flow

- Walk through the analytics journey: user enters details, gets rent prediction, sees budget gap, and receives recommendations.

## Slide 8: Key Features Implemented

- Highlight Dashboard, Map, Rent Analytics, Trends, Compare Listings, Scorecard, and local persistence features.

## Slide 9: UI and UX Improvements

- Explain that the interface was simplified for non-technical users.
- Mention card layouts, theme toggle, and map URL input as practical usability improvements.

## Slide 10: Testing and Validation

- Say that backend syntax, frontend syntax, and automated API tests were run.
- Mention that model artifacts were retrained in the current environment for clean verification.

## Slide 11: Results and Practical Value

- Emphasize that the app supports actual rental decision making, not only model prediction.
- Mention compare, shortlist, scorecard, and saved searches as practical modules.

## Slide 12: Future Scope

- Keep this realistic: live listing freshness, travel-time integration, visit workflow, and cloud accounts.

## Slide 13: Demo Screens

- Introduce each screen quickly and connect it to a real user action.

## Slide 14: Conclusion

- Summarize that the project combines data engineering, machine learning, and a usable web interface.

## Slide 15: Questions

- Invite questions on dataset, ML approach, API design, or frontend workflow.
