# User Types and Use Cases

## 1. Urban Migrants
**Persona**: Individuals or families moving to a new metropolitan city looking for affordable and suitable housing.
**Use Cases**:
1.  **Rent Estimation**: Enter property details (BHK, Size, Locality) to get a fair rent estimate before negotiating with landlords.
2.  **Neighborhood Discovery**: View the "Affordability Zones" map to identify which areas of a city fit their budget (e.g., Red vs. Green zones).
3.  **Livability Assessment**: Check the *Neighborhood Livability Score* to ensure a potential area has good amenities and infrastructure.
4.  **Budget Planning**: View average rental trends in a target city to plan finances for the move (e.g., "Is 20k enough for a 2BHK in Mumbai?").
5.  **Requirement Filtering**: Check if a specific neighborhood typically supports their lifestyle (e.g., "Bachelors" vs. "Family" preferred areas).
6.  **Comparative Analysis**: Compare rent prices between two similar sized apartments in different clusters/zones.

## 2. Real Estate Analysts
**Persona**: Data analysts or researchers studying housing market trends, rental yields, and urban development.
**Use Cases**:
1.  **Market Trend Analysis**: visualizing the distribution of rental prices across different cities and property types (1BHK vs 3BHK).
2.  **Cluster Identification**: Analyze the output of the K-Means clustering to identify emerging "hotspots" or undervalued neighborhoods.
3.  **Model Performance Review**: Evaluate the accuracy of the Price Prediction Model (MAE, R2) to assess market volatility.
4.  **Feature Impact Study**: Determine how factors like "Furnishing Status" or "Bathroom Type" quantitatively impact rental value.
5.  **Geospatial Distribution**: Study the density of listings in specific latitude/longitude buckets to understand urban sprawl.
6.  **Data Export**: Access the cleaned and augmented datasets (`Cleaned_House_Rent_Dataset.csv`) for external deep-dive reporting.

## 3. System Administrators
**Persona**: technical staff responsible for maintaining the application, data pipelines, and infrastructure.
**Use Cases**:
1.  **Pipeline Triggering**: Run the `preprocess_data.py` script to accept and clean new raw datasets.
2.  **Model Retraining**: Execute `train_models.py` to update the Random Forest and K-Means models with fresh data.
3.  **End-to-End Verification**: Verify that the Flask API endpoints (`/api/predict`, `/api/stats`) are returning 200 OK statuses.
4.  **Configuration Management**: Adjust parameters for synthetic data generation (e.g., changing the date range in `preprocess_data.py`).
5.  **Artifact Management**: Ensure `model_artifacts/` directory contains valid `.pkl` and `.json` files for the web app to consume.
6.  **Error Log Monitoring**: Debug issues if the frontend fails to load the map or prediction components.
