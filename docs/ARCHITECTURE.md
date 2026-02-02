# System Architecture: Urban Rental Decision Support System

## High-Level Overview
The system follows a **Service-Oriented Architecture (SOA)** tailored for Data Science applications. It decouples the data processing pipelines from the user-facing web application, ensuring that model training and inference can evolve independently of the UI.

```mermaid
graph TD
    User[Urban Migrant / Analyst] -->|Interact| Frontend[Frontend Layer (SPA)]
    Frontend -->|HTTP / JSON| API[Backend API Layer (Flask)]
    
    subgraph "Machine Learning Core"
        API -->|Inference| Model[ML Model Layer]
        Model -->|Load| Artifacts[(Model Artifacts)]
    end
    
    subgraph "Data Pipeline"
        RawData[(Raw CSV)] -->|ETL Script| Processor[Data Processing Layer]
        Processor -->|Clean & Augment| ProcessedData[(Structured Data Store)]
        Processor -->|Train| ModelTrainer[Model Trainer]
        ModelTrainer -->|Save| Artifacts
    end
    
    API -->|Read| ProcessedData
```

## Layer Definitions

### 1. Frontend Layer (Presentation)
*   **Tech Stack**: HTML5, CSS3 (Variables + Glassmorphism), Vanilla JavaScript ES6.
*   **Responsibilities**:
    *   **Dashboard**: Renders Chart.js visualizations for market trends.
    *   **Map Interface**: Leaflet.js interactive map for rendering geospatial clusters.
    *   **Input Form**: Captures user requirements (BHK, Location) for prediction.
*   **Communication**: Asynchronous `fetch` calls to the Backend API.

### 2. Backend API Layer (Application)
*   **Tech Stack**: Python (Flask).
*   **Responsibilities**:
    *   **Routing**: Serving static assets and API endpoints (`/api/predict`, `/api/stats`).
    *   **Orchestration**: Loading ML models into memory at startup.
    *   **Data Serving**: Reading JSON statistics and CSV data to send to the frontend.
    *   **Validation**: Ensuring input data types match model requirements.

### 3. Machine Learning Layer (Intelligence)
*   **Tech Stack**: Scikit-Learn (Random Forest, K-Means), Joblib.
*   **Components**:
    *   **Inference Engine**: Loaded via `joblib` in the Flask app to provide real-time estimates.
    *   **Training Pipeline**: Offline scripts (`train_models.py`) that generate `.pkl` files.
    *   **Clustering Engine**: Pre-calculates neighborhood zones (Affordability/Livability).

### 4. Data Processing Layer (ETL)
*   **Tech Stack**: Pandas, NumPy.
*   **Workflow**:
    1.  **Ingest**: Load raw `House_Rent_Dataset.csv`.
    2.  **Clean**: Fix missing values, normalize dates to 2025-2026.
    3.  **Augment**: Generate synthetic Geospatial data (Lat/Lon) and Livability Scores.
    4.  **Export**: Save to `Cleaned_House_Rent_Dataset.csv` and `eda_stats.json`.

### 5. Data Storage Layer (Persistence)
*   **Current Implementation**: File-based Storage.
    *   **Primary Data**: `Cleaned_House_Rent_Dataset.csv` (Acts as the Read-Only DB for the app).
    *   **Model Store**: `model_artifacts/` directory containing seralized objects.
*   **Future Scalability**: Can be migrated to **PostgreSQL** (with PostGIS for spatial queries) without changing the Frontend.

### 6. External APIs
*   **Mapping**: OpenStreetMap Tiles (via CartoDB Dark Matter) for the Leaflet map background.
*   **Fonts**: Google Fonts (Inter) for typography.

## Data Flow
1.  **Preprocessing**: Administrator runs `preprocess_data.py` -> Raw Data is transformed -> Cleaned CSV is saved.
2.  **Training**: Administrator runs `train_models.py` -> Reads Cleaned CSV -> Trains Forest/KMeans -> Saves `.pkl` models and `.json` stats.
3.  **Startup**: Flask App starts -> Loads `.pkl` models into RAM.
4.  **User Action (View Map)**: Frontend requests `/api/map_data` -> Backend reads CSV subset -> Returns JSON -> Map renders dots.
5.  **User Action (Predict)**: Frontend sends Form Data -> Backend creates DataFrame -> Passes to Random Forest -> Returns Rent Estimate.
