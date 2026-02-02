# MySQL Database Schema: Urban Rental Decision Support System

## Overview
This schema is designed to support the scalable requirements of the Urban Rental DSS, moving beyond the static CSV file used in the MVP. It supports users, dynamic property listings, historical price tracking, and ML prediction logging.

## Entity-Relationship Diagram (ERD) Concept
*   **Users** have 0..N **Saved Searches** and **ML Predictions**.
*   **Neighborhoods** contain 0..N **Properties** and are linked to **Amenities**.
*   **Properties** have 0..N **Rental Listings** (allowing price history).

## Table Definitions

### 1. Users & Personalization
**Table**: `users`
*   Stores user credentials and profile info.
*   **Columns**:
    *   `user_id` (INT, PK, Auto Increment)
    *   `email` (VARCHAR(255), Unique, Not Null)
    *   `password_hash` (VARCHAR(255), Not Null)
    *   `full_name` (VARCHAR(100))
    *   `role` (ENUM('user', 'analyst', 'admin'), Default 'user')
    *   `created_at` (DATETIME, Default NOW())

**Table**: `saved_searches`
*   Stores queries users want to revisit (e.g., "2BHK in Mumbai under 20k").
*   **Columns**:
    *   `search_id` (INT, PK, Auto Increment)
    *   `user_id` (INT, FK -> users.user_id)
    *   `search_name` (VARCHAR(100))
    *   `criteria_json` (JSON) - *Stores filters like {city: "Mumbai", bhk: 2}*
    *   `created_at` (DATETIME)

### 2. Location & Neighborhoods
**Table**: `neighborhoods`
*   Master list of localities with calculated livability metrics.
*   **Columns**:
    *   `neighborhood_id` (INT, PK, Auto Increment)
    *   `city` (VARCHAR(50), Index)
    *   `locality_name` (VARCHAR(100))
    *   `latitude` (DECIMAL(10, 8))
    *   `longitude` (DECIMAL(10, 8))
    *   `livability_score` (DECIMAL(3, 1)) - *Pre-calculated metric*
    *   `affordability_cluster_id` (INT) - *From K-Means model*

**Table**: `amenities`
*   Points of Interest (POIs) like schools, hospitals.
*   **Columns**:
    *   `amenity_id` (INT, PK, Auto Increment)
    *   `name` (VARCHAR(100))
    *   `type` (ENUM('School', 'Hospital', 'Mall', 'Park', 'Transport'))
    *   `latitude` (DECIMAL(10, 8))
    *   `longitude` (DECIMAL(10, 8))

**Table**: `neighborhood_amenity_map` (Junction)
*   Links amenities to neighborhoods if they are within a certain radius.
*   **Columns**:
    *   `neighborhood_id` (INT, FK)
    *   `amenity_id` (INT, FK)
    *   `distance_km` (DECIMAL(5, 2))
    *   **Primary Key**: (`neighborhood_id`, `amenity_id`)

### 3. Properties & Listings
**Table**: `properties`
*   Physical housing units.
*   **Columns**:
    *   `property_id` (INT, PK, Auto Increment)
    *   `neighborhood_id` (INT, FK -> neighborhoods.neighborhood_id)
    *   `bhk` (INT, Not Null)
    *   `size_sqft` (INT, Not Null)
    *   `floor_level` (VARCHAR(50)) - *e.g., "Ground out of 2"*
    *   `furnishing_status` (ENUM('Furnished', 'Semi-Furnished', 'Unfurnished'))
    *   `bathroom_count` (INT)
    *   `bathroom_type` (ENUM('Attached', 'Common', 'En-suite', 'Standard'))
    *   `tenant_preference` (ENUM('Bachelors', 'Family', 'Bachelors/Family'))

**Table**: `rental_listings`
*   Temporal listing data (Price History).
*   **Columns**:
    *   `listing_id` (INT, PK, Auto Increment)
    *   `property_id` (INT, FK -> properties.property_id)
    *   `posted_on` (DATE)
    *   `rent_amount` (DECIMAL(12, 2))
    *   `is_active` (BOOLEAN, Default True)
    *   `point_of_contact` (VARCHAR(100))

### 4. Machine Learning Logs
**Table**: `ml_predictions`
*   Logs user predictions for model monitoring and analytics.
*   **Columns**:
    *   `prediction_id` (INT, PK, Auto Increment)
    *   `user_id` (INT, FK -> users.user_id, Nullable)
    *   `input_features` (JSON) - *The inputs provided by user*
    *   `predicted_rent` (DECIMAL(12, 2))
    *   `model_version` (VARCHAR(50)) - *e.g., "v1.0-random-forest"*
    *   `created_at` (DATETIME, Default NOW())

## Relationships Summary
*   `users` 1 -- * `saved_searches`
*   `neighborhoods` 1 -- * `properties`
*   `properties` 1 -- * `rental_listings`
*   `neighborhoods` * -- * `amenities` (Many-to-Many)
