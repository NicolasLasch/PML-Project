import pandas as pd
from math import radians, cos, sin, asin, sqrt

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two geographic points

    Parameters:
    - lat1, lon1: latitude and longitude of the first point (degrees)
    - lat2, lon2: latitude and longitude of the second point (degrees)
    """
    # Convert decimal degrees to radians for trigonometric functions
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Compute differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of Earth in km
    r = 6371
    return c * r


def get_nearby_stations(user_lat, user_lon, stations_df, radius_km=2.0):
    """
    Find stations within a given radius from the user's location

    Parameters:
    - user_lat, user_lon: User's latitude and longitude (degrees)
    - stations_df: DataFrame containing station locations (columns 'start_station_latitude', 'start_station_longitude')
    - radius_km: Radius distance in kilometers to filter nearby stations (default 2 km)
    """
    stations = stations_df.copy()

    # Calculate distance from user location to each station
    stations['distance_km'] = stations.apply(
        lambda row: calculate_distance(
            user_lat,
            user_lon, 
            row['start_station_latitude'], row['start_station_longitude']
        ),
        axis = 1
    )
    # Filter sections within radius and sort by closest
    nearby = stations[
        stations['distance_km'] <= radius_km
    ].sort_values('distance_km')
    return nearby


def calculate_recommendation_score(distance, predicted_trips, max_distance=2.0):
    """
    Calculate a combined recommendation score based on proximity and predicted usage

    Parameters:
    - distance: Distance to the station in kilometers
    - predicted_trips: Predicted number of trips/usage at the station
    - max_distance: Maximum distance considered for normalization (default 2 km)
    """
    distance_score = 1 - (distance / max_distance)
    # Normalize predicted trips, capped at 1
    availability_score = min(predicted_trips / 10, 1.0)
    # Weighted sum
    combined_score = 0.4 * distance_score + 0.6 * availability_score
    # Returns weighted score where closer stations and higher predicted trips score better
    return combined_score


def recommend_best_station(user_lat, user_lon, target_hour, stations_info, features_df, model, feature_cols, top_k=3):
    """
    Recommend the best stations near the user by predicting
    bike availability/demand and scoring

    Parameters:
    - user_lat, user_lon: User's latitude and longitude for locating nearby stations
    - target_hour: The hour of day for which predictions are made
    - stations_info: DataFrame with station metadata (id, name, latitude, longitude)
    - features_df: DataFrame containing feature values for all stations and hours
    - model: Trained machine learning model for prediction
    - feature_cols: List of feature column names used for prediction
    - top_k: Number of top recommendations to return (default 3)
    """
    # Extract unique stations for distance calculations
    unique_stations = stations_info[
        [
        'start_station_id',
        'start_station_name',
        'start_station_latitude',
        'start_station_longitude'
        ]
    ].drop_duplicates()
    # Get stations close to user within 2km radius
    nearby = get_nearby_stations(
        user_lat, user_lon, unique_stations, radius_km = 2.0
    )
    
    if nearby.empty:
        # Empty if no nearby stations found
        return pd.DataFrame()
    
    recommendations = []
    for _, station in nearby.iterrows():
        # Filter features for the station and target hour
        station_features = features_df[
            (features_df['start_station_id'] == station['start_station_id']) &
            (features_df['hour'] == target_hour)
        ]
        
        # Skip if no features available for that hour
        if station_features.empty:
            continue
        
        # Latest fetures and select only model feature columns
        latest_features = station_features.iloc[-1:][feature_cols]
        # Predict expected trips at this station and hour
        predicted_trips = model.predict(latest_features)[0]
        # Combined recommendation score
        score = calculate_recommendation_score(station['distance_km'], predicted_trips)
        # Station info and recommendation scores
        recommendations.append({
            'station_id': station['start_station_id'],
            'station_name': station['start_station_name'],
            'latitude': station['start_station_latitude'],
            'longitude': station['start_station_longitude'],
            'distance_km': station['distance_km'],
            'predicted_trips': predicted_trips,
            'recommendation_score': score
        })
    
    rec_df = pd.DataFrame(recommendations)
    # Sort top_k values by descending recommendation score
    rec_df = rec_df.sort_values('recommendation_score', ascending=False).head(top_k)
    return rec_df


def format_recommendation_output(recommendations):
    """
    Format recommendation DataFrame into readable string output
    """
    if recommendations.empty:
        return "No stations found nearby. Try increasing the search radius."
    
    output_lines = ["Top Station Recommendations:\n"]
    for idx, row in recommendations.iterrows():
        output_lines.append(f"Station: {row['station_name']}")
        output_lines.append(f"  Distance: {row['distance_km']:.2f} km")
        output_lines.append(f"  Predicted Activity: {row['predicted_trips']:.1f} trips/hour")
        output_lines.append(f"  Score: {row['recommendation_score']:.3f}")
        output_lines.append("")
    return "\n".join(output_lines)