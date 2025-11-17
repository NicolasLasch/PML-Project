import pandas as pd

# Run setup.py before executing to get dataframe if needed

try:
    features_df = pd.read_csv("data/processed/feature_engineered_bergen_merged.csv")

    # --- Hypo 1 : Higher activity during rush hours? ---
    # Average trips on rush hour [7, 8, 9, 16, 17, 18]
    rush_hour_trips = features_df[features_df['is_rush_hour'] == 1]['trip_count'].mean()
    non_rush_trips = features_df[features_df['is_rush_hour'] == 0]['trip_count'].mean()

    print("[ HYPOTHESIS 1: Higher activity during rush hours ]")
    print(f"  Average trips during rush hours: {rush_hour_trips:.2f}")
    print(f"  Average trips during non-rush hours: {non_rush_trips:.2f}")
    print(f"  Difference: {rush_hour_trips - non_rush_trips:.2f} more trips during rush hours")
    print(f"  Result: {'CONFIRMED' if rush_hour_trips > non_rush_trips else 'NOT CONFIRMED'}")


    # --- Hypo 2 : Temperature affecting bike rentals? ---
    # Correlation coefficient between temperature and trip counts
    temp_correlation = features_df[['temperature', 'trip_count']].corr().iloc[0, 1]

    print("\n[ HYPOTHESIS 2: Temperature affects bike rentals ]")
    print(f"  Correlation between temperature and trips: {temp_correlation:.4f}")
    print(f"  Result: {'CONFIRMED' if abs(temp_correlation) > 0.1 else 'NOT CONFIRMED'}")


    # --- Hypo 3 : Bike rentals will reduce if there is rain? ---
    # Avg number of trip_count with precipitation greater than 0 (rainy)
    rainy_trips = features_df[features_df['precipitation'] > 0]['trip_count'].mean()
    # Avg number of trips when there is no precipitation (dry)
    dry_trips = features_df[features_df['precipitation'] == 0]['trip_count'].mean()

    print("\n[ HYPOTHESIS 3: Rain reduces bike rentals ]")
    print(f"  Average trips during rain: {rainy_trips:.2f}")
    print(f"  Average trips when dry: {dry_trips:.2f}")
    # Percentage reduction
    print(f"  Reduction: {((dry_trips - rainy_trips) / dry_trips * 100):.1f}%")
    print(f"  Result: {'CONFIRMED' if rainy_trips < dry_trips else 'NOT CONFIRMED'}")


    # --- Hypo 4 : Are the trip counts for weekdays different from weekends? ---
    weekday_trips = features_df[features_df['is_weekend'] == 0]['trip_count'].mean()
    weekend_trips = features_df[features_df['is_weekend'] == 1]['trip_count'].mean()

    print("\n[ HYPOTHESIS 4: Trip counts on weekend differ from weekdays ]")
    print(f"  Average weekday trips: {weekday_trips:.2f}")
    print(f"  Average weekend trips: {weekend_trips:.2f}")
    print(f"  Difference: {abs(weekday_trips - weekend_trips):.2f} trips")
    print(f"  Result: {'CONFIRMED' if abs(weekday_trips - weekend_trips) > 0.5 else 'NOT CONFIRMED'}")


except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error}. Run setup.py before this script if needed.")

except Exception as e:  # Catch any other errors
    print(f"An unexpected error occurred: {e}. Make sure setup.py has run and all dependencies exist.")    