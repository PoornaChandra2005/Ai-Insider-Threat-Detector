import pandas as pd
import numpy as np
import joblib
import argparse
from models.xgboost_model import XGBoostModel

def run_simulation(user_id, model_path, scaler_path, data_path, threshold_percentile=90):
    """
    Simulates a real-world analyst monitoring a specific user day-by-day.
    """
    print(f"--- Starting Analyst Simulation for User: {user_id} ---\n")

    # --- Load all necessary components ---
    try:
        model = XGBoostModel.load(model_path)
        scaler = joblib.load(scaler_path)
        full_df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Error: Could not load required file. {e}")
        print("Please ensure you have run the full pipeline first to generate the necessary files.")
        return

    # Filter data for the specified user
    user_df = full_df[full_df['user'] == user_id].sort_values(by='date').reset_index(drop=True)

    if user_df.empty:
        print(f"Error: No data found for user '{user_id}'. Please choose a different user.")
        return

    # Get the feature names from the scaler
    # Assumes scaler was fit on a DataFrame with the feature columns
    # We need to get the feature names to align the data correctly
    if hasattr(scaler, 'feature_names_in_'):
        features = scaler.feature_names_in_
    else: # Fallback for older scalers
        X_if_test = user_df.drop(columns=['anomaly', 'user', 'date'])
        features = X_if_test.columns

    # Determine the anomaly score threshold from the model's training on normal data
    # (This is a simplified approach; a real system might use a more robust method)
    # For this simulation, we will use a fixed high score as the threshold
    # Let's find the threshold from the test set scores
    y_true = user_df['anomaly']
    X_test = user_df[features]
    scores = model.predict_proba(X_test)
    threshold = np.percentile(scores, threshold_percentile)
    
    print(f"Alerting threshold set at {threshold:.4f} (the {threshold_percentile}th percentile of this user's scores).\n")

    # --- Replay the user's activity day-by-day ---
    for index, day_data in user_df.iterrows():
        date = day_data['date']
        
        # Prepare the single day's data for prediction
        day_features = pd.DataFrame([day_data[features]], columns=features)
        
        # Get the anomaly score
        anomaly_score = model.predict_proba(day_features)[0]

        # Check if the score triggers an alert
        if anomaly_score >= threshold:
            print(f"--- 🚨 ALERT TRIGGERED 🚨 ---")
            print(f"Date: {date}")
            print(f"User: {user_id}")
            print(f"Anomaly Score: {anomaly_score:.4f} (Threshold: {threshold:.4f})")
            print("\nDaily Activity Report:")
            
            # Show the most significant features for this day
            significant_features = day_data[features][day_data[features] > 0].sort_values(ascending=False)
            print(significant_features.head(10))
            print("\n--- END OF ALERT ---\n")
        else:
            # Optional: Print normal days for a full timeline
            # print(f"Date: {date} | User: {user_id} | Score: {anomaly_score:.4f} (Normal)")
            pass

    print(f"--- Simulation for User {user_id} Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate real-world monitoring for a specific user.")
    parser.add_argument('--user-id', type=str, required=True, help="The user ID to monitor (e.g., 'MOH0530').")
    
    # Set default paths assuming the script is run from the project root
    parser.add_argument('--model-path', type=str, default='trained_models/xgboost_model.joblib')
    parser.add_argument('--scaler-path', type=str, default='processed_data/daily_summary_scaler.joblib')
    parser.add_argument('--data-path', type=str, default='processed_data/if_test.csv')
    parser.add_argument('--threshold', type=int, default=90, help="The percentile to use as an alerting threshold (e.g., 90).")

    args = parser.parse_args()
    
    run_simulation(args.user_id, args.model_path, args.scaler_path, args.data_path, args.threshold)