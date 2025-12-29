import pandas as pd
import os

print("--- Starting Scores File Check ---")
scores_file_path = os.path.join('results', 'scores.csv')

if not os.path.exists(scores_file_path):
    print(f"ERROR: Cannot find the file at '{scores_file_path}'")
    print("Please run evaluate.py first.")
else:
    try:
        df = pd.read_csv(scores_file_path)
        print(f"Successfully loaded '{scores_file_path}'.\n")

        # Check data for Isolation Forest
        print("--- Checking Isolation Forest Labels ---")
        iso_df = df[df['model'] == 'IsolationForest']
        if not iso_df.empty:
            print(iso_df['label'].value_counts())
        else:
            print("No data found for Isolation Forest in scores.csv.")

        # Check data for LSTM Autoencoder
        print("\n--- Checking LSTM Autoencoder Labels ---")
        lstm_df = df[df['model'] == 'LSTMAutoencoder']
        if not lstm_df.empty:
            print(lstm_df['label'].value_counts())
        else:
            print("No data found for LSTM Autoencoder in scores.csv.")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

print("\n--- Scores Check Finished ---")