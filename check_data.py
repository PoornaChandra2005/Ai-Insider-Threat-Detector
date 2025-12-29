import pandas as pd
import os

print("--- Starting Diagnostic Check ---")
test_file_path = os.path.join('processed_data', 'if_test.csv')

if not os.path.exists(test_file_path):
    print(f"ERROR: Cannot find the file at '{test_file_path}'")
    print("Please run data_preprocess.py first.")
else:
    try:
        df = pd.read_csv(test_file_path)
        print(f"Successfully loaded '{test_file_path}'.")
        print("Shape of the test data:", df.shape)
        
        if 'anomaly' in df.columns:
            print("\nValue counts for the 'anomaly' column:")
            print(df['anomaly'].value_counts())
        else:
            print("ERROR: The 'anomaly' column is missing from the test file.")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

print("\n--- Diagnostic Check Finished ---")