import pandas as pd
import joblib
import numpy as np
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = os.path.join('trained_models', 'xgboost_model.joblib')
DATA_PATH = os.path.join('processed_data', 'xgb_test.csv')
THRESHOLD = 0.5  # Sensitivity of the detector

# ---------------------------------------------------------
# 1. LOAD RESOURCES
# ---------------------------------------------------------
print(f"Loading Model from: {MODEL_PATH}")
print(f"Loading Data from: {DATA_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Could not find the file. {e}")
    print("Make sure you have run 'data_preprocess.py' and 'train_xgboost.py' first.")
    exit()

# ---------------------------------------------------------
# 2. RUN PREDICTION ON ALL DATA (To find the rare threats)
# ---------------------------------------------------------
print("Scanning entire dataset to find anomalies...")

# Prepare features by dropping non-numeric/metadata columns
cols_to_drop = ['user', 'anomaly', 'day', 'day_id', 'date']
features = data.drop(columns=cols_to_drop, errors='ignore')

# Predict scores for EVERYONE
try:
    probs = model.predict_proba(features)[:, 1]
    data['model_score'] = probs  # Add score to the dataframe temporarily
except Exception as e:
    print(f"\n❌ PREDICTION ERROR: {e}")
    exit()

# ---------------------------------------------------------
# 3. SMART SAMPLING (Ensure we show both Threats and Safe users)
# ---------------------------------------------------------
# Sort data: Highest scores on top
sorted_data = data.sort_values(by='model_score', ascending=False)

# Pick top 3 "Threats" (Highest risk)
threats = sorted_data.head(3)

# Pick 3 random "Safe" users (from the bottom half of risks)
safe_pool = sorted_data[sorted_data['model_score'] < THRESHOLD]
if not safe_pool.empty:
    safe = safe_pool.sample(min(3, len(safe_pool)))
else:
    safe = pd.DataFrame()

# Combine them for the demo display
samples = pd.concat([threats, safe])

print("\n" + "="*60)
print("      INSIDER THREAT DETECTION SYSTEM - LIVE SCANNER")
print("="*60 + "\n")

print(f"{'User ID':<15} | {'Anomaly Score':<15} | {'Status':<20}")
print("-" * 60)

# ---------------------------------------------------------
# 4. DISPLAY RESULTS
# ---------------------------------------------------------
for i, (index, row) in enumerate(samples.iterrows()):
    # Get User ID for display (if it exists, otherwise use a placeholder)
    user_id = row.get('user', f"User_{index}")
    
    score = row['model_score']
    
    # Determine Status
    if score > THRESHOLD:
        status = "🔴 THREAT DETECTED"
    else:
        status = "🟢 Safe"
        
    print(f"{str(user_id):<15} | {score:.4f}          | {status}")

print("-" * 60)
print("\nSystem Scan Complete.")