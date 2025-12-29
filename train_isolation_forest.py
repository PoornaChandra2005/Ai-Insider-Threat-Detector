import os
import argparse
import pandas as pd
from models.isolation_forest_model import IsolationForestModel

def main(data_path, model_dir, seed):
    """Main function to train the Isolation Forest model."""
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading training data from {data_path}...")
    train_df = pd.read_csv(data_path)
    
    # Exclude labels and user identifiers from training data
    features = [col for col in train_df.columns if col not in ['anomaly', 'user']]
    X_train = train_df[features]
    
    print(f"Training on {len(X_train)} samples with {len(features)} features.")
    
    # Initialize and train the model
    if_model = IsolationForestModel(random_state=seed)
    if_model.fit(X_train)
    
    # Save the trained model
    model_path = os.path.join(model_dir, 'isolation_forest.joblib')
    if_model.save(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an Isolation Forest model.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the training data CSV file (if_train.csv).")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    main(args.data_path, args.model_dir, args.seed)