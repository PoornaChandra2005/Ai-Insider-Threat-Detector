import os
import argparse
import pandas as pd
from models.xgboost_model import XGBoostModel

def main(data_path, model_dir, seed):
    """Main function to train the XGBoost model."""
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading training data from {data_path}...")
    train_df = pd.read_csv(data_path)
    
    # XGBoost is supervised, so we need both features (X) and labels (y)
    y_train = train_df['anomaly']
    X_train = train_df.drop(columns=['user', 'date', 'anomaly'])
    
    print(f"Training on {len(X_train)} samples with {len(X_train.columns)} features.")
    
    # Initialize and train the model
    # We can calculate scale_pos_weight for better handling of imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = XGBoostModel(scale_pos_weight=scale_pos_weight)
    
    xgb_model.fit(X_train, y_train)
    
    # Save the trained model
    model_path = os.path.join(model_dir, 'xgboost_model.joblib')
    xgb_model.save(model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an XGBoost model.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the training data CSV file.")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    main(args.data_path, args.model_dir, args.seed)