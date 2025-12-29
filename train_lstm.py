import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from models.lstm_autoencoder import LSTMAutoencoder, set_seed

def main(data_path, model_dir, epochs, batch_size, seed):
    """Main function to train the LSTM Autoencoder."""
    set_seed(seed)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading training data from {data_path}...")
    with np.load(data_path) as data:
        X = data['X']
    
    # Split data into training and validation sets
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=seed)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    
    # Initialize and train the model
    lstm_autoencoder = LSTMAutoencoder(timesteps=timesteps, n_features=n_features)
    history = lstm_autoencoder.train(X_train, X_val, epochs, batch_size, model_dir)
    
    # Model is already saved by ModelCheckpoint callback inside the train method.
    # We can re-save the final best model just in case.
    final_model_path = os.path.join(model_dir, 'lstm_autoencoder_final.keras')
    lstm_autoencoder.save(final_model_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an LSTM Autoencoder model.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the training data NPZ file (lstm_train.npz).")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=64, help="Training batch size.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    main(args.data_path, args.model_dir, args.epochs, args.batch_size, args.seed)