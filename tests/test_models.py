import os
import pytest
import numpy as np
import tensorflow as tf
from models.isolation_forest_model import IsolationForestModel
from models.lstm_autoencoder import LSTMAutoencoder

@pytest.fixture
def synthetic_data():
    """Generates synthetic data for testing models."""
    # For Isolation Forest (2D)
    X_if = np.random.rand(100, 10)
    # For LSTM Autoencoder (3D)
    X_lstm = np.random.rand(100, 10, 5) # (samples, timesteps, features)
    return X_if, X_lstm

def test_isolation_forest_pipeline(tmpdir, synthetic_data):
    """Tests the fit, predict, save, and load pipeline for IsolationForestModel."""
    X_if, _ = synthetic_data
    model_path = os.path.join(tmpdir, 'if_model.joblib')
    
    # Train and save
    model = IsolationForestModel(n_estimators=10, random_state=42)
    model.fit(X_if)
    model.save(model_path)
    assert os.path.exists(model_path)
    
    # Load and predict
    loaded_model = IsolationForestModel.load(model_path)
    scores = loaded_model.predict_scores(X_if)
    
    assert scores.shape == (100,)
    assert not np.isnan(scores).any()

def test_lstm_autoencoder_pipeline(tmpdir, synthetic_data):
    """Tests the train, predict, save, and load pipeline for LSTMAutoencoder."""
    _, X_lstm = synthetic_data
    model_path = os.path.join(tmpdir, 'lstm_model.keras')
    
    # Initialize and train for one epoch
    timesteps = X_lstm.shape[1]
    n_features = X_lstm.shape[2]
    model = LSTMAutoencoder(timesteps=timesteps, n_features=n_features, latent_dim=8)
    
    # Mock training call
    model.model.fit(X_lstm, X_lstm, epochs=1, batch_size=16, verbose=0)
    
    # Save model
    model.save(model_path)
    assert os.path.exists(model_path)
    
    # Load and predict
    loaded_model = LSTMAutoencoder.load(model_path)
    errors = loaded_model.predict_reconstruction_error(X_lstm)
    
    assert errors.shape == (100,)
    assert not np.isnan(errors).any()
    assert loaded_model.timesteps == timesteps
    assert loaded_model.n_features == n_features