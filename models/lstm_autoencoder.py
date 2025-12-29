import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class LSTMAutoencoder:
    """
    An LSTM Autoencoder model for sequence anomaly detection.
    """
    def __init__(self, timesteps, n_features, latent_dim=32):
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.model = self._build_model()

    def _build_model(self):
        """Builds the Keras LSTM Autoencoder model."""
        # Encoder
        inputs = Input(shape=(self.timesteps, self.n_features))
        # REMOVED activation='relu' to use cuDNN optimized kernels
        encoder = LSTM(128, return_sequences=True)(inputs)
        encoder = LSTM(64, return_sequences=False)(encoder)
        latent_vector = Dense(self.latent_dim, activation='relu')(encoder)
        
        # Decoder
        decoder_input = RepeatVector(self.timesteps)(latent_vector)
        # REMOVED activation='relu' to use cuDNN optimized kernels
        decoder = LSTM(64, return_sequences=True)(decoder_input)
        decoder = LSTM(128, return_sequences=True)(decoder)
        output = TimeDistributed(Dense(self.n_features))(decoder)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mae') # Mean Absolute Error is common for reconstruction
        model.summary()
        return model

    def train(self, X_train, X_val, epochs, batch_size, model_dir):
        """Trains the model with early stopping and model checkpointing."""
        checkpoint_path = os.path.join(model_dir, 'lstm_autoencoder.keras')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        ]
        
        print("Training LSTM Autoencoder...")
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=callbacks,
            shuffle=True
        )
        print("Training complete.")
        # Load the best model saved by ModelCheckpoint
        self.model = load_model(checkpoint_path)
        return history

    def predict_reconstruction_error(self, X):
        """Calculates the reconstruction error for input sequences."""
        X_pred = self.model.predict(X)
        # Calculate Mean Absolute Error for each sequence
        errors = np.mean(np.abs(X_pred - X), axis=(1, 2))
        return errors

    def save(self, filepath):
        """Saves the trained Keras model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads a Keras model from a file."""
        print(f"Loading model from {filepath}...")
        model = load_model(filepath)
        # Infer parameters from the loaded model's input shape
        timesteps = model.input_shape[1]
        n_features = model.input_shape[2]
        latent_dim = model.layers[3].output_shape[1] # Infer latent dim from dense layer
        
        instance = cls(timesteps, n_features, latent_dim)
        instance.model = model
        print("Model loaded successfully.")
        return instance

def set_seed(seed):
    """Set random seed for reproducibility in TensorFlow."""
    tf.random.set_seed(seed)
    np.random.seed(seed)