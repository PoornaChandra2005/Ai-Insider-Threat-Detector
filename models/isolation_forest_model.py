import joblib
from sklearn.ensemble import IsolationForest

class IsolationForestModel:
    """A wrapper for the scikit-learn IsolationForest model."""
    def __init__(self, n_estimators=100, contamination='auto', random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )

    def fit(self, X):
        """Trains the Isolation Forest model."""
        print("Training Isolation Forest model...")
        self.model.fit(X)
        print("Training complete.")

    def predict_scores(self, X):
        """
        Predicts anomaly scores for the input data.
        Scores are inverted to follow the convention: higher score = more anomalous.
        """
        scores = self.model.score_samples(X)
        # Invert scores: original scores are higher for normal points.
        # We want higher scores for anomalies.
        return -1 * scores

    def save(self, filepath):
        """Saves the trained model to a file."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads a model from a file."""
        model_instance = cls()
        model_instance.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model_instance