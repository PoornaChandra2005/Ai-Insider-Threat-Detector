import xgboost as xgb
import joblib

class XGBoostModel:
    """A wrapper for the XGBoost classifier model."""
    def __init__(self, **params):
        if not params:
            # Default parameters that work well for imbalanced datasets
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'eta': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'use_label_encoder': False,
                'scale_pos_weight': 10 # Crucial for imbalanced data
            }
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X_train, y_train):
        """Trains the XGBoost model."""
        print("Training XGBoost model...")
        self.model.fit(X_train, y_train, verbose=True)
        print("Training complete.")

    def predict_proba(self, X):
        """Predicts the probability of being an anomaly (class 1)."""
        # We want the probability of the positive class (anomaly)
        return self.model.predict_proba(X)[:, 1]

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