import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from models.isolation_forest_model import IsolationForestModel
from models.lstm_autoencoder import LSTMAutoencoder

def evaluate_model(y_true, scores, model_name):
    """
    Calculates and returns evaluation metrics for a given model.
    A percentile-based threshold is used to convert scores to binary predictions.
    """
    # Use 95th percentile of scores as the anomaly threshold
    threshold = np.percentile(scores, 95)
    y_pred = (scores > threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'model': model_name,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist() # [[TN, FP], [FN, TP]]
    }
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Threshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return metrics

def main(if_data_path, lstm_data_path, model_dir, results_dir):
    """Main function to evaluate both models."""
    os.makedirs(results_dir, exist_ok=True)
    all_metrics = {}
    all_scores = []
    
    # --- Evaluate Isolation Forest ---
    print("Evaluating Isolation Forest...")
    if_test_df = pd.read_csv(if_data_path)
    X_if_test = if_test_df.drop(columns=['anomaly', 'user'])
    y_if_true = if_test_df['anomaly']
    
    if_model = IsolationForestModel.load(os.path.join(model_dir, 'isolation_forest.joblib'))
    if_scores = if_model.predict_scores(X_if_test)
    
    if_metrics = evaluate_model(y_if_true, if_scores, 'IsolationForest')
    all_metrics['isolation_forest'] = if_metrics
    
    all_scores.append(pd.DataFrame({'score': if_scores, 'label': y_if_true, 'model': 'IsolationForest'}))

    # --- Evaluate LSTM Autoencoder ---
    print("\nEvaluating LSTM Autoencoder...")
    with np.load(lstm_data_path) as data:
        X_lstm_test, y_lstm_true = data['X'], data['y']
    
    lstm_model = LSTMAutoencoder.load(os.path.join(model_dir, 'lstm_autoencoder.keras'))
    lstm_scores = lstm_model.predict_reconstruction_error(X_lstm_test)

    lstm_metrics = evaluate_model(y_lstm_true, lstm_scores, 'LSTMAutoencoder')
    all_metrics['lstm_autoencoder'] = lstm_metrics
    
    all_scores.append(pd.DataFrame({'score': lstm_scores, 'label': y_lstm_true, 'model': 'LSTMAutoencoder'}))
    
    # --- Save Results ---
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    scores_df = pd.concat(all_scores, ignore_index=True)
    scores_path = os.path.join(results_dir, 'scores.csv')
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores saved to {scores_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument('--if-data', type=str, required=True, help="Path to the IF test data CSV.")
    parser.add_argument('--lstm-data', type=str, required=True, help="Path to the LSTM test data NPZ.")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory where trained models are stored.")
    parser.add_argument('--results-dir', type=str, required=True, help="Directory to save evaluation results.")
    args = parser.parse_args()
    
    main(args.if_data, args.lstm_data, args.model_dir, args.results_dir)