import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import MinMaxScaler
from models.xgboost_model import XGBoostModel
from models.lstm_autoencoder import LSTMAutoencoder

def plot_anomaly_distribution(scores_df, model_name, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=scores_df, x='score', hue='label', kde=True, bins=50, palette=['#3498db', '#e74c3c'])
    plt.title(f'Anomaly Score Distribution for {model_name}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved score distribution plot to {save_path}")

def plot_roc_curve(y_true, scores, model_name, save_path):
    if len(np.unique(y_true)) < 2: return
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curve plot to {save_path}")

def plot_pr_curve(y_true, scores, model_name, save_path):
    if len(np.unique(y_true)) < 2: return
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:0.2f})')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PR curve plot to {save_path}")

def evaluate_model(y_true, scores, model_name, threshold_percentile=90):
    """Calculates and prints performance metrics."""
    if len(scores) == 0:
        print(f"\n--- {model_name} Final Metrics ---")
        print("No data available for evaluation.")
        return {'model': model_name, 'error': 'No data'}

    threshold = np.percentile(scores, threshold_percentile)
    y_pred = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {'model': model_name, 'threshold': threshold, 'precision': precision, 'recall': recall, 'f1_score': f1, 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'confusion_matrix': cm.tolist()}
    print(f"\n--- {model_name} Final Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics

def main(xgb_data_path, lstm_data_path, lstm_ids_path, model_dir, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    all_metrics = {}

    # --- Evaluate XGBoost ---
    print("\n--- Evaluating XGBoost ---")
    xgb_test_df = pd.read_csv(xgb_data_path)
    y_xgb_true = xgb_test_df['anomaly']
    X_xgb_test = xgb_test_df.drop(columns=['anomaly', 'user', 'date'])
    xgb_model = XGBoostModel.load(os.path.join(model_dir, 'xgboost_model.joblib'))
    xgb_scores = xgb_model.predict_proba(X_xgb_test)
    all_metrics['xgboost'] = evaluate_model(y_xgb_true, xgb_scores, 'XGBoost')
    
    # --- Evaluate LSTM ---
    print("\n--- Evaluating LSTM Autoencoder ---")
    with np.load(lstm_data_path) as data:
        X_lstm_test, y_lstm_true = data['X'], data.get('y') # Use .get for safety
    lstm_model = LSTMAutoencoder.load(os.path.join(model_dir, 'lstm_autoencoder.keras'))
    lstm_scores = lstm_model.predict_reconstruction_error(X_lstm_test)
    all_metrics['lstm_autoencoder'] = evaluate_model(y_lstm_true, lstm_scores, 'LSTMAutoencoder')
    
    # --- Create Ensemble ---
    print("\n--- Evaluating Ensemble Model ---")
    xgb_results = xgb_test_df[['user', 'date', 'anomaly']].copy()
    xgb_results['xgb_score'] = xgb_scores
    xgb_results['date'] = pd.to_datetime(xgb_results['date'])

    lstm_ids_df = pd.read_csv(lstm_ids_path)
    lstm_results = lstm_ids_df.copy()
    lstm_results['lstm_score'] = lstm_scores
    lstm_results['date'] = pd.to_datetime(lstm_results['date'])
    
    ensemble_df = pd.merge(xgb_results, lstm_results, on=['user', 'date'], how='left')
    
    scaler = MinMaxScaler()
    ensemble_df[['xgb_score', 'lstm_score']] = scaler.fit_transform(ensemble_df[['xgb_score', 'lstm_score']])
    ensemble_df['lstm_score'].fillna(ensemble_df['xgb_score'], inplace=True)
    ensemble_df['ensemble_score'] = (ensemble_df['xgb_score'] * 0.6) + (ensemble_df['lstm_score'] * 0.4) # Weighted average
    
    all_metrics['ensemble'] = evaluate_model(ensemble_df['anomaly'], ensemble_df['ensemble_score'], 'Ensemble')
    
    # --- Save Metrics & Generate Plots ---
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")

    # Plotting
    xgb_scores_df = pd.DataFrame({'score': xgb_scores, 'label': y_xgb_true})
    lstm_scores_df = pd.DataFrame({'score': lstm_scores, 'label': y_lstm_true})
    ensemble_scores_df = pd.DataFrame({'score': ensemble_df['ensemble_score'], 'label': ensemble_df['anomaly']})

    plot_anomaly_distribution(xgb_scores_df, 'XGBoost', os.path.join(plots_dir, 'xgboost_distribution.png'))
    plot_roc_curve(y_xgb_true, xgb_scores, 'XGBoost', os.path.join(plots_dir, 'xgboost_roc_curve.png'))
    plot_pr_curve(y_xgb_true, xgb_scores, 'XGBoost', os.path.join(plots_dir, 'xgboost_pr_curve.png'))

    plot_anomaly_distribution(lstm_scores_df, 'LSTMAutoencoder', os.path.join(plots_dir, 'lstm_autoencoder_distribution.png'))
    plot_roc_curve(y_lstm_true, lstm_scores, 'LSTMAutoencoder', os.path.join(plots_dir, 'lstm_autoencoder_roc_curve.png'))
    plot_pr_curve(y_lstm_true, lstm_scores, 'LSTMAutoencoder', os.path.join(plots_dir, 'lstm_autoencoder_pr_curve.png'))

    plot_anomaly_distribution(ensemble_scores_df, 'Ensemble', os.path.join(plots_dir, 'ensemble_distribution.png'))
    plot_roc_curve(ensemble_df['anomaly'], ensemble_df['ensemble_score'], 'Ensemble', os.path.join(plots_dir, 'ensemble_roc_curve.png'))
    plot_pr_curve(ensemble_df['anomaly'], ensemble_df['ensemble_score'], 'Ensemble', os.path.join(plots_dir, 'ensemble_pr_curve.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate models and generate visualizations.")
    parser.add_argument('--xgb-data', type=str, required=True, help="Path to the XGB test data CSV.")
    parser.add_argument('--lstm-data', type=str, required=True, help="Path to the LSTM test data NPZ.")
    parser.add_argument('--lstm-ids-path', type=str, required=True, help="Path to the LSTM test identifiers CSV.")
    parser.add_argument('--model-dir', type=str, required=True, help="Directory where trained models are stored.")
    parser.add_argument('--results-dir', type=str, required=True, help="Directory to save evaluation results.")
    args = parser.parse_args()
    main(args.xgb_data, args.lstm_data, args.lstm_ids_path, args.model_dir, args.results_dir)