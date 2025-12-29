import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def plot_anomaly_distribution(scores_df, model_name, save_path):
    """Plots the distribution of anomaly scores for normal and anomaly classes."""
    plt.figure(figsize=(10, 6))
    data = scores_df[scores_df['model'] == model_name]
    sns.histplot(data=data, x='score', hue='label', kde=True, bins=50, palette=['#3498db', '#e74c3c'])
    plt.title(f'Anomaly Score Distribution for {model_name}')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend(title='Class', labels=['Anomaly (1)', 'Normal (0)'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved score distribution plot to {save_path}")

def plot_roc_curve(y_true, scores, model_name, save_path):
    """Plots the Receiver Operating Characteristic (ROC) curve."""
    # ADD THIS CHECK: Ensure there are both positive and negative samples
    if y_true.nunique() < 2:
        print(f"Skipping ROC curve for {model_name} because the data only contains one class.")
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curve plot to {save_path}")

def plot_pr_curve(y_true, scores, model_name, save_path):
    """Plots the Precision-Recall (PR) curve."""
    # ADD THIS CHECK: Ensure there are both positive and negative samples
    if y_true.nunique() < 2:
        print(f"Skipping PR curve for {model_name} because the data only contains one class.")
        return
        
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:0.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved PR curve plot to {save_path}")

def main(metrics_path, scores_path, results_dir):
    """Main function to generate and save all visualizations."""
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    scores_df = pd.read_csv(scores_path)

    for model_name in metrics.keys():
        print(f"\n--- Generating plots for {model_name} ---")
        model_scores = scores_df[scores_df['model'] == model_name]
        y_true = model_scores['label']
        scores = model_scores['score']
        
        # Plot distribution
        dist_path = os.path.join(plots_dir, f'{model_name}_score_distribution.png')
        plot_anomaly_distribution(scores_df, model_name, dist_path)
        
        # Plot ROC curve
        roc_path = os.path.join(plots_dir, f'{model_name}_roc_curve.png')
        plot_roc_curve(y_true, scores, model_name, roc_path)
        
        # Plot PR curve
        pr_path = os.path.join(plots_dir, f'{model_name}_pr_curve.png')
        plot_pr_curve(y_true, scores, model_name, pr_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate visualizations from evaluation results.")
    parser.add_argument('--metrics-path', type=str, required=True, help="Path to the metrics.json file.")
    parser.add_argument('--scores-path', type=str, required=True, help="Path to the scores.csv file.")
    parser.add_argument('--results-dir', type=str, required=True, help="Directory to save the plots.")
    args = parser.parse_args()
    
    main(args.metrics_path, args.scores_path, args.results_dir)