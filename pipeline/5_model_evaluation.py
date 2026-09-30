"""
model_evaluation.py
--------------------
Stage 5 (final) of the pipeline.

Loads the test-set predictions produced by model_building.py, computes
regression metrics for every model, ranks them, and produces diagnostic
plots for the best-performing model.

Input:
    outputs/model_building/predictions.csv   (override with --input)

Output (written to outputs/model_evaluation/):
    metrics.csv                    - RMSE / MAE / R2 / MAPE per model, ranked
    model_comparison.png           - bar chart comparing R2 across models
    actual_vs_predicted_best.png   - scatter plot for the best model
    residuals_best.png             - residual plot for the best model
    evaluation_summary.txt         - best model + headline numbers
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_predictions(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        sys.exit(f"[model_evaluation] ERROR: input file not found: {input_path}. "
                  f"Run model_building.py first.")
    return pd.read_csv(input_path)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions_df['actual']
    model_names = [c for c in predictions_df.columns if c != 'actual']

    rows = []
    for name in model_names:
        y_pred = predictions_df[name]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rows.append({
            "Model": name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2 Score": round(r2, 4),
            "MAPE (%)": round(mape, 2),
        })

    metrics_df = pd.DataFrame(rows).sort_values('R2 Score', ascending=False).reset_index(drop=True)
    return metrics_df


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))
    order = metrics_df.sort_values('R2 Score', ascending=True)
    plt.barh(order['Model'], order['R2 Score'], color='steelblue')
    plt.xlabel('R2 Score')
    plt.title('Model Comparison (R2 Score, higher is better)')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[model_evaluation] Saved model comparison chart -> {out_path}")


def plot_actual_vs_predicted(predictions_df: pd.DataFrame, best_model: str, output_dir: str):
    plt.figure(figsize=(7, 7))
    y_true = predictions_df['actual']
    y_pred = predictions_df[best_model]
    plt.scatter(y_true, y_pred, alpha=0.4, s=15)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel('Actual sale_price')
    plt.ylabel('Predicted sale_price')
    plt.title(f'Actual vs Predicted \u2014 {best_model}')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'actual_vs_predicted_best.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[model_evaluation] Saved actual-vs-predicted plot -> {out_path}")


def plot_residuals(predictions_df: pd.DataFrame, best_model: str, output_dir: str):
    plt.figure(figsize=(8, 5))
    y_true = predictions_df['actual']
    y_pred = predictions_df[best_model]
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.4, s=15)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Predicted sale_price')
    plt.ylabel('Residual (actual - predicted)')
    plt.title(f'Residual Plot \u2014 {best_model}')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'residuals_best.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[model_evaluation] Saved residual plot -> {out_path}")


def write_summary(metrics_df: pd.DataFrame, best_model: str, output_dir: str):
    best_row = metrics_df.iloc[0]
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("=========================\n\n")
        f.write(f"Best model: {best_model}\n\n")
        f.write(f"RMSE:     {best_row['RMSE']}\n")
        f.write(f"MAE:      {best_row['MAE']}\n")
        f.write(f"R2 Score: {best_row['R2 Score']}\n")
        f.write(f"MAPE:     {best_row['MAPE (%)']}%\n\n")
        f.write("Full ranking:\n")
        f.write(metrics_df.to_string(index=False))
    print(f"[model_evaluation] Saved evaluation summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare trained regression models.")
    parser.add_argument('--input', default='outputs/model_building/predictions.csv')
    parser.add_argument('--output-dir', default='outputs/model_evaluation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[model_evaluation] Loading predictions from: {args.input}")
    predictions_df = load_predictions(args.input)

    print("[model_evaluation] Computing metrics...")
    metrics_df = compute_metrics(predictions_df)

    metrics_path = os.path.join(args.output_dir, 'metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[model_evaluation] Saved metrics table -> {metrics_path}")
    print(metrics_df.to_string(index=False))

    best_model = metrics_df.iloc[0]['Model']
    print(f"[model_evaluation] Best model: {best_model}")

    plot_model_comparison(metrics_df, args.output_dir)
    plot_actual_vs_predicted(predictions_df, best_model, args.output_dir)
    plot_residuals(predictions_df, best_model, args.output_dir)
    write_summary(metrics_df, best_model, args.output_dir)

    print("[model_evaluation] Done.")


if __name__ == '__main__':
    main()
