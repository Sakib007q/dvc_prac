"""
eda.py
------
Stage 2 of the pipeline.

Loads the preprocessed, product-level data and generates exploratory plots
and summary tables.

Input:
    outputs/preprocessing/preprocessed_data.csv   (override with --input)

Output (written to outputs/eda/):
    summary_statistics.csv
    numeric_distributions.png
    correlation_heatmap.png
    sale_price_by_category.png
    category_counts.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless / no display needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

NUMERIC_COLS = ['product_count', 'subcat_count', 'rating', 'sale_price', 'market_log', 'brandC_log']


def load_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        sys.exit(f"[eda] ERROR: input file not found: {input_path}. Run preprocessing.py first.")
    return pd.read_csv(input_path)


def save_summary_statistics(df: pd.DataFrame, output_dir: str):
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    summary = df[numeric_cols].describe().T
    out_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(out_path)
    print(f"[eda] Saved summary statistics -> {out_path}")


def plot_distributions(df: pd.DataFrame, output_dir: str):
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    n = len(numeric_cols)
    ncols = 3
    nrows = -(-n // ncols)  # ceil division

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(col)

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'numeric_distributions.png')
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[eda] Saved distribution plots -> {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (numeric features)')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[eda] Saved correlation heatmap -> {out_path}")


def plot_sale_price_by_category(df: pd.DataFrame, output_dir: str):
    if 'category' not in df.columns:
        return
    plt.figure(figsize=(12, 6))
    order = df.groupby('category')['sale_price'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='category', y='sale_price', order=order)
    plt.xticks(rotation=75, ha='right')
    plt.title('Sale Price Distribution by Category')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'sale_price_by_category.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[eda] Saved sale-price-by-category plot -> {out_path}")


def plot_category_counts(df: pd.DataFrame, output_dir: str):
    if 'category' not in df.columns:
        return
    plt.figure(figsize=(10, 6))
    df['category'].value_counts().plot(kind='barh')
    plt.title('Number of Products per Category')
    plt.xlabel('Count')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'category_counts.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[eda] Saved category counts plot -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run EDA on preprocessed product data.")
    parser.add_argument('--input', default='outputs/preprocessing/preprocessed_data.csv',
                         help="Path to the preprocessed CSV file.")
    parser.add_argument('--output-dir', default='outputs/eda',
                         help="Directory to write outputs to.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[eda] Loading preprocessed data from: {args.input}")
    df = load_data(args.input)

    save_summary_statistics(df, args.output_dir)
    plot_distributions(df, args.output_dir)
    plot_correlation_heatmap(df, args.output_dir)
    plot_sale_price_by_category(df, args.output_dir)
    plot_category_counts(df, args.output_dir)

    print("[eda] Done.")


if __name__ == '__main__':
    main()
