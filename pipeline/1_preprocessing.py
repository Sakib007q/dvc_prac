"""
preprocessing.py
----------------
Stage 1 of the pipeline.

Loads the raw BigBasket Products CSV, cleans it, engineers frequency-count
features, aggregates to one row per product, and log-transforms skewed
numeric columns.

Input:
    data/raw/BigBasket Products.csv   (override with --input)

Output (written to outputs/preprocessing/):
    preprocessed_data.csv   - one row per product, cleaned + engineered
    preprocessing_report.txt - shapes, missing-value counts, summary stats
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

RAW_COLUMNS = ['product', 'category', 'sub_category', 'brand',
               'sale_price', 'market_price', 'rating']


def load_raw_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        sys.exit(f"[preprocessing] ERROR: input file not found: {input_path}")
    df = pd.read_csv(input_path, index_col=False)
    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"[preprocessing] ERROR: missing expected columns: {missing_cols}")
    return df[RAW_COLUMNS]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['product', 'brand']).copy()
    df['rating'] = df['rating'].fillna(df['rating'].mean()).astype(np.float32).round(2)
    return df


def engineer_row_level_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['product_count'] = df['product'].map(df['product'].value_counts())
    df['brand_count'] = df['brand'].map(df['brand'].value_counts())
    df['subcat_count'] = df['sub_category'].map(df['sub_category'].value_counts())
    return df


def aggregate_to_product_level(df: pd.DataFrame) -> pd.DataFrame:
    Pdf = df.groupby('product').agg({
        'category': 'first',
        'product_count': 'first',
        'brand_count': 'mean',
        'subcat_count': 'mean',
        'rating': 'mean',
        'market_price': 'mean',
        'sale_price': 'mean',
    }).reset_index()
    return Pdf


def log_transform(Pdf: pd.DataFrame) -> pd.DataFrame:
    Pdf = Pdf.copy()
    Pdf['market_log'] = np.log1p(Pdf['market_price'])
    Pdf['brandC_log'] = np.log1p(Pdf['brand_count'])
    Pdf = Pdf.drop(columns=['brand_count', 'market_price'])
    return Pdf


def write_report(report_path: str, raw_shape, clean_shape, final_df: pd.DataFrame, null_before, null_after):
    with open(report_path, 'w') as f:
        f.write("PREPROCESSING REPORT\n")
        f.write("=====================\n\n")
        f.write(f"Raw shape (selected columns): {raw_shape}\n")
        f.write(f"Shape after dropping rows with missing product/brand: {clean_shape}\n")
        f.write(f"Final product-level shape: {final_df.shape}\n\n")
        f.write("Missing values BEFORE cleaning:\n")
        f.write(null_before.to_string())
        f.write("\n\nMissing values AFTER cleaning:\n")
        f.write(null_after.to_string())
        f.write("\n\nFinal column dtypes:\n")
        f.write(final_df.dtypes.to_string())
        f.write("\n\nSummary statistics (numeric columns):\n")
        f.write(final_df.describe().to_string())
    


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw BigBasket product data.")
    parser.add_argument('--input', default='data/raw/BigBasket Products.csv',
                         help="Path to the raw CSV file.")
    parser.add_argument('--output-dir', default='outputs/preprocessing',
                         help="Directory to write outputs to.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[preprocessing] Loading raw data from: {args.input}")
    raw_df = load_raw_data(args.input)
    raw_shape = raw_df.shape
    null_before = raw_df.isnull().sum()

    print("[preprocessing] Cleaning data...")
    clean_df = clean_data(raw_df)
    clean_shape = clean_df.shape
    null_after = clean_df.isnull().sum()

    print("[preprocessing] Engineering frequency features...")
    feat_df = engineer_row_level_features(clean_df)

    print("[preprocessing] Aggregating to one row per product...")
    Pdf = aggregate_to_product_level(feat_df)

    print("[preprocessing] Log-transforming skewed columns...")
    final_df = log_transform(Pdf)

    out_csv = os.path.join(args.output_dir, 'preprocessed_data.csv')
    final_df.to_csv(out_csv, index=False)
    print(f"[preprocessing] Saved preprocessed data -> {out_csv}  (shape={final_df.shape})")

    report_path = os.path.join(args.output_dir, 'preprocessing_report.txt')
    write_report(report_path, raw_shape, clean_shape, final_df, null_before, null_after)
    print(f"[preprocessing] Saved report -> {report_path}")


if __name__ == '__main__':
    main()
