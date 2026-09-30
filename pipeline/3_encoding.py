"""
encoding.py
-----------
Stage 3 of the pipeline.

Loads the preprocessed, product-level data, one-hot encodes the categorical
feature, splits into train/test, and scales the continuous numeric features
(fit on train only, to avoid data leakage).

Input:
    outputs/preprocessing/preprocessed_data.csv   (override with --input)

Output (written to outputs/encoding/):
    train.csv               - scaled/encoded training features + target
    test.csv                - scaled/encoded test features + target
    onehot_encoder.pkl       - fitted OneHotEncoder (joblib)
    scaler.pkl               - fitted StandardScaler (joblib)
    feature_columns.json     - ordered list of final feature column names
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = 'sale_price'
CATEGORICAL_COL = 'category'
CONTINUOUS_COLS = ['product_count', 'subcat_count', 'rating', 'market_log', 'brandC_log']
ID_COLS = ['product']


def load_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        sys.exit(f"[encoding] ERROR: input file not found: {input_path}. Run preprocessing.py first.")
    return pd.read_csv(input_path)


def one_hot_encode(df: pd.DataFrame):
    ohe = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoded = ohe.fit_transform(df[[CATEGORICAL_COL]])
    cat_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out([CATEGORICAL_COL]),
        index=df.index,
    )
    combined = pd.concat(
        [df.drop(columns=[CATEGORICAL_COL] + [c for c in ID_COLS if c in df.columns]), cat_df],
        axis=1,
    )
    return combined, ohe


def scale_continuous_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_test_scaled[CONTINUOUS_COLS] = scaler.transform(X_test[CONTINUOUS_COLS])

    return X_train_scaled, X_test_scaled, scaler


def main():
    parser = argparse.ArgumentParser(description="Encode and split preprocessed product data.")
    parser.add_argument('--input', default='outputs/preprocessing/preprocessed_data.csv',
                         help="Path to the preprocessed CSV file.")
    parser.add_argument('--output-dir', default='outputs/encoding',
                         help="Directory to write outputs to.")
    parser.add_argument('--test-size', type=float, default=0.2,
                         help="Fraction of data to hold out for testing.")
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[encoding] Loading preprocessed data from: {args.input}")
    df = load_data(args.input)

    print("[encoding] One-hot encoding categorical feature(s)...")
    combined, ohe = one_hot_encode(df)

    X = combined.drop(columns=[TARGET_COL])
    y = combined[TARGET_COL]

    print(f"[encoding] Splitting train/test (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    print("[encoding] Scaling continuous features (fit on train only)...")
    X_train_scaled, X_test_scaled, scaler = scale_continuous_features(X_train, X_test)

    train_out = X_train_scaled.copy()
    train_out[TARGET_COL] = y_train.values
    test_out = X_test_scaled.copy()
    test_out[TARGET_COL] = y_test.values

    train_path = os.path.join(args.output_dir, 'train.csv')
    test_path = os.path.join(args.output_dir, 'test.csv')
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    print(f"[encoding] Saved train set -> {train_path}  (shape={train_out.shape})")
    print(f"[encoding] Saved test set  -> {test_path}  (shape={test_out.shape})")

    ohe_path = os.path.join(args.output_dir, 'onehot_encoder.pkl')
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    joblib.dump(ohe, ohe_path)
    joblib.dump(scaler, scaler_path)
    print(f"[encoding] Saved fitted encoder -> {ohe_path}")
    print(f"[encoding] Saved fitted scaler  -> {scaler_path}")

    feature_cols_path = os.path.join(args.output_dir, 'feature_columns.json')
    with open(feature_cols_path, 'w') as f:
        json.dump(list(X_train_scaled.columns), f, indent=2)
    print(f"[encoding] Saved feature column list -> {feature_cols_path}")


if __name__ == '__main__':
    main()
