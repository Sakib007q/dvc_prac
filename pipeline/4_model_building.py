"""
model_building.py
------------------
Stage 4 of the pipeline.

Loads the encoded train/test sets, trains several regression models, and
saves the fitted models plus their test-set predictions. Metric calculation
and comparison happens downstream in model_evaluation.py.

Input:
    outputs/encoding/train.csv   (override with --train)
    outputs/encoding/test.csv    (override with --test)

Output (written to outputs/model_building/):
    models/<model_name>.pkl   - one fitted model per algorithm (joblib)
    predictions.csv           - actual sale_price + each model's predictions on the test set
    training_log.txt          - which models were trained and how long each took
"""

import argparse
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

TARGET_COL = 'sale_price'


def build_model_registry(random_state: int = 42) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "SGD Regressor": SGDRegressor(max_iter=5000, tol=1e-3, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }


def load_split(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[model_building] ERROR: {name} file not found: {path}. Run encoding.py first.")
    return pd.read_csv(path)


def safe_filename(name: str) -> str:
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '')


def main():
    parser = argparse.ArgumentParser(description="Train regression models on encoded product data.")
    parser.add_argument('--train', default='outputs/encoding/train.csv')
    parser.add_argument('--test', default='outputs/encoding/test.csv')
    parser.add_argument('--output-dir', default='outputs/model_building')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    models_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print(f"[model_building] Loading train data from: {args.train}")
    train_df = load_split(args.train, 'train')
    print(f"[model_building] Loading test data from: {args.test}")
    test_df = load_split(args.test, 'test')

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    models = build_model_registry(args.random_state)

    predictions_df = pd.DataFrame({'actual': y_test.values})
    log_lines = [f"Training set shape: {X_train.shape}", f"Test set shape: {X_test.shape}", ""]

    for name, model in models.items():
        print(f"[model_building] Training: {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = model.predict(X_test)
        predictions_df[name] = preds

        model_path = os.path.join(models_dir, f"{safe_filename(name)}.pkl")
        joblib.dump(model, model_path)

        log_line = f"{name}: trained in {elapsed:.3f}s -> saved to {model_path}"
        print(f"[model_building] {log_line}")
        log_lines.append(log_line)

    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"[model_building] Saved predictions -> {predictions_path}")

    log_path = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines))
    print(f"[model_building] Saved training log -> {log_path}")


if __name__ == '__main__':
    main()
