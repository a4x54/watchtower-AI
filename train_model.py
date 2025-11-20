import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from model_utils import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_FEATURE_ORDER,
    MODEL_PATH,
    SCALER_PATH,
    save_metadata,
)


FEATURE_ORDER = DEFAULT_FEATURE_ORDER


def load_training_data(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix.lower() == ".csv":
        import csv

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")

    normalized: List[Dict[str, float]] = []
    for row in data:
        normalized.append({key: float(row[key]) for key in FEATURE_ORDER})
    return normalized


def compute_threshold(decision_values: np.ndarray) -> float:
    mean = decision_values.mean()
    std = decision_values.std()
    threshold = float(mean - 2 * std)
    return threshold


def train(
    data_path: Path,
    manual_threshold: float | None = None,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> None:
    records = load_training_data(data_path)
    feature_matrix = np.array(
        [[rec[feature] for feature in FEATURE_ORDER] for rec in records], dtype=float
    )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    svm.fit(scaled_features)

    decision_values = svm.decision_function(scaled_features)
    computed_threshold = compute_threshold(decision_values)
    threshold_to_use = manual_threshold if manual_threshold is not None else computed_threshold

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    save_metadata(FEATURE_ORDER, computed_threshold, threshold_to_use)

    print("Training complete")
    print(f"Data path: {data_path}")
    print(f"Samples: {len(records)} | Features: {len(FEATURE_ORDER)}")
    print(f"Decision function mean: {decision_values.mean():.4f} std: {decision_values.std():.4f}")
    print(f"Computed threshold (mean - 2*std): {computed_threshold:.4f}")
    if manual_threshold is not None:
        print(f"Manual threshold override provided: {manual_threshold:.4f}")
    print(f"Using threshold: {threshold_to_use:.4f}")
    print(f"Artifacts saved to {artifact_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train One-Class SVM for anomaly detection on normal data"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/training_data.json"),
        help="Path to JSON/CSV file containing normal training data",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Manually override decision_function threshold (optional)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, manual_threshold=args.threshold)
