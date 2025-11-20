import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


DEFAULT_FEATURE_ORDER = ["cpu", "memory", "connections", "bytes_sent", "bytes_recv"]
DEFAULT_ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = DEFAULT_ARTIFACT_DIR / "oneclass_svm.joblib"
SCALER_PATH = DEFAULT_ARTIFACT_DIR / "scaler.joblib"
METADATA_PATH = DEFAULT_ARTIFACT_DIR / "metadata.json"


class ModelBundle:
    def __init__(
        self,
        model: OneClassSVM,
        scaler: StandardScaler,
        threshold: float,
        feature_order: List[str],
    ):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.feature_order = feature_order


_loaded_bundle: ModelBundle | None = None


def load_artifacts(
    model_path: Path = MODEL_PATH,
    scaler_path: Path = SCALER_PATH,
    metadata_path: Path = METADATA_PATH,
) -> ModelBundle:
    """Load model, scaler, and metadata once and reuse."""
    global _loaded_bundle
    if _loaded_bundle:
        return _loaded_bundle

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    feature_order = metadata.get("feature_order", DEFAULT_FEATURE_ORDER)
    base_threshold = metadata.get("computed_threshold", -0.1)
    manual_override = os.getenv("SVM_THRESHOLD")
    threshold = float(manual_override) if manual_override is not None else metadata.get(
        "manual_threshold", base_threshold
    )

    _loaded_bundle = ModelBundle(model, scaler, threshold, feature_order)
    return _loaded_bundle


def preprocess_features(
    raw_features: Dict[str, float],
    scaler: StandardScaler,
    feature_order: List[str] = DEFAULT_FEATURE_ORDER,
) -> np.ndarray:
    vector = np.array([[float(raw_features[key]) for key in feature_order]], dtype=float)
    return scaler.transform(vector)


def predict_anomaly(features: Dict[str, float]) -> Dict[str, float | bool]:
    bundle = load_artifacts()
    scaled = preprocess_features(features, bundle.scaler, bundle.feature_order)
    decision_value = float(bundle.model.decision_function(scaled)[0])
    is_anomaly = decision_value < bundle.threshold
    anomaly_score = bundle.threshold - decision_value
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "decision_value": decision_value,
        "threshold": bundle.threshold,
    }


def save_metadata(
    feature_order: List[str],
    computed_threshold: float,
    manual_threshold: float | None,
    metadata_path: Path = METADATA_PATH,
) -> None:
    metadata = {
        "feature_order": feature_order,
        "computed_threshold": computed_threshold,
        "manual_threshold": manual_threshold
        if manual_threshold is not None
        else computed_threshold,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
