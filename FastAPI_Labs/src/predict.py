import json
from pathlib import Path

import joblib
import numpy as np

from .data import CULTIVARS, FEATURE_NAMES

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def load_model():
    """
    Load the trained pipeline from disk.
    Returns:
        The fitted pipeline, or None if train.py has not been run yet.
    """
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_metrics():
    """Return the metrics report written by train.py, or an empty dict."""
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text())


def to_matrix(samples):
    """Turn a list of feature dicts into a 2D array in training column order."""
    return np.array([[sample[name] for name in FEATURE_NAMES] for sample in samples])


def predict_data(model, samples):
    """
    Predict the cultivar for one or more wine samples.
    Args:
        model: The fitted pipeline.
        samples (list[dict]): Feature dicts keyed by FEATURE_NAMES.
    Returns:
        list[dict]: class_id, cultivar, confidence and per-class probabilities.
    """
    probabilities = model.predict_proba(to_matrix(samples))
    results = []
    for row in probabilities:
        class_id = int(np.argmax(row))
        results.append(
            {
                "class_id": class_id,
                "cultivar": CULTIVARS[class_id],
                "confidence": round(float(row[class_id]), 4),
                "probabilities": {name: round(float(p), 4) for name, p in zip(CULTIVARS, row)},
            }
        )
    return results
