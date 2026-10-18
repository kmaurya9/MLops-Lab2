import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import CULTIVARS, FEATURE_NAMES, load_data, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def build_pipeline():
    """
    Scaling + Random Forest. Scaling is not strictly needed for trees, but keeping
    it in the pipeline lets the estimator be swapped without touching the API.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("forest", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=7)),
        ]
    )


def fit_model(X_train, y_train):
    """
    Train the pipeline on the training split.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        Pipeline: The fitted pipeline.
    """
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline, X_test, y_test):
    """
    Score the fitted pipeline on the held-out split.
    Returns:
        dict: accuracy, macro F1 and the confusion matrix.
    """
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": int(len(y_test)),
    }


def save_artifacts(pipeline, metrics):
    """Persist the model and a metrics report next to it."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    report = {
        "model": "StandardScaler + RandomForestClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURE_NAMES,
        "classes": CULTIVARS,
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    scores = evaluate(model, X_test, y_test)
    save_artifacts(model, scores)
    print(f"accuracy={scores['accuracy']}  f1_macro={scores['f1_macro']}")
    print(f"model   -> {MODEL_PATH}")
    print(f"metrics -> {METRICS_PATH}")
