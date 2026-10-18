# Wine Cultivar Classifier — FastAPI Lab

Serving a scikit-learn model as a REST API with [FastAPI](https://fastapi.tiangolo.com/) and [uvicorn](https://www.uvicorn.org/).

The starter lab trains a Decision Tree on Iris and exposes a single `/predict` route. This version moves to the **UCI Wine recognition dataset** (178 samples, 13 chemical measurements, 3 grape cultivars) and treats the API more like a small production service.

## Changes from the starter lab

| Area | Starter lab | This version |
|------|-------------|--------------|
| Dataset | Iris (4 features) | Wine (13 features), stratified 75/25 split |
| Model | `DecisionTreeClassifier(max_depth=3)` | `Pipeline(StandardScaler → RandomForestClassifier)` |
| Evaluation | none | accuracy, macro-F1 and confusion matrix written to `model/metrics.json` |
| Model loading | `joblib.load` on every request, relative to the working directory | loaded once in a FastAPI `lifespan` hook, path resolved from the package |
| Prediction output | class id only | class id, cultivar name, confidence and per-class probabilities |
| Endpoints | `/`, `/predict` | `/`, `/model/metadata`, `/predict`, `/predict/batch` |
| Validation | types only | per-feature range bounds; batches limited to 1–100 samples |
| Errors | 500 on failure | 422 bad input, 503 model not trained, 500 prediction failure |
| Tests | none | pytest suite (`tests/test_main.py`) |

## Project structure

```
FastAPI_Labs/
├── model/
│   ├── wine_model.pkl      # fitted pipeline (created by train.py)
│   └── metrics.json        # held-out scores + metadata (created by train.py)
├── src/
│   ├── data.py             # dataset loading, split, feature/class names
│   ├── train.py            # build, fit, evaluate and save the pipeline
│   ├── predict.py          # model/metrics loading and inference
│   ├── schemas.py          # Pydantic request/response models
│   └── main.py             # FastAPI app and routes
├── tests/
│   └── test_main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All commands below run from the `FastAPI_Labs/` folder (not from `src/`).

## Train

```bash
python -m src.train
```

```
accuracy=0.9778  f1_macro=0.9771
model   -> .../FastAPI_Labs/model/wine_model.pkl
metrics -> .../FastAPI_Labs/model/metrics.json
```

## Serve

```bash
uvicorn src.main:app --reload
```

Interactive docs: <http://127.0.0.1:8000/docs>

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Liveness check, plus `model_ready` flag |
| GET | `/model/metadata` | Feature order, class names and test-set metrics |
| POST | `/predict` | Classify one wine sample |
| POST | `/predict/batch` | Classify up to 100 samples at once |

### Single prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
       "magnesium": 100, "total_phenols": 2.65, "flavanoids": 2.76,
       "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
       "hue": 1.05, "od280_od315_ratio": 3.4, "proline": 1050}'
```

```json
{
  "class_id": 0,
  "cultivar": "cultivar_A",
  "confidence": 1.0,
  "probabilities": {"cultivar_A": 1.0, "cultivar_B": 0.0, "cultivar_C": 0.0}
}
```

### Batch prediction

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [ {...}, {...} ]}'
```

Returns `{"count": N, "predictions": [ ... ]}` with one entry per sample, in order.

### Error responses

- **422**: missing field, wrong type, or value outside the allowed range (e.g. `alcohol: 40`)
- **503**: no trained model on disk; run `python -m src.train`
- **500**: unexpected failure inside the model

## Tests

```bash
pytest -v
```

The suite covers health, metadata, single and batch predictions, range/type/missing-field validation, empty batches, and the 503 path when no model is loaded.

## Key FastAPI concepts used

- **Pydantic models** (`src/schemas.py`) define the request body; FastAPI parses the JSON, converts types and returns a 422 with details when validation fails.
- **`response_model`** controls how the output is serialized and documents the schema in `/docs`.
- **`lifespan`** runs startup code once, so the model is loaded into `app.state` before the first request.
- **`HTTPException`** maps failures to explicit status codes instead of crashing the worker.
