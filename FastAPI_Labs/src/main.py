from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from .predict import load_metrics, load_model, predict_data
from .schemas import BatchPrediction, BatchRequest, ModelMetadata, WinePrediction, WineSample


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    app.state.model = load_model()
    app.state.metrics = load_metrics()
    yield


app = FastAPI(
    title="Wine Cultivar Classifier",
    description="Predicts which of three cultivars a wine came from, given its chemical analysis.",
    version="0.2.0",
    lifespan=lifespan,
)


def get_model(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained model found. Run `python -m src.train` first.",
        )
    return model


@app.get("/", status_code=status.HTTP_200_OK, tags=["status"])
async def health_ping(request: Request):
    """Liveness check that also reports whether a model is ready to serve."""
    return {"status": "ok", "model_ready": request.app.state.model is not None}


@app.get("/model/metadata", response_model=ModelMetadata, tags=["model"])
async def model_metadata(request: Request):
    """Features, classes and the held-out scores recorded at training time."""
    get_model(request)
    return request.app.state.metrics


@app.post("/predict", response_model=WinePrediction, tags=["model"])
async def predict_wine(sample: WineSample, request: Request):
    """Classify a single wine sample."""
    model = get_model(request)
    try:
        return predict_data(model, [sample.model_dump()])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch", response_model=BatchPrediction, tags=["model"])
async def predict_batch(batch: BatchRequest, request: Request):
    """Classify up to 100 wine samples in one call."""
    model = get_model(request)
    try:
        results = predict_data(model, [s.model_dump() for s in batch.samples])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    return {"count": len(results), "predictions": results}
