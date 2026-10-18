import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.schemas import EXAMPLE_SAMPLE


@pytest.fixture(scope="module")
def client():
    # Using the client as a context manager runs the lifespan, which loads the model.
    with TestClient(app) as c:
        yield c


def test_health_reports_model_ready(client):
    body = client.get("/").json()
    assert body == {"status": "ok", "model_ready": True}


def test_metadata_lists_features_and_scores(client):
    body = client.get("/model/metadata").json()
    assert len(body["features"]) == 13
    assert body["classes"] == ["cultivar_A", "cultivar_B", "cultivar_C"]
    assert 0 <= body["metrics"]["accuracy"] <= 1


def test_single_prediction(client):
    response = client.post("/predict", json=EXAMPLE_SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["class_id"] in (0, 1, 2)
    assert body["confidence"] == max(body["probabilities"].values())
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)


def test_batch_prediction(client):
    response = client.post("/predict/batch", json={"samples": [EXAMPLE_SAMPLE] * 3})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len({p["class_id"] for p in body["predictions"]}) == 1


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"samples": []}).status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [("alcohol", 40.0), ("proline", -1.0), ("hue", "purple")],
)
def test_out_of_range_values_rejected(client, field, value):
    assert client.post("/predict", json={**EXAMPLE_SAMPLE, field: value}).status_code == 422


def test_missing_field_rejected(client):
    partial = {k: v for k, v in EXAMPLE_SAMPLE.items() if k != "flavanoids"}
    assert client.post("/predict", json=partial).status_code == 422


def test_503_when_model_missing(client):
    saved = app.state.model
    app.state.model = None
    try:
        assert client.post("/predict", json=EXAMPLE_SAMPLE).status_code == 503
    finally:
        app.state.model = saved
