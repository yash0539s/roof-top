import json
import time

import pytest
from fastapi.testclient import TestClient

EXAMPLE = {"uid": "IND_B12", "area_m2": 451.66, "perimeter_m": 102.05,
           "mean_r": 85.06, "mean_g": 105.91, "mean_b": 156.42,
           "std_r": 12.03, "std_g": 10.59, "std_b": 9.26}


@pytest.fixture(scope="module")
def client(model_dir, tmp_path_factory):
    log = tmp_path_factory.mktemp("logs") / "predictions.jsonl"
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_DIR", str(model_dir))
    mp.setenv("PREDICTION_LOG", str(log))
    mp.delenv("RASTER_DIR", raising=False)
    from api.main import app
    with TestClient(app) as c:
        c.log_path = log
        yield c
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["model_version"] == "roof-classifier:test"
    assert info["test_macro_f1"] == 0.9 and info["trained_at"]


def test_predict_contract(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["uid"] == "IND_B12"
    assert body["roof_class"] in {"ASBESTOS", "METAL", "RCC", "TILED"}
    assert set(body["probabilities"]) == {"ASBESTOS", "METAL", "RCC", "TILED"}
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert body["confidence"] == max(body["probabilities"].values())
    assert body["model_version"] == "roof-classifier:test"


def test_predict_is_fast(client):
    client.post("/predict", json=EXAMPLE)
    t0 = time.perf_counter()
    for _ in range(20):
        client.post("/predict", json=EXAMPLE)
    assert (time.perf_counter() - t0) / 20 < 0.2


def test_prediction_is_logged(client):
    client.post("/predict", json=EXAMPLE)
    rec = json.loads(client.log_path.read_text().strip().splitlines()[-1])
    assert rec["uid"] == "IND_B12" and rec["model_version"] == "roof-classifier:test"
    assert rec["features"]["elongation"] == pytest.approx(102.05 / 451.66 ** 0.5, rel=1e-4)


@pytest.mark.parametrize("field,value", [("mean_r", 0), ("mean_g", 300), ("area_m2", -5), ("std_b", -1)])
def test_bad_input_names_field(client, field, value):
    r = client.post("/predict", json={**EXAMPLE, field: value})
    assert r.status_code == 422
    assert any(e["field"] == field for e in r.json()["detail"])


def test_missing_and_unknown_fields(client):
    r = client.post("/predict", json={k: v for k, v in EXAMPLE.items() if k != "mean_b"})
    assert r.status_code == 422 and r.json()["detail"][0]["field"] == "mean_b"
    r = client.post("/predict", json={**EXAMPLE, "colour": "blue"})
    assert r.status_code == 422 and r.json()["detail"][0]["field"] == "colour"


def test_impossible_perimeter(client):
    r = client.post("/predict", json={**EXAMPLE, "perimeter_m": 10})
    assert r.status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"buildings": [EXAMPLE, {**EXAMPLE, "uid": "X", "mean_r": 200}]})
    assert r.status_code == 200 and [p["uid"] for p in r.json()] == ["IND_B12", "X"]


def test_geometry_endpoint(client):
    square = {"type": "Polygon", "coordinates": [[[689300, 1193400], [689315, 1193400], [689315, 1193415],
                                                  [689300, 1193415], [689300, 1193400]]]}
    colour = {k: v for k, v in EXAMPLE.items() if k.startswith(("mean", "std"))}
    r = client.post("/predict/geometry", json={"type": "Feature", "geometry": square,
                                               "properties": {"uid": "NEW_1", **colour}})
    assert r.status_code == 200, r.text
    assert r.json()["uid"] == "NEW_1"
    # without a raster, colour is required
    r = client.post("/predict/geometry", json={"type": "Feature", "geometry": square, "properties": {}})
    assert r.status_code == 422 and "mean_r" in r.text
