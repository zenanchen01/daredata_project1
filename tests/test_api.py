from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_predict_sanity(monkeypatch):
    # ensure model is loaded for test session; call startup event
    for fn in app.router.on_startup:
        fn()
    payload = {
        "age": 39, "job": "technician", "marital": "single", "education": "tertiary",
        "default": "no", "balance": 1500, "housing": "yes", "loan": "no",
        "contact": "cellular", "day": 15, "month": "may", "campaign": 2,
        "pdays": 999, "previous": 0, "poutcome": "unknown"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()