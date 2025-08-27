from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib, os
import pandas as pd

app = FastAPI(title="Bank Marketing API")

ART = None
FEATURES = None
MODEL = None

# Expected input structure (subset is fine; missing keys become None)
class BankPayload(BaseModel):
    age: int | None = Field(default=None)
    job: str | None = None
    marital: str | None = None
    education: str | None = None
    default: str | None = None
    balance: int | None = None
    housing: str | None = None
    loan: str | None = None
    contact: str | None = None
    day: int | None = None
    month: str | None = None
    campaign: int | None = None
    pdays: int | None = None
    previous: int | None = None
    poutcome: str | None = None

@app.on_event("startup")
def _load_model():
    global ART, MODEL, FEATURES
    path = os.getenv("MODEL_PATH", "artifacts/model.joblib")
    ART = joblib.load(path)
    MODEL = ART["model"]
    FEATURES = ART["features"]

@app.get("/health")
def health():
    ok = MODEL is not None and FEATURES is not None
    return {"status": "ok" if ok else "not_ready"}

# adjust if you have other numeric columns
NUMERIC_COLS = {"age", "balance", "day", "campaign", "pdays", "previous"}

def _coerce_row_to_features(row_in: dict) -> pd.DataFrame:
    """Return a 1-row DataFrame with columns=FEATURES and proper dtypes."""
    if FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    row = {}
    for f in FEATURES:
        v = row_in.get(f, None)
        if f in NUMERIC_COLS:
            if v is None or v == "":
                v = 0
            try:
                v = int(v)
            except Exception:
                raise HTTPException(status_code=422, detail=f"Field '{f}' must be numeric, got {row_in.get(f)!r}")
        else:
            v = "unknown" if v is None else str(v)
        row[f] = v
    return pd.DataFrame([row], columns=FEATURES)

@app.post("/predict")
def predict(p: BankPayload):
    if MODEL is None or FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train first and restart API.")
    # Build a dict from the Pydantic model fields (flat payload)
    incoming = {f: getattr(p, f, None) for f in FEATURES}
    X = _coerce_row_to_features(incoming)
    try:
        proba = MODEL.predict_proba(X)[0][1]
        pred = int(proba >= 0.5)
        return {"prediction": pred, "proba": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {type(e).__name__}: {e}")