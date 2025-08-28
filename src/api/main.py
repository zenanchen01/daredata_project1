from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os, io, joblib, pandas as pd
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import JSONB

app = FastAPI(title="Bank Marketing API (S3 in-memory)")

# Globals (in-memory cache)
ART = MODEL = None
FEATURES = None

NUMERIC_COLS = {"age", "balance", "day", "campaign", "pdays", "previous"}

ENGINE = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)

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

def _load_artifact_from_s3_into_memory():
    """Fetch joblib artifact from S3 and load into memory (no local file)."""
    bucket = os.environ["S3_BUCKET"]
    key    = os.environ["S3_MODEL_KEY"]
    region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
    s3 = boto3.client("s3", region_name=region)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        art = joblib.load(buf)
        return art
    except (ClientError, BotoCoreError) as e:
        raise RuntimeError(f"S3 get_object failed for s3://{bucket}/{key}: {e}")
    except Exception as e:
        raise RuntimeError(f"joblib.load failed: {e}")

def _coerce_row_to_features(row_in: dict) -> pd.DataFrame:
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
            v = "unknown" if v is None else str(v).strip().lower()
        row[f] = v
    return pd.DataFrame([row], columns=FEATURES)


def log_prediction(req: dict, pred: int, proba: float):
    try:
        ds = ART.get("dataset_meta", {}) if ART else {}
        stmt = text("""
            INSERT INTO prediction_logs (request_json, prediction, proba, model_s3_key, model_etag, dataset_etag, dataset_key)
            VALUES (:req, :pred, :proba, :mkey, :metag, :detag, :dkey)
        """).bindparams(bindparam("req", type_=JSONB))
        with ENGINE.begin() as conn:
            conn.execute(stmt, {
            "req": req, "pred": pred, "proba": proba,
            "mkey": os.getenv("S3_MODEL_KEY"), "metag": os.getenv("MODEL_ETAG",""),
            "detag": ds.get("etag"), "dkey": ds.get("s3_key")
            })
    except Exception as e:
        print(f"[warn] prediction log failed: {e}")

@app.on_event("startup")
def _startup():
    global ART, MODEL, FEATURES
    try:
        ART = _load_artifact_from_s3_into_memory()
        MODEL = ART["model"]
        FEATURES = ART["features"]
        print(f"[startup] Model loaded from S3 into memory (features={len(FEATURES)})")
    except Exception as e:
        ART = MODEL = FEATURES = None
        print(f"[startup] not_ready: {e}")

@app.get("/health")
def health():
    ds = ART.get("dataset_meta", {}) if ART else {}
    return {
        "status": "ok" if MODEL else "not_ready",
        "feature_count": len(ART.get("features", [])) if ART else 0,
        "dataset": {
            "s3_key": ds.get("s3_key"),
            "etag": ds.get("etag"),
            "row_count": ds.get("row_count"),
        },
    }

@app.post("/admin/reload", include_in_schema=False)
def admin_reload(secret: str | None = None):
    if os.getenv("RELOAD_TOKEN") and secret != os.getenv("RELOAD_TOKEN"):
        raise HTTPException(status_code=403, detail="Forbidden")
    return _reload_from_s3()

def _reload_from_s3():
    global ART, MODEL, FEATURES
    ART = _load_artifact_from_s3_into_memory()
    MODEL = ART["model"]
    FEATURES = ART["features"]
    return {"status": "ok", "features": len(FEATURES)}

@app.post("/predict")
def predict(p: BankPayload):
    if MODEL is None or FEATURES is None:
        raise HTTPException(status_code=503, detail="Model not loaded from S3.")
    incoming = {f: getattr(p, f, None) for f in FEATURES}
    X = _coerce_row_to_features(incoming)
    proba = float(MODEL.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)
    log_prediction(incoming, pred, proba)
    return {"prediction": pred, "proba": proba}