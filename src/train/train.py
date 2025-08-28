import os, io, joblib
import pandas as pd
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from datetime import datetime
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from common.db import get_engine

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

BUCKET       = os.environ["S3_BUCKET"]
CURATED_KEY  = os.environ["S3_CURATED_KEY"]          # e.g. curated/bank_clean.csv
MODEL_KEY    = os.environ["S3_MODEL_KEY"]            # e.g. models/bank_model.joblib
MODEL_PATH   = os.getenv("MODEL_PATH", "artifacts/model.joblib")
REGION       = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")

def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    # curated csv written by preprocess; default comma separator
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def upload_file_to_s3(local_path: str, bucket: str, key: str):
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
    print(f"✅ Uploaded model to s3://{bucket}/{key}")

def s3_copy_object(bucket: str, src_key: str, dst_key: str, region: str = REGION):
    s3 = boto3.client("s3", region_name=region)
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        ServerSideEncryption="AES256",
    )

def main():
    print(f"📥 Reading curated data from s3://{BUCKET}/{CURATED_KEY}")
    df = read_csv_from_s3(BUCKET, CURATED_KEY)
    s3 = boto3.client("s3", region_name=REGION)
    cur_head = s3.head_object(Bucket=BUCKET, Key=CURATED_KEY)
    dataset_etag = cur_head.get("ETag", "").strip('"')
    dataset_key  = CURATED_KEY
    row_count    = len(df)

    if "y" not in df.columns:
        raise SystemExit("❌ Column 'y' not found in curated dataset.")
    df["target"] = (df["y"].str.lower().eq("yes")).astype(int)

    drop_cols = ["y", "duration", "target"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["target"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                            remainder="passthrough")
    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline([("prep", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    f1 = f1_score(yte, pred)
    print(f"AUC: {auc:.4f} | F1: {f1:.4f}")

    # save artifact locally
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    artifact = {
        "model": pipe,
        "features": X.columns.tolist(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "dataset_meta": {
            "s3_key": dataset_key,
            "etag": dataset_etag,
            "row_count": int(row_count),
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"💾 Saved {MODEL_PATH}")

    # --- S3 versioning: unique + latest ---
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    unique_key = f"models/bank_model_{stamp}.joblib"

    # upload ONCE to unique key
    upload_file_to_s3(MODEL_PATH, BUCKET, unique_key)

    # copy to fixed 'latest' key so API keeps using the same path
    s3_copy_object(BUCKET, unique_key, MODEL_KEY)

    # --- Log version to Postgres ---
    ENGINE = get_engine()
    s3 = boto3.client("s3", region_name=REGION)
    head = s3.head_object(Bucket=BUCKET, Key=unique_key)
    etag = head.get("ETag", "").strip('"')

    stmt = text("""
        INSERT INTO model_versions (s3_key, etag, metrics, dataset_etag, dataset_key)
        VALUES (:k, :e, :m, :d_etag, :d_key)
    """).bindparams(bindparam("m", type_=JSONB))

    with ENGINE.begin() as conn:
        conn.execute(stmt, {
            "k": unique_key,
            "e": etag,                  
            "m": {"auc": float(auc), "f1": float(f1)},
            "d_etag": dataset_etag,     
            "d_key": dataset_key,       
        })

if __name__ == "__main__":
    main()