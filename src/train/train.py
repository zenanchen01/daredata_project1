import os, io, joblib
import pandas as pd
import boto3
from botocore.exceptions import ClientError, BotoCoreError

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

def main():
    print(f"📥 Reading curated data from s3://{BUCKET}/{CURATED_KEY}")
    df = read_csv_from_s3(BUCKET, CURATED_KEY)

    # Target -> binary
    if "y" not in df.columns:
        raise SystemExit("❌ Column 'y' not found in curated dataset.")
    df["target"] = (df["y"].str.lower().eq("yes")).astype(int)

    # Drop leakage + target from features
    drop_cols = ["y", "duration", "target"]  # 'duration' leaks info if present
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["target"]

    # Split categorical vs numeric
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    # Preprocess + model
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough"
    )
    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])

    # Train/validate
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    f1 = f1_score(yte, pred)
    print(f"AUC: {auc:.4f} | F1: {f1:.4f}")

    # Save artifact (pipeline + feature order)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    artifact = {
        "model": pipe,
        "features": X.columns.tolist(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"💾 Saved {MODEL_PATH}")

    # Upload to S3
    try:
        upload_file_to_s3(MODEL_PATH, BUCKET, MODEL_KEY)
    except (ClientError, BotoCoreError) as e:
        raise SystemExit(f"❌ Failed to upload model to S3: {e}")

if __name__ == "__main__":
    main()