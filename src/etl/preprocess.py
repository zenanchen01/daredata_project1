import os
import io
import pandas as pd
import boto3
from sqlalchemy import create_engine, text
from common.db import get_engine

# Env vars set in .env
BUCKET       = os.environ["S3_BUCKET"]
RAW_KEY      = os.environ["S3_DATA_KEY"]       # e.g. datasets/bank.csv
CURATED_KEY  = os.environ["S3_CURATED_KEY"]    # e.g. curated/bank_clean.csv
REGION       = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
ENGINE = get_engine()

def read_csv_s3(bucket: str, key: str) -> pd.DataFrame:
    """Read CSV file directly from S3 into Pandas."""
    s3 = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), sep=";", na_values=["unknown"])

def write_csv_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    """Write Pandas DataFrame to S3 as CSV."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    boto3.client("s3", region_name=REGION).put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ServerSideEncryption="AES256",
    )
    print(f"✅ Wrote curated CSV to s3://{bucket}/{key} ({len(df):,} rows)")

def log_dataset_version(bucket: str, key: str, row_count: int):
    # fetch ETag of the just-written object
    s3 = boto3.client("s3", region_name=REGION)
    head = s3.head_object(Bucket=bucket, Key=key)
    etag = head.get("ETag", "").strip('"')

    ENGINE = get_engine()
    with ENGINE.begin() as conn:
        conn.execute(
            text("INSERT INTO dataset_versions (s3_key, etag, row_count) VALUES (:k,:e,:r)"),
            {"k": key, "e": etag, "r": int(row_count)}
        )
    print(f"🗂️ Logged dataset version: key={key}, etag={etag}, rows={row_count}")
    # return etag if you want to pass it to training later
    return etag

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset into curated form."""
    # 1) normalize column names
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]

    # 2) clean string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().str.lower().replace("unknown", pd.NA)

    # 3) numeric casting
    int_like = ["age", "balance", "day", "campaign", "pdays", "previous", "duration"]
    for c in [col for col in int_like if col in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) remove leakage column
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # 5) target cleanup
    if "y" in df.columns:
        df["y"] = df["y"].str.strip().str.lower()
        df = df[df["y"].isin(["yes", "no"])]

    # 6) drop rows missing critical fields
    must_have = [
        "age", "job", "marital", "education", "balance", "housing", "loan",
        "contact", "day", "month", "campaign", "pdays", "previous", "poutcome", "y"
    ]
    df = df.dropna(subset=[c for c in must_have if c in df.columns])

    # 7) simple outlier handling
    if "balance" in df.columns:
        q99 = df["balance"].quantile(0.99)
        df.loc[df["balance"] > q99, "balance"] = q99

    return df.reset_index(drop=True)

def main():
    print(f"📥 Reading raw from s3://{BUCKET}/{RAW_KEY}")
    raw = read_csv_s3(BUCKET, RAW_KEY)
    curated = preprocess(raw)
    write_csv_s3(curated, BUCKET, CURATED_KEY)
    log_dataset_version(BUCKET, CURATED_KEY, len(curated))

if __name__ == "__main__":
    main()