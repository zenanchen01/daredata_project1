# src/etl/sample_to_pg.py
import os, io, pandas as pd, boto3
from sqlalchemy import create_engine, text

BUCKET=os.environ["S3_BUCKET"]
KEY=os.environ["S3_CURATED_KEY"]
REGION=os.getenv("AWS_DEFAULT_REGION","eu-west-1")
ENGINE = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)

s3=boto3.client("s3", region_name=REGION)
obj=s3.get_object(Bucket=BUCKET, Key=KEY)
df=pd.read_csv(io.BytesIO(obj["Body"].read()))

sample=df.sample(n=min(10_000, len(df)), random_state=42)

with ENGINE.begin() as conn:
    conn.execute(text("TRUNCATE TABLE bank_sample"))
    # fast batched insert
    sample.to_sql(
        "bank_sample",
        conn,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5_000,
    )

print(f"Inserted {len(sample)} rows into bank_sample")