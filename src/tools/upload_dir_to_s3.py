# src/tools/upload_dir_to_s3.py
import os
from pathlib import Path
import boto3
from boto3.s3.transfer import TransferConfig

ALLOWED_EXT = {".csv", ".parquet"}

def upload_dir(local_dir: str, bucket: str, prefix: str = "datasets"):
    s3 = boto3.client("s3")
    cfg = TransferConfig(multipart_threshold=8*1024*1024, multipart_chunksize=8*1024*1024)
    base = Path(local_dir)
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            rel = p.relative_to(base).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}"
            s3.upload_file(str(p), bucket, key, Config=cfg, ExtraArgs={"ServerSideEncryption": "AES256"})
            print("Uploaded", f"s3://{bucket}/{key}")

if __name__ == "__main__":
    upload_dir(
        os.getenv("LOCAL_DIR", "/app/data"),        # <— your mounted data dir
        os.environ["S3_BUCKET"],
        os.getenv("S3_PREFIX", "datasets"),
    )