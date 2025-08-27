import os
from pathlib import Path
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def main():
    bucket = os.environ["S3_BUCKET"]              # e.g. "my-bucket"
    key    = os.environ.get("S3_DATA_KEY", "datasets/bank.csv")
    local  = Path(os.environ.get("LOCAL_DATA_PATH", "data/bank.csv"))

    if not local.exists():
        raise FileNotFoundError(f"Local file not found: {local}")

    s3 = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION"))
    extra = {"ServerSideEncryption": "AES256"}    # SSE-S3 (good default)

    try:
        s3.upload_file(str(local), bucket, key, ExtraArgs=extra)
        print(f"✅ Uploaded {local} → s3://{bucket}/{key}")
    except (BotoCoreError, ClientError) as e:
        raise SystemExit(f"❌ S3 upload failed: {e}")

if __name__ == "__main__":
    main()