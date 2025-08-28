# common/db.py
import os, json, boto3
from sqlalchemy import create_engine

def _database_url_from_secret() -> str:
    secret_id = os.environ["RDS_SECRET_ARN"].strip()
    region = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
    sm = boto3.client("secretsmanager", region_name=region)
    data = json.loads(sm.get_secret_value(SecretId=secret_id)["SecretString"])

    user = data["username"]
    pwd  = data["password"]

    # 1) Preferred: full connection info in the secret
    host = data.get("host")
    port = int(data.get("port") or 5432)
    db   = data.get("dbname") or data.get("database")

    # 2) Fallback: env overrides
    host = host or os.getenv("RDS_DB_HOST")
    port = int(port or os.getenv("RDS_DB_PORT") or 5432)
    db   = db   or os.getenv("RDS_DB_NAME") or "postgres"

    if not host:
        raise RuntimeError("No host in secret and RDS_DB_HOST not set.")

    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=require"

# singleton
_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(_database_url_from_secret(), pool_pre_ping=True)
    return _engine