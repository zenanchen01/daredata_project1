import sys
import pandas as pd
from sqlalchemy import create_engine

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/bank.csv"
ENGINE = create_engine("postgresql+psycopg2://user:pass@db:5432/ml_db")

# Read CSV (UCI’s bank-full/bank.csv style)
df = pd.read_csv(CSV, sep=";", na_values=["unknown"])

# Normalize columns
df.columns = [c.strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_") for c in df.columns]

# Load into Postgres
df.to_sql("bank", ENGINE, if_exists="replace", index=False, chunksize=5000, method="multi")
print(f"Loaded {len(df)} rows into 'bank'.")