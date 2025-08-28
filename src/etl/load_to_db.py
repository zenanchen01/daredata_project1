import sys
import pandas as pd
from sqlalchemy import 
from common.db import get_engine

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/bank.csv"
ENGINE = get_engine()

# Read CSV (UCI’s bank-full/bank.csv style)
df = pd.read_csv(CSV, sep=";", na_values=["unknown"])

# Normalize columns
df.columns = [c.strip().lower().replace("-", "_").replace(".", "_").replace(" ", "_") for c in df.columns]

# Load into Postgres
df.to_sql("bank", ENGINE, if_exists="replace", index=False, chunksize=5000, method="multi")
print(f"Loaded {len(df)} rows into 'bank'.")