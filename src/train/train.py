import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

ENGINE = create_engine("postgresql+psycopg2://user:pass@db:5432/ml_db")

# Read from Postgres
df = pd.read_sql("SELECT * FROM bank", ENGINE)

# Target -> binary
df["target"] = (df["y"].str.lower().eq("yes")).astype(int)

# Drop leakage + target from features
drop_cols = ["y", "duration", "target"]  # 'duration' leaks info
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
y = df["target"]

# Split categorical vs numeric
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocess
pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough"
)

# Model (simple, robust baseline)
clf = LogisticRegression(max_iter=2000, n_jobs=None)

pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe.fit(Xtr, ytr)

# Metrics
proba = pipe.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)
auc = roc_auc_score(yte, proba)
f1 = f1_score(yte, pred)

print(f"AUC: {auc:.4f} | F1: {f1:.4f}")

# Save artifact (pipeline + feature order)
joblib.dump(
    {
        "model": pipe,
        "features": X.columns.tolist(),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
    },
    "artifacts/model.joblib"
)

print("Saved artifacts/model.joblib")