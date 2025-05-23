"""
Model Trainer for Synthetic Loan Default Dataset

This script builds a synthetic dataset using simulated features and programmatic rules, 
trains a LightGBM model, and exports both the model and its feature metadata.

- All data is fully synthetic and generated from randomized logic.
- No real-world, proprietary, or competition datasets are used.
- The target variable (default risk) is artificially constructed using heuristic weights.

This script is part of an educational and demonstration project.
"""


import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pickle
from pandas.api.types import CategoricalDtype
import warnings
warnings.filterwarnings("ignore", "No further splits with positive gain")


# ----------------------
# 1. CONFIG IMPORTS
# ----------------------
from tools.feature_config import RAW_SCHEMA, DTYPE_MAP, DEFAULT_CATS
from pipeline.orchestrator import orchestrate_features

# ----------------------
# 2. Build Dummy Payload
# ----------------------
def build_dummy_payload():
    """
    Build a single dummy JSON payload with synthetic values
    """
    with open(Path(__file__).resolve().parents[1] / "feature_ranges.json", "r") as f:
        feature_ranges = json.load(f)

    def sample_col(stats):
        col_dtype = stats["dtype"]
        if col_dtype == "category":
            return np.random.choice(stats.get("categories", ["Unknown"]))
        else:
            mean, std, min_val, max_val = stats["mean"], stats["std"], stats["min"], stats["max"]
            val = np.random.normal(loc=mean, scale=std)
            return float(np.clip(val, min_val, max_val))

    def sample_table(table_name):
        row = {}
        for col_name, stats in feature_ranges.get(table_name, {}).items():
            if col_name.upper() == "TARGET":
                continue
            row[col_name] = sample_col(stats)
        return row

    payload = {
        "CUSTOMER_ID": 123456,
        "customer_profile": sample_table("customer_profile"),
        "credit_summary": [sample_table("credit_summary") for _ in range(2)],
        "credit_timeline": [sample_table("credit_timeline") for _ in range(4)],
        "card_activity": [sample_table("card_activity")],
        "cash_pos_records": [sample_table("cash_pos_records")],
        "installment_records": [sample_table("installment_records")],
        "prior_loan_history": [sample_table("prior_loan_history")],
    }

    # Ensure all rows have correct CUSTOMER_ID
    for section in ["credit_summary", "card_activity", "cash_pos_records", "installment_records", "prior_loan_history"]:
        for row in payload[section]:
            row["CUSTOMER_ID"] = payload["CUSTOMER_ID"]

    # Assign CREDIT_RECORD_IDs to credit_summary records
    credit_record_ids = [900000 + i for i in range(len(payload["credit_summary"]))]
    for i, summary_row in enumerate(payload["credit_summary"]):
        summary_row["CREDIT_RECORD_ID"] = credit_record_ids[i]

    # Build credit_timeline rows mapped to CREDIT_RECORD_IDs
    credit_timeline_rows = []
    for summary_row in payload["credit_summary"]:
        for _ in range(2):  # Generate 2 timeline records per credit record
            timeline_row = sample_table("credit_timeline")
            timeline_row["CREDIT_RECORD_ID"] = summary_row["CREDIT_RECORD_ID"]
            credit_timeline_rows.append(timeline_row)
    payload["credit_timeline"] = credit_timeline_rows

    return payload


# ----------------------
# 3. Train and Save Model
# ----------------------
N_ROWS = 800                                      
all_payloads  = [build_dummy_payload() for _ in range(N_ROWS)]
all_features  = [orchestrate_features(p) for p in all_payloads]
df            = pd.concat(all_features, ignore_index=True)

# safe_fill: numeric → 0, categorical → "Missing"
def safe_fill(df):
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            if "Missing" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["Missing"])
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(0)
    return df

df = safe_fill(df)

# Cast any leftover object columns to category
for col in df.select_dtypes("object"):
    df[col] = df[col].astype("category")

# Drop columns that are entirely NaN
nan_cols = [c for c in df.columns if df[c].isna().all()]
if nan_cols:
    df.drop(columns=nan_cols, inplace=True)

# -------------------------------------------------
# 3-A. Build a weak, plausible TARGET
# -------------------------------------------------

# 1) Basic signal generators
credit_score      = (df["LOAN_AMOUNT"]   > df["LOAN_AMOUNT"].median()).astype(float)
employment_score  = (df["EMPLOYMENT_DURATION_DAYS"] < df["EMPLOYMENT_DURATION_DAYS"].median()).astype(float)
good_ext = df["EXTERNAL_CREDIT_SCORE"].fillna(0)           # 0 = bad, 1 = good
bad_ext  = 1 - good_ext                           # 0 = good, 1 = bad
loan_term_score   = (df["LOAN_TERM_MONTHS"] > df["LOAN_TERM_MONTHS"].median()).astype(float)

# 2) Occupation-risk lookup
occ_risk_map = {
    "Job A":  2.0,   # higher-risk
    "Job B":  1.0,   # medium
    "Job C":  0.0,   # baseline
    "Job D": -1.0,   # lower risk
    "Job E": -2.0,   # lowest risk
}
occ_score = (
    df["OCCUPATION_TYPE"]
      .map(occ_risk_map)          # map strings → numeric risk
      .astype(float)              # <- ensure plain float dtype
      .fillna(0)
)

# 3) Combined into a probability
prob = (
      0.05                       # base default rate
    + 0.20 * credit_score        # bigger loans  ↑
    + 0.15 * employment_score    # shorter employment ↑
    + 0.50 * bad_ext             # bad external score ↑ 
    + 0.35 * occ_score           # occupation modifier
    + 0.10 * loan_term_score     # longer term ↑
).clip(0.01, 0.95)               # stay inside (0,1)

# 4) Draw the synthetic binary target
df["TARGET"] = np.random.binomial(1, prob)



# -------------------------------------------------
# 3-B. Train / Validate Model
# -------------------------------------------------
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

model = LGBMClassifier(
    n_estimators      = 300,
    learning_rate     = 0.05,
    random_state      = 42,
    class_weight      = "balanced",
    min_data_in_leaf  = 10,
    min_gain_to_split = 0.0,
    n_jobs            = 4,
    categorical_feature = "auto"         
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        early_stopping(stopping_rounds=20),
        log_evaluation(period=10)
    ]
)
# Evaluate
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]
print("\nClassification Report:")
print(classification_report(y_val, y_pred))
print("AUC:", round(roc_auc_score(y_val, y_prob), 4))

# ----------------------
# 4. Save model & metadata
# ----------------------
BASE_DIR = Path(__file__).resolve().parents[1]

out_model = BASE_DIR / "models" / "lightgbm_loan_default.pkl"
out_model.parent.mkdir(parents=True, exist_ok=True)
with open(out_model, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Saved model to {out_model.resolve()}")

out_features = BASE_DIR / "df_final_features.json"
out_data = {
    "features": list(X.columns),
    "dtypes": {c: str(X[c].dtype) for c in X.columns},
    "category_mappings": {
        c: list(X[c].cat.categories) for c in X.columns
        if str(X[c].dtype) == "category"
    }
}
with open(out_features, "w") as f:
    json.dump(out_data, f, indent=2)
print(f"✅ Saved feature metadata to {out_features.resolve()}")

