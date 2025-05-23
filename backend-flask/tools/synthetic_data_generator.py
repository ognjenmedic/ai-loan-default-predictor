"""
Synthetic Data Generator for Loan Default Prediction App

This script generates synthetic feature ranges based on a defined schema
to simulate realistic input data for a credit risk prediction model.

- Feature names follow a structure typical of credit modeling datasets.
- All feature values are 100% synthetic and generated using randomized logic.
- No proprietary or real-world datasets are used or referenced.

This script is intended solely for educational and portfolio use.
"""



import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from tools.feature_config import RAW_SCHEMA, DTYPE_MAP, DEFAULT_CATS, DOMAIN_OVERRIDES


# —————————————————————————————————————————
# 1) BUILD SYNTHETIC RANGES
# —————————————————————————————————————————
feature_ranges = {}
for table, cols in RAW_SCHEMA.items():
    # 1a) Build a synthetic DataFrame of the right shape
    #    n_samples should be > #cols so describe() is meaningful
    n_samples = max(1000, len(cols) * 20)
    # Classification-style matrix for simplicity
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=len(cols),
        n_informative=min(10, len(cols)),
        n_redundant=0,
        n_repeated=0,
        random_state=42
    )

    df_syn = pd.DataFrame(X, columns=cols)

    # 1b) Convert columns to their target dtype
    for col in cols:
        dtype = DTYPE_MAP.get(col, "float64")
        if dtype == "category":
            cats = DEFAULT_CATS.get(col, [f"Cat_{i}" for i in range(3)])
            df_syn[col] = pd.Categorical(
                np.random.choice(cats, size=n_samples),
                categories=cats
            )
        elif dtype in ("int64", "Int64"):
            # round & cast
            df_syn[col] = np.round(df_syn[col] * 100).astype(int)
        else:
            # float64 or other numerics
            df_syn[col] = df_syn[col].astype(float)

    # 1c) Describe numeric columns
    desc = df_syn.describe().T

    # 1d) Assemble table’s ranges
    feature_ranges[table] = {}
    for col in cols:
        col_info = {"feature_name": col, "dtype": DTYPE_MAP.get(col, "float64")}
        if isinstance(df_syn[col].dtype, pd.CategoricalDtype):
            col_info["categories"] = list(df_syn[col].cat.categories)
        else:
            stats = desc.loc[col]
            col_info.update({
                "min":   float(stats["min"]),
                "max":   float(stats["max"]),
                "mean":  float(stats["mean"]),
                "std":   float(stats["std"])
            })
        feature_ranges[table][col] = col_info

# Apply overrides to the generated stats
for table, cols in feature_ranges.items():
    for feature_name, stats in cols.items():
        if feature_name in DOMAIN_OVERRIDES:
            stats.update(DOMAIN_OVERRIDES[feature_name])


# # ------------------------------------------------------------------
# # 2)   Money features overrides - 
# #      illustrative USD ranges (chosen by the author, purely for UX)
# # ------------------------------------------------------------------
# feature_ranges["customer_profile"].update({
#     "MONTHLY_INCOME": {
#         "feature_name": "MONTHLY_INCOME",
#         "dtype": "float64",
#         "min":   1_000,    
#         "max":  10_000,    
#         "mean":  4_000,    
#         "std":   2_000,    
#     },
#     "LOAN_AMOUNT": {
#         "feature_name": "LOAN_AMOUNT",
#         "dtype": "float64",
#         "min":     5_000,    
#         "max":   150_000,
#         "mean":  30_000,
#         "std":   20_000
#     },
#     "LOAN_ANNUITY": {
#         "feature_name": "LOAN_ANNUITY",
#         "dtype": "float64",
#         "min":       200,    
#         "max":     5_000,
#         "mean":   1_200,
#         "std":     700
#     },
# })

# feature_ranges["customer_profile"]["LOAN_TERM_MONTHS"] = {
#     "feature_name": "LOAN_TERM_MONTHS",
#     "dtype": "int64",
#     "min":     6,        
#     "max":     72,
#     "mean":    36,
#     "std":     15
# }

# —————————————————————————————————————————
# 3) WRITE OUT THE JSON to backend-flask/
# —————————————————————————————————————————
out = Path(__file__).resolve().parents[1] / "feature_ranges.json"
with open(out, "w") as f:
    json.dump(feature_ranges, f, indent=2)

print(f"✅ Wrote synthetic ranges to {out.resolve()}")

