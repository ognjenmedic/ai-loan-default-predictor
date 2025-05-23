import pandas as pd
import numpy as np
from functools import reduce
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def aggregate_numeric_features(df_credit_summary):
    """
    Aggregate numeric columns (excluding CREDIT_RECORD_ID) by CUSTOMER_ID.
    Uses mean, sum, max, and min to summarize values per customer.
    """
    numeric_df = df_credit_summary.select_dtypes(include=["number"]).drop(columns=["CREDIT_RECORD_ID"], errors="ignore")
    agg_funcs = ["mean", "sum", "max", "min"]
    agg_numeric = numeric_df.groupby("CUSTOMER_ID").agg(agg_funcs)
    agg_numeric.columns = [f"credit_summary_agg_{col}_{stat}" for col, stat in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    return agg_numeric

def aggregate_categorical_features(df_credit_summary):
    """
    Aggregate categorical columns (excluding CREDIT_RECORD_ID) by CUSTOMER_ID.
    Returns the most frequent value (mode) for each categorical column per customer.
    Falls back to 'Unknown' if mode cannot be determined.
    """
    categorical_df = df_credit_summary.select_dtypes(include=["object", "category"]).drop(columns=["CREDIT_RECORD_ID"], errors="ignore")
    if categorical_df.empty:
        return None
    
    # Include CUSTOMER_ID for grouping
    categorical_df = df_credit_summary[["CUSTOMER_ID"]].join(categorical_df)

    def safe_mode(series):
        """Return most frequent value or 'Unknown' if empty or ambiguous."""
        mode_vals = series.mode()
        return mode_vals[0] if not mode_vals.empty else "Unknown"

    agg_categorical = categorical_df.groupby("CUSTOMER_ID").agg(safe_mode)
    agg_categorical.columns = [f"credit_summary_agg_{col}_most_frequent" for col in agg_categorical.columns]
    agg_categorical.reset_index(inplace=True)
    return agg_categorical

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge numeric and categorical aggregates into a unified table.
    """
    if agg_categorical is not None:
        merged = pd.merge(agg_numeric, agg_categorical, on="CUSTOMER_ID", how="left")
    else:
        merged = agg_numeric
    return merged

def safe_merge(df_main, df_new, merge_on="CUSTOMER_ID", name=""):
    """
    Merge two DataFrames and log shape changes and missing values.
    Useful for debugging and feature pipeline tracking.
    """
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how="left")
    logging.info(f"✅ Merged {name}: {prev_shape} -> {df_main.shape}")
    
    missing = df_main.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logging.warning(f"🛠️ Missing Values in {name} After Merge:\n{missing}")
    logging.info("-" * 50)
    return df_main

def aggregate_credit_summary_features(df_credit_summary, additional_feature_dfs):
    """
    Compute and combine all credit_summary features for modeling.
    Steps:
      1. Aggregate numeric and categorical features from the raw credit_summary table.
      2. Merge in additional engineered feature DataFrames.

    Parameters:
      - df_credit_summary: Raw synthetic credit_summary DataFrame with CUSTOMER_ID key.
      - additional_feature_dfs: List of tuples (df, name) representing additional engineered feature sets.

    Returns:
      Final merged credit_summary feature DataFrame per CUSTOMER_ID.
    """
    agg_numeric = aggregate_numeric_features(df_credit_summary)
    agg_categorical = aggregate_categorical_features(df_credit_summary)
    aggregated = merge_aggregated_features(agg_numeric, agg_categorical)

    for df_new, name in additional_feature_dfs:
        aggregated = safe_merge(aggregated, df_new, merge_on="CUSTOMER_ID", name=name)

    return aggregated
