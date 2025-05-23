import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def aggregate_numeric_features(df_card_activity):
    """
    Aggregate numeric columns from the card activity DataFrame.
    Excludes PRIOR_LOAN_ID and groups by CUSTOMER_ID.
    Uses mean, sum, max, and min.
    """
    numeric_df = df_card_activity.select_dtypes(include=["number"]).drop(columns=["PRIOR_LOAN_ID"], errors="ignore")
    agg_funcs = ["mean", "sum", "max", "min"]
    agg_numeric = numeric_df.groupby("CUSTOMER_ID").agg(agg_funcs)

    agg_numeric.columns = [f"card_activity_agg_{col}_{stat}" for col, stat in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    return agg_numeric

def aggregate_categorical_features(df_card_activity):
    """
    Aggregate categorical columns (excluding PRIOR_LOAN_ID) by CUSTOMER_ID.
    Computes the most frequent value (mode) per column.
    Falls back to 'Unknown' if mode is ambiguous or missing.
    """
    categorical_df = df_card_activity.select_dtypes(include=["object", "category"]).drop(columns=["PRIOR_LOAN_ID"], errors="ignore")
    
    if categorical_df.empty:
        return None

    categorical_df = df_card_activity[["CUSTOMER_ID"]].join(categorical_df)

    def safe_mode(series):
        return series.value_counts().idxmax() if not series.empty else "Unknown"

    agg_categorical = categorical_df.groupby("CUSTOMER_ID").agg(safe_mode)
    agg_categorical.columns = [f"card_activity_agg_{col}_most_frequent" for col in agg_categorical.columns]
    agg_categorical.reset_index(inplace=True)
    return agg_categorical

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge numeric and categorical aggregates on CUSTOMER_ID.
    """
    if agg_categorical is not None:
        merged = agg_numeric.merge(agg_categorical, on="CUSTOMER_ID", how="left")
    else:
        merged = agg_numeric
    return merged

def safe_merge(df_main, df_new, merge_on="CUSTOMER_ID", name=""):
    """
    Merge and log shape and null counts for debugging purposes.
    """
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how="left")

    logging.info(f"✅ Merged {name}: {prev_shape} → {df_main.shape}")
    
    missing = df_main.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logging.warning(f"🛠️ Missing values in {name} after merge:\n{missing}")
    return df_main

def aggregate_card_activity_features(df_card_activity, additional_feature_dfs):
    """
    Aggregate all card activity features by CUSTOMER_ID.

    Steps:
      1. Aggregate numeric features (drop PRIOR_LOAN_ID).
      2. Aggregate categorical features (drop PRIOR_LOAN_ID).
      3. Merge numeric and categorical aggregates.
      4. Merge additional engineered features.
      5. Run sanity checks for missing/empty/infinite values.

    Parameters:
      - df_card_activity: Input synthetic card activity data.
      - additional_feature_dfs: List of (DataFrame, name) tuples for merging.

    Returns:
      Aggregated DataFrame keyed by CUSTOMER_ID.
    """
    logging.info("Starting card activity aggregation...")

    # Step 1–3: Aggregate core features
    agg_numeric = aggregate_numeric_features(df_card_activity)
    agg_categorical = aggregate_categorical_features(df_card_activity)
    df_aggregated = merge_aggregated_features(agg_numeric, agg_categorical)
    logging.info(f"✅ Aggregation complete. Shape: {df_aggregated.shape}")

    # Step 4: Merge additional features
    for df_new, name in additional_feature_dfs:
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="CUSTOMER_ID", name=name)

    # Step 5: Sanity checks
    logging.info("🔍 Checking for standard missing values...")
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    logging.warning(missing_values) if not missing_values.empty else logging.info("✅ No missing values.")

    logging.info("🔍 Checking for hidden NaNs (empty strings, text 'nan')...")
    hidden_nans = (df_aggregated == "").sum() + (df_aggregated == "nan").sum()
    hidden_nans = hidden_nans[hidden_nans > 0]
    logging.warning(hidden_nans) if not hidden_nans.empty else logging.info("✅ No hidden NaNs.")

    logging.info("🔍 Checking for infinite values...")
    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    logging.warning(inf_values) if not inf_values.empty else logging.info("✅ No infinite values detected.")

    return df_aggregated
