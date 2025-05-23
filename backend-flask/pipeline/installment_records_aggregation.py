import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def aggregate_numeric_features(df_installment_records):
    """
    Aggregate all numeric features in installment records by CUSTOMER_ID.
    Drops PRIOR_LOAN_ID and computes mean, sum, max, and min.
    Any synthetic features like MISSED_PAYMENT will be included automatically.
    """
    logging.debug(f"Shape before aggregation: {df_installment_records.shape}")

    numeric_df = df_installment_records.select_dtypes(include=["number"]).drop(columns=["PRIOR_LOAN_ID"], errors="ignore")
    agg_funcs = ["mean", "sum", "max", "min"]
    agg_numeric = numeric_df.groupby("CUSTOMER_ID").agg(agg_funcs)

    agg_numeric.columns = [f"installments_agg_{col}_{stat}" for col, stat in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)

    logging.debug(f"Aggregated columns: {agg_numeric.columns.tolist()}")
    logging.debug(f"Unique CUSTOMER_ID after aggregation: {agg_numeric['CUSTOMER_ID'].nunique()}")

    return agg_numeric

def safe_merge(df_main, df_new, merge_on="CUSTOMER_ID", name=""):
    """
    Merge two DataFrames with logging for diagnostics.
    """
    if df_new is None:
        logging.error(f"❌ Cannot merge '{name}' — DataFrame is None.")
        return df_main

    logging.info(f"Merging '{name}' — shape: {df_new.shape}")
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how="left")
    logging.info(f"✅ Merge complete: {prev_shape} → {df_main.shape}")

    missing_cols = set(df_new.columns) - set(df_main.columns)
    if missing_cols:
        logging.warning(f"⚠️ Columns missing after merge '{name}': {missing_cols}")

    return df_main

def aggregate_installment_records_features(df_installment_records, additional_feature_dfs=None):
    """
    Perform final aggregation of installment records features.

    Steps:
    1. Aggregate all numeric features by CUSTOMER_ID.
    2. Merge with any additional engineered features.
    3. Clean known NaNs in std and ratio columns.
    4. Log diagnostics on missing/infinite values.

    Parameters:
      - df_installment_records: Raw or feature-engineered installment records.
      - additional_feature_dfs: Optional list of (df, name) tuples for merging.

    Returns:
      Fully aggregated DataFrame by CUSTOMER_ID.
    """
    logging.info("Starting installment records aggregation...")

    if additional_feature_dfs is None:
        additional_feature_dfs = []

    agg_numeric = aggregate_numeric_features(df_installment_records)
    logging.info(f"✅ Core numeric aggregation complete. Shape: {agg_numeric.shape}")

    df_aggregated = agg_numeric
    for df_new, name in additional_feature_dfs:
        logging.info(f"Merging in: {name} — shape: {df_new.shape}")
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="CUSTOMER_ID", name=name)

    for col in ["installments_STD_DAYS_ENTRY_PAYMENT", "installments_STD_PAYMENT_DELAY"]:
        if col in df_aggregated.columns:
            df_aggregated[col] = df_aggregated[col].fillna(0)

    for col in df_aggregated.columns:
        if "PAYMENT_RATIO" in col:
            df_aggregated[col] = df_aggregated[col].fillna(1)

    # Final diagnostics
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    logging.info("🔍 Missing values check:")
    logging.warning(missing_values) if not missing_values.empty else logging.info("✅ No NaNs found.")

    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    logging.info("🔍 Infinite values check:")
    logging.warning(inf_values) if not inf_values.empty else logging.info("✅ No infinities found.")

    return df_aggregated
