import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def aggregate_numeric_features(df_credit_timeline):
    """
    Aggregate numeric columns from credit_timeline (grouped by CREDIT_RECORD_ID),
    using standard summary statistics.
    """
    numeric_df = df_credit_timeline.select_dtypes(include=["number"])
    agg_funcs = ["mean", "sum", "max", "min"]
    agg_numeric = numeric_df.groupby("CREDIT_RECORD_ID").agg(agg_funcs)

    agg_numeric.columns = [f"credit_timeline_agg_{col}_{stat}" for col, stat in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    return agg_numeric

def aggregate_categorical_features(df_credit_timeline):
    """
    Aggregate categorical columns from credit_timeline by computing mode per CREDIT_RECORD_ID.
    Falls back to 'Unknown' if no mode can be determined.
    """
    cat_df = df_credit_timeline.select_dtypes(include=["object", "category"]).drop(columns=["CREDIT_RECORD_ID"], errors="ignore")
    
    if not cat_df.empty:
        cat_df = df_credit_timeline[["CREDIT_RECORD_ID"]].join(cat_df)

        def safe_mode(x):
            return x.value_counts().idxmax() if not x.empty else "Unknown"

        agg_categorical = cat_df.groupby("CREDIT_RECORD_ID").agg(safe_mode)
        agg_categorical.columns = [f"credit_timeline_agg_{col}_most_frequent" for col in agg_categorical.columns]
        agg_categorical.reset_index(inplace=True)
        return agg_categorical
    else:
        return None

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge numeric and categorical aggregates on CREDIT_RECORD_ID.
    """
    if agg_categorical is not None:
        merged = agg_numeric.merge(agg_categorical, on="CREDIT_RECORD_ID", how="left")
    else:
        merged = agg_numeric
    return merged

def safe_merge(df_main, df_new, merge_on="CREDIT_RECORD_ID", name=""):
    """
    Merge and log shape/missing info to help debug feature pipelines.
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

def aggregate_credit_timeline_features(df_credit_timeline, additional_feature_dfs):
    """
    Aggregate the credit_timeline table:
      1. Aggregate numeric + categorical features by CREDIT_RECORD_ID.
      2. Merge additional engineered feature tables (if any).
      3. Run sanity checks on missing or infinite values.
    """
    agg_numeric = aggregate_numeric_features(df_credit_timeline)
    agg_categorical = aggregate_categorical_features(df_credit_timeline)
    df_aggregated = merge_aggregated_features(agg_numeric, agg_categorical)
    logging.info(f"✅ Aggregation complete. Shape: {df_aggregated.shape}")

    for df_new, name in additional_feature_dfs:
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="CREDIT_RECORD_ID", name=name)

    # --- Sanity Checks ---
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    logging.info("\n🔍 Standard Missing Values After Merge:")
    logging.warning(missing_values) if not missing_values.empty else logging.info("✅ None found.")

    hidden_nans = (df_aggregated == "").sum() + (df_aggregated == "nan").sum()
    hidden_nans = hidden_nans[hidden_nans > 0]
    logging.info("\n🔍 Hidden NaNs:")
    logging.warning(hidden_nans) if not hidden_nans.empty else logging.info("✅ None found.")

    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    logging.info("🔍 Infinite Values:")
    logging.warning(inf_values) if not inf_values.empty else logging.info("✅ None found.")

    return df_aggregated

def final_aggregate_credit_timeline_by_customer_id(df_credit_timeline_aggregated_with_curr):
    """
    Final aggregation step:
      - Converts CREDIT_RECORD_ID-level data (already joined with CUSTOMER_ID) into CUSTOMER_ID-level aggregates.
      - Applies multiple summary functions (min, max, mean, sum, or mode).
    """
    logging.info("Before final aggregation:")
    logging.info(f"Unique CUSTOMER_ID: {df_credit_timeline_aggregated_with_curr['CUSTOMER_ID'].nunique()}")
    logging.info(f"Total rows: {df_credit_timeline_aggregated_with_curr.shape[0]}")

    # Remove CREDIT_RECORD_ID (no longer needed)
    df_temp = df_credit_timeline_aggregated_with_curr.drop("CREDIT_RECORD_ID", axis=1)

    # Final aggregation rules
    agg_dict_final = {
        col: ["min", "max", "mean", "sum"]
        for col in df_temp.select_dtypes(include=["number"]).columns
        if col != "CUSTOMER_ID"
    }

    # Add custom aggregation for any remaining categorical columns
    cat_cols = df_temp.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        agg_dict_final[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]

    # Group by CUSTOMER_ID
    df_final = df_temp.groupby("CUSTOMER_ID", as_index=False).agg(agg_dict_final)

    # Flatten MultiIndex columns
    df_final.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in df_final.columns
    ]

    # Ensure CUSTOMER_ID is clean
    df_final["CUSTOMER_ID"] = df_final["CUSTOMER_ID"].astype("int64")

    logging.info("After final aggregation:")
    logging.info(f"Unique CUSTOMER_ID: {df_final['CUSTOMER_ID'].nunique()}")
    logging.info(f"Total rows: {df_final.shape[0]}")

    return df_final
