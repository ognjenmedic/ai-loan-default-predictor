import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def aggregate_numeric_features(df_prior_loan_history):
    """
    Aggregate numeric columns (excluding PRIOR_LOAN_ID and DAYS_SINCE_DECISION_BIN)
    using mean, sum, max, and min — grouped by CUSTOMER_ID.
    """
    numeric_df = df_prior_loan_history.select_dtypes(include=['number']).drop(columns=['PRIOR_LOAN_ID'], errors='ignore')

    if 'DAYS_SINCE_DECISION_BIN' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['DAYS_SINCE_DECISION_BIN'])

    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('CUSTOMER_ID').agg(agg_funcs)
    agg_numeric.columns = ['prior_loan_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)

    logging.info(f"✅ Numeric aggregation complete. Shape: {agg_numeric.shape}")
    return agg_numeric

def aggregate_categorical_features(df_prior_loan_history):
    """
    Aggregate categorical columns by most frequent value per CUSTOMER_ID.
    PRIOR_LOAN_ID is excluded.
    """
    categorical_df = df_prior_loan_history.select_dtypes(include=['object', 'category']).drop(columns=['PRIOR_LOAN_ID'], errors='ignore')

    if categorical_df.empty:
        logging.warning("⚠️ No categorical features found for aggregation.")
        return None

    categorical_df = df_prior_loan_history[['CUSTOMER_ID']].join(categorical_df)
    agg_categorical = categorical_df.groupby('CUSTOMER_ID').agg(
        lambda x: x.value_counts().idxmax() if not x.empty else "Unknown"
    )
    agg_categorical.columns = ['prior_loan_agg_' + col + '_most_frequent' for col in agg_categorical.columns]
    agg_categorical.reset_index(inplace=True)

    logging.info(f"✅ Categorical aggregation complete. Shape: {agg_categorical.shape}")
    return agg_categorical

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge numeric and categorical aggregates.
    """
    if agg_categorical is not None:
        merged = agg_numeric.merge(agg_categorical, on='CUSTOMER_ID', how='left')
        logging.info(f"✅ Merged numeric and categorical features. Final shape: {merged.shape}")
    else:
        merged = agg_numeric
    return merged

def safe_merge(df_main, df_new, merge_on="CUSTOMER_ID", name=""):
    """
    Merge two DataFrames with logging and diagnostics.
    """
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how='left')

    logging.info(f"Merged '{name}': {prev_shape} → {df_main.shape}")
    missing = df_main.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logging.warning(f"⚠️ Missing values in '{name}' after merge:\n{missing}")
    return df_main

def aggregate_prior_loan_history_features(df_prior_loan_history, additional_feature_dfs):
    """
    Perform full aggregation on prior loan history data:
      1. Aggregate numeric features by CUSTOMER_ID.
      2. Aggregate categorical features (mode).
      3. Merge both aggregates.
      4. Merge additional engineered feature sets.
      5. Handle known NaNs in std-dev columns.
      6. Log diagnostics for NaNs, hidden strings, and infinities.
    """
    logging.info("Starting full aggregation of prior loan history features...")

    agg_numeric = aggregate_numeric_features(df_prior_loan_history)
    agg_categorical = aggregate_categorical_features(df_prior_loan_history)
    df_aggregated = merge_aggregated_features(agg_numeric, agg_categorical)
    logging.info(f"✅ Core aggregation complete. Shape: {df_aggregated.shape}")

    for df_new, name in additional_feature_dfs:
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="CUSTOMER_ID", name=name)

    # Fill known std-dev NaNs
    for col in ["prior_loan_STD_APPLICATION_AMOUNT", "prior_loan_STD_APPROVED_AMOUNT"]:
        if col in df_aggregated.columns:
            df_aggregated[col] = df_aggregated[col].fillna(0)

    # Sanity checks
    logging.info("🔍 Checking for missing values...")
    missing = df_aggregated.isna().sum()
    missing = missing[missing > 0]
    logging.warning(missing) if not missing.empty else logging.info("✅ No standard NaNs.")

    logging.info("🔍 Checking for hidden string-based NaNs...")
    hidden_nans = (df_aggregated == "").sum() + (df_aggregated == "nan").sum()
    hidden_nans = hidden_nans[hidden_nans > 0]
    logging.warning(hidden_nans) if not hidden_nans.empty else logging.info("✅ No hidden NaNs.")

    logging.info("🔍 Checking for infinite values...")
    inf_check = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_check = inf_check[inf_check > 0]
    logging.warning(inf_check) if not inf_check.empty else logging.info("✅ No infinite values.")

    return df_aggregated
