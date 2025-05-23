import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def aggregate_numeric_features(df_cash_pos_records):
    """
    Aggregate numeric columns (excluding PRIOR_LOAN_ID) using mean, sum, max, and min,
    grouped by CUSTOMER_ID. Derived synthetic features like LOAN_AGE_GROUP are excluded.
    """
    logging.info(f"Aggregating numeric features — Input shape: {df_cash_pos_records.shape}")

    numeric_df = df_cash_pos_records.select_dtypes(include=['number']).drop(columns=['PRIOR_LOAN_ID'], errors='ignore')

    if 'LOAN_AGE_GROUP' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['LOAN_AGE_GROUP'])

    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('CUSTOMER_ID').agg(agg_funcs)

    agg_numeric.columns = ['cash_pos_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)

    logging.info(f"✅ Aggregated numeric features — Output shape: {agg_numeric.shape}")
    return agg_numeric

def aggregate_categorical_features(df_cash_pos_records):
    """
    Aggregate categorical columns (excluding PRIOR_LOAN_ID) by most frequent value per column.
    """
    logging.info("🔍 Aggregating categorical features...")

    categorical_df = df_cash_pos_records.select_dtypes(include=['object', 'category']).drop(columns=['PRIOR_LOAN_ID'], errors='ignore')
    
    if categorical_df.empty:
        logging.warning("⚠️ No categorical features found.")
        return None

    categorical_df = df_cash_pos_records[['CUSTOMER_ID']].join(categorical_df)

    agg_categorical = categorical_df.groupby('CUSTOMER_ID').agg(
        lambda x: x.value_counts().idxmax() if not x.empty else "Unknown"
    )
    agg_categorical.columns = ['cash_pos_agg_' + col + '_most_frequent' for col in agg_categorical.columns]
    agg_categorical.reset_index(inplace=True)

    logging.info(f"✅ Aggregated categorical features — Output shape: {agg_categorical.shape}")
    return agg_categorical

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge numeric and categorical aggregates on CUSTOMER_ID.
    """
    logging.info("🔄 Merging numeric and categorical aggregates...")
    if agg_categorical is not None:
        merged = agg_numeric.merge(agg_categorical, on='CUSTOMER_ID', how='left')
    else:
        merged = agg_numeric
    logging.info(f"✅ Merged result — Shape: {merged.shape}")
    return merged

def safe_merge(df_main, df_new, merge_on="CUSTOMER_ID", name=""):
    """
    Merge with logging and diagnostic reporting.
    """
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how="left")
    logging.info(f"✅ Merged '{name}': {prev_shape} → {df_main.shape}")

    missing = df_main.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logging.warning(f"🛠️ Missing values after merging '{name}':\n{missing}")
    return df_main

def aggregate_cash_pos_records_features(df_cash_pos_records, additional_feature_dfs):
    """
    Perform full aggregation for cash_pos_records features:
      1. Aggregate numeric columns with standard stats.
      2. Aggregate categorical columns using mode.
      3. Merge both into a single DataFrame.
      4. Merge in additional engineered features.
      5. Run final data integrity checks.
    """
    logging.info(f"Starting cash_pos_records aggregation... Input shape: {df_cash_pos_records.shape}")

    agg_numeric = aggregate_numeric_features(df_cash_pos_records)
    agg_categorical = aggregate_categorical_features(df_cash_pos_records)

    df_aggregated = merge_aggregated_features(agg_numeric, agg_categorical)

    for df_new, name in additional_feature_dfs:
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="CUSTOMER_ID", name=name)

    logging.info("🔍 Checking for standard NaN values...")
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    logging.warning(missing_values) if not missing_values.empty else logging.info("✅ No standard NaNs detected.")

    logging.info("🔍 Checking for hidden NaNs (empty strings or text 'nan')...")
    hidden_nans = (df_aggregated == "").sum() + (df_aggregated == "nan").sum()
    hidden_nans = hidden_nans[hidden_nans > 0]
    logging.warning(hidden_nans) if not hidden_nans.empty else logging.info("✅ No hidden NaNs detected.")

    logging.info("🔍 Checking for infinite values...")
    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    logging.warning(inf_values) if not inf_values.empty else logging.info("✅ No infinite values detected.")

    logging.info("Aggregation complete.")
    return df_aggregated
