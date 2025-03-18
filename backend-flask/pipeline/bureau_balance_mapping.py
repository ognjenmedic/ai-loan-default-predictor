import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def map_bureau_balance_to_curr(df_bureau_balance_aggregated, bureau_mapping):
    """
    Map the aggregated bureau balance features (keyed by SK_ID_BUREAU) to SK_ID_CURR
    using a bureau mapping DataFrame that contains 'SK_ID_BUREAU' and 'SK_ID_CURR' columns.

    Parameters:
      - df_bureau_balance_aggregated: DataFrame of aggregated bureau balance features (keyed by SK_ID_BUREAU)
      - bureau_mapping: DataFrame containing the mapping with columns ['SK_ID_BUREAU', 'SK_ID_CURR']

    Returns:
      A DataFrame with the aggregated bureau balance features merged with SK_ID_CURR.
    """
    # Log initial shape before merge
    logging.info(f"df_bureau_balance_aggregated shape before merge: {df_bureau_balance_aggregated.shape}")
    
    # Merge to bring SK_ID_CURR into aggregated bureau balance dataset
    df_aggregated_with_curr = df_bureau_balance_aggregated.merge(
        bureau_mapping,
        on='SK_ID_BUREAU',
        how='left'
    )
    
    # Log shape and columns after merge
    logging.info(f"✅ df_bureau_balance_aggregated_with_curr shape after merge: {df_aggregated_with_curr.shape}")
    logging.debug(f"Columns in the merged DataFrame: {df_aggregated_with_curr.columns.tolist()}")
    
    # Verify that SK_ID_CURR exists
    if 'SK_ID_CURR' in df_aggregated_with_curr.columns:
        logging.info("✅ SK_ID_CURR column found!")
    else:
        logging.error("❌ SK_ID_CURR column not found!")
    
    # Count missing SK_ID_CURR values
    missing_sk_id_curr = df_aggregated_with_curr['SK_ID_CURR'].isnull().sum()
    logging.warning(f"⚠️ Number of missing SK_ID_CURR values: {missing_sk_id_curr}" if missing_sk_id_curr > 0 else "✅ No missing SK_ID_CURR values.")
    
    # Display sample rows from the merged dataframe
    logging.debug(f"\n🔍 Sample rows from the merged DataFrame:\n{df_aggregated_with_curr.head()}")
    
    # Display rows where SK_ID_CURR is missing
    missing_rows = df_aggregated_with_curr[df_aggregated_with_curr['SK_ID_CURR'].isnull()]
    if not missing_rows.empty:
        logging.warning(f"\n⚠️ Rows with missing SK_ID_CURR:\n{missing_rows.head()}")
    
    # Find a sample aggregated numeric column to show summary statistics
    sample_col = None
    for col in df_aggregated_with_curr.columns:
        if col.startswith("bureau_balance_agg_") and col.endswith("_mean"):
            sample_col = col
            break

    if sample_col:
        mask_missing = df_aggregated_with_curr['SK_ID_CURR'].isnull()
        mask_valid = ~mask_missing
        logging.info(f"\nSummary for rows with SK_ID_CURR:\n{df_aggregated_with_curr.loc[mask_valid, sample_col].describe()}")
        logging.info(f"\nSummary for rows without SK_ID_CURR:\n{df_aggregated_with_curr.loc[mask_missing, sample_col].describe()}")
    
    # Drop rows where SK_ID_CURR is missing
    df_aggregated_with_curr = df_aggregated_with_curr.dropna(subset=['SK_ID_CURR'])
    logging.info(f"✅ New shape after dropping missing SK_ID_CURR: {df_aggregated_with_curr.shape}")
    
    return df_aggregated_with_curr
