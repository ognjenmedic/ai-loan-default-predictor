import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def map_credit_timeline_to_customer(df_credit_timeline_aggregated, credit_timeline_mapping):
    """
    Map aggregated credit_timeline features (keyed by CREDIT_RECORD_ID) to CUSTOMER_ID.
    
    This function merges the aggregated monthly-level credit_timeline features 
    with the main-level customer key via a mapping table.

    Parameters:
      - df_credit_timeline_aggregated: DataFrame of features keyed by CREDIT_RECORD_ID.
      - credit_timeline_mapping: DataFrame with ['CREDIT_RECORD_ID', 'CUSTOMER_ID'] mapping rows to customers.

    Returns:
      A merged DataFrame where each row has CUSTOMER_ID attached to credit_timeline aggregates.
    """
    logging.info(f"📦 Input shape (aggregated): {df_credit_timeline_aggregated.shape}")
    
    # Merge credit_timeline features with CUSTOMER_ID via the mapping
    df_aggregated_with_customer = df_credit_timeline_aggregated.merge(
        credit_timeline_mapping,
        on='CREDIT_RECORD_ID',
        how='left'
    )

    logging.info(f"✅ Merged shape: {df_aggregated_with_customer.shape}")
    logging.debug(f"Columns: {df_aggregated_with_customer.columns.tolist()}")

    # Confirm CUSTOMER_ID exists after merge
    if 'CUSTOMER_ID' in df_aggregated_with_customer.columns:
        logging.info("Column CUSTOMER_ID found after merge.")
    else:
        logging.error("❌ Column CUSTOMER_ID is missing after merge!")

    # Report missing values
    missing_customer_id = df_aggregated_with_customer['CUSTOMER_ID'].isnull().sum()
    if missing_customer_id > 0:
        logging.warning(f"⚠️ Missing CUSTOMER_ID values: {missing_customer_id}")
    else:
        logging.info("✅ No missing CUSTOMER_ID values.")

    # Show a few rows with missing CUSTOMER_ID
    missing_rows = df_aggregated_with_customer[df_aggregated_with_customer['CUSTOMER_ID'].isnull()]
    if not missing_rows.empty:
        logging.debug(f"🔍 Example rows with missing CUSTOMER_ID:\n{missing_rows.head()}")

    # Try to find a sample numeric column for summary
    sample_col = next(
        (col for col in df_aggregated_with_customer.columns if col.startswith("credit_timeline_agg_") and col.endswith("_mean")),
        None
    )

    if sample_col:
        logging.info(f"Summary for column: {sample_col}")
        logging.info(f"\n Rows WITH CUSTOMER_ID:\n{df_aggregated_with_customer.loc[~df_aggregated_with_customer['CUSTOMER_ID'].isnull(), sample_col].describe()}")
        logging.info(f"\n❌ Rows WITHOUT CUSTOMER_ID:\n{df_aggregated_with_customer.loc[df_aggregated_with_customer['CUSTOMER_ID'].isnull(), sample_col].describe()}")

    # Drop records without CUSTOMER_ID (could not be mapped)
    df_aggregated_with_customer = df_aggregated_with_customer.dropna(subset=["CUSTOMER_ID"])
    logging.info(f"🧹 Shape after dropping missing CUSTOMER_ID: {df_aggregated_with_customer.shape}")

    return df_aggregated_with_customer
