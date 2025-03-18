import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def aggregate_numeric_features(df_credit_card_balance):
    """
    Aggregate numeric columns from the credit card balance DataFrame (excluding SK_ID_PREV)
    using the functions: mean, sum, max, and min.
    """
    # Select numeric columns (drop SK_ID_PREV)
    numeric_df = df_credit_card_balance.select_dtypes(include=['number']).drop(columns=['SK_ID_PREV'], errors='ignore')
    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('SK_ID_CURR').agg(agg_funcs)
    
    # Rename columns to avoid MultiIndex issues and add a prefix
    agg_numeric.columns = ['credit_card_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    
    return agg_numeric

def aggregate_categorical_features(df_credit_card_balance):
    """
    Aggregate categorical columns from the credit card balance DataFrame (excluding SK_ID_PREV)
    by taking the most frequent value for each column.
    """
    categorical_df = df_credit_card_balance.select_dtypes(include=['object', 'category']).drop(columns=['SK_ID_PREV'], errors='ignore')
    
    if not categorical_df.empty:
        # Re-attach SK_ID_CURR for grouping
        categorical_df = df_credit_card_balance[['SK_ID_CURR']].join(categorical_df)
        agg_categorical = categorical_df.groupby('SK_ID_CURR').agg(
            lambda x: x.value_counts().idxmax() if not x.empty else "Unknown"
        )
        # Rename columns with a prefix and suffix for clarity
        agg_categorical.columns = ['credit_card_agg_' + col + '_most_frequent' for col in agg_categorical.columns]
        agg_categorical.reset_index(inplace=True)
        return agg_categorical
    else:
        return None

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge aggregated numeric and categorical features on SK_ID_CURR.
    """
    if agg_categorical is not None:
        merged = agg_numeric.merge(agg_categorical, on='SK_ID_CURR', how='left')
    else:
        merged = agg_numeric
    return merged

def safe_merge(df_main, df_new, merge_on="SK_ID_CURR", name=""):
    """
    Merge two DataFrames on the specified key and log debugging information.
    """
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how="left")
    
    logging.info(f"✅ Merged {name}: {prev_shape} -> {df_main.shape}")

    missing = df_main.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logging.warning(f"🛠️ Missing Values in {name} After Merge:\n{missing}")
    
    return df_main

def aggregate_credit_card_balance_features(df_credit_card_balance, additional_feature_dfs):
    """
    Aggregate the credit card balance DataFrame by:
      1. Aggregating numeric features (excluding SK_ID_PREV) by SK_ID_CURR.
      2. Aggregating categorical features (excluding SK_ID_PREV) by SK_ID_CURR.
      3. Merging the numeric and categorical aggregates.
      4. Merging additional engineered feature tables using safe_merge.
      5. Performing sanity checks for standard NaNs, hidden NaNs (empty strings), and infinite values.
    
    Parameters:
      - df_credit_card_balance: Raw credit card balance DataFrame.
      - additional_feature_dfs: List of tuples (df, name) for additional engineered feature DataFrames.
    
    Returns:
      The final aggregated credit card balance DataFrame.
    """

    logging.info("Starting credit card balance feature aggregation...")
    
    # Step 1: Aggregate numeric features
    agg_numeric = aggregate_numeric_features(df_credit_card_balance)
    
    # Step 2: Aggregate categorical features
    agg_categorical = aggregate_categorical_features(df_credit_card_balance)
    
    # Step 3: Merge numeric and categorical aggregates
    df_aggregated = merge_aggregated_features(agg_numeric, agg_categorical)
    logging.info(f"✅ Aggregation complete. New df_credit_card_balance_aggregated shape: {df_aggregated.shape}")
    
    # Step 4: Merge in each additional engineered feature DataFrame
    for df_new, name in additional_feature_dfs:
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="SK_ID_CURR", name=name)
    
    # Step 5: Sanity checks for missing values, hidden NaNs, and infinite values
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    logging.info("🔍 Standard Missing Values in Aggregated Credit Card Features After Merging:")
    if missing_values.empty:
        logging.info("✅ No standard NaN values detected.")
    else:
        logging.warning(missing_values)
    
    hidden_nans = (df_aggregated == "").sum() + (df_aggregated == "nan").sum()
    hidden_nans = hidden_nans[hidden_nans > 0]
    logging.info("🔍 Hidden NaNs (Empty Strings or 'nan' as Text) in Aggregated Credit Card Features After Merging:")
    if hidden_nans.empty:
        logging.info("✅ No hidden NaNs detected.")
    else:
        logging.warning(hidden_nans)
    
    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    logging.info("🔍 Infinite Values in Aggregated Credit Card Features After Merging:")
    if inf_values.empty:
        logging.info("✅ No Inf values detected.")
    else:
        logging.warning(inf_values)
    
    return df_aggregated
