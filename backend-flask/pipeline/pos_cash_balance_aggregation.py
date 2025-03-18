import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def aggregate_numeric_features(df_pos_cash_balance):
    """
    Aggregate numeric columns from the pos cash balance DataFrame (excluding SK_ID_PREV)
    using mean, sum, max, and min, grouping by SK_ID_CURR.
    """

    logging.info(f"🔍 Aggregating numeric features from POS Cash Balance - Initial shape: {df_pos_cash_balance.shape}")

    # Select numeric columns and drop SK_ID_PREV
    numeric_df = df_pos_cash_balance.select_dtypes(include=['number']).drop(columns=['SK_ID_PREV'], errors='ignore')

    if 'LOAN_AGE_GROUP' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['LOAN_AGE_GROUP'])


    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('SK_ID_CURR').agg(agg_funcs)
    
    # Flatten MultiIndex columns and add prefix
    agg_numeric.columns = ['pos_cash_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)

    logging.info(f"✅ Aggregated numeric features - New shape: {agg_numeric.shape}")
    
    return agg_numeric

def aggregate_categorical_features(df_pos_cash_balance):
    """
    Aggregate categorical columns from the pos cash balance DataFrame (excluding SK_ID_PREV)
    by taking the most frequent value for each column, grouping by SK_ID_CURR.
    """

    logging.info("🔍 Aggregating categorical features from POS Cash Balance...")
    
    categorical_df = df_pos_cash_balance.select_dtypes(include=['object', 'category']).drop(columns=['SK_ID_PREV'], errors='ignore')
    
    if not categorical_df.empty:
        # Re-attach SK_ID_CURR for grouping
        categorical_df = df_pos_cash_balance[['SK_ID_CURR']].join(categorical_df)
        agg_categorical = categorical_df.groupby('SK_ID_CURR').agg(
            lambda x: x.value_counts().idxmax() if not x.empty else "Unknown"
        )
        # Rename columns with prefix and suffix for clarity
        agg_categorical.columns = ['pos_cash_agg_' + col + '_most_frequent' for col in agg_categorical.columns]
        agg_categorical.reset_index(inplace=True)

        logging.info(f"✅ Aggregated categorical features - New shape: {agg_categorical.shape}")
        return agg_categorical
    else:
        logging.warning("⚠️ No categorical features found in POS Cash Balance.")
        return None

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge aggregated numeric and categorical features on SK_ID_CURR.
    """

    logging.info("🔍 Merging numeric and categorical aggregated features...")

    if agg_categorical is not None:
        merged = agg_numeric.merge(agg_categorical, on='SK_ID_CURR', how='left')
    else:
        merged = agg_numeric
    
    logging.info(f"✅ Merged features - New shape: {merged.shape}")
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

def aggregate_pos_cash_balance_features(df_pos_cash_balance, additional_feature_dfs):
    """
    1) Numeric aggregator => 'pos_cash_agg_...'
    2) Categorical aggregator => 'pos_cash_agg_*_most_frequent'
    3) Merge numeric + categorical
    4) Merge in additional_feature_dfs => e.g. pos_cash_engineered_feats from generate_pos_cash_balance_features
    5) Return final
    """

    logging.info(f"Starting aggregation for POS Cash Balance dataset... Shape: {df_pos_cash_balance.shape}")
    
    # 1) Numeric aggregator
    agg_numeric = aggregate_numeric_features(df_pos_cash_balance)
    
    # 2) Categorical aggregator
    agg_categorical = aggregate_categorical_features(df_pos_cash_balance)
    
    # 3) Merge numeric + categorical
    df_aggregated = merge_aggregated_features(agg_numeric, agg_categorical)
    
    # 4) Merge in additional engineered DFs
    for df_new, name in additional_feature_dfs:
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="SK_ID_CURR", name=name)
    
    logging.info("✅ All POS Cash Balance feature tables merged successfully!")
    
    # 5) Sanity checks
    
    # Check standard missing values
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    logging.info("\n🔍 Standard Missing Values in Aggregated POS Cash Balance Features After Merging:")
    if missing_values.empty:
        logging.info("✅ No standard NaN values detected.")
    else:
        logging.warning(missing_values)
    
    # Check for hidden NaNs (empty strings or "nan" as text)
    hidden_nans = (df_aggregated == "").sum() + (df_aggregated == "nan").sum()
    hidden_nans = hidden_nans[hidden_nans > 0]
    logging.info("\n🔍 Hidden NaNs (Empty Strings or 'nan' as Text) in Aggregated POS Cash Balance Features After Merging:")
    if hidden_nans.empty:
        logging.info("✅ No hidden NaNs detected.")
    else:
        logging.warning(hidden_nans)
    
    # Check for infinite values
    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    logging.info("\nInfinite Values in Aggregated POS Cash Balance Features After Merging:")
    if inf_values.empty:
        logging.info("✅ No Inf values detected.")
    else:
        logging.warning(inf_values)
    
    return df_aggregated
