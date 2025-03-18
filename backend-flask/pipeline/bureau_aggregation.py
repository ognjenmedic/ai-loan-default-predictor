import pandas as pd
import numpy as np
from functools import reduce
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def aggregate_numeric_features(df_bureau):
    """
    Aggregate numeric columns (excluding SK_ID_BUREAU) by SK_ID_CURR using mean, sum, max, and min.
    """
    numeric_df = df_bureau.select_dtypes(include=['number']).drop(columns=['SK_ID_BUREAU'], errors='ignore')
    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('SK_ID_CURR').agg(agg_funcs)
    # Flatten MultiIndex columns
    agg_numeric.columns = ['bureau_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    return agg_numeric

def aggregate_categorical_features(df_bureau):
    """
    Aggregate categorical columns (excluding SK_ID_BUREAU) by SK_ID_CURR using the most frequent value.
    """
    categorical_df = df_bureau.select_dtypes(include=['object', 'category']).drop(columns=['SK_ID_BUREAU'], errors='ignore')
    if categorical_df.empty:
        return None
    # Join with SK_ID_CURR for grouping
    categorical_df = df_bureau[['SK_ID_CURR']].join(categorical_df)
    agg_categorical = categorical_df.groupby('SK_ID_CURR').agg(
        lambda x: x.value_counts().idxmax() if not x.empty else "Unknown"
    )
    agg_categorical.columns = ['bureau_agg_' + col + '_most_frequent' for col in agg_categorical.columns]
    agg_categorical.reset_index(inplace=True)
    return agg_categorical

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge numeric and categorical aggregated features.
    """
    if agg_categorical is not None:
        merged = pd.merge(agg_numeric, agg_categorical, on='SK_ID_CURR', how='left')
    else:
        merged = agg_numeric
    return merged

def safe_merge(df_main, df_new, merge_on="SK_ID_CURR", name=""):
    """
    Merge two dataframes and log some debugging information.
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

def aggregate_bureau_features(df_bureau, additional_feature_dfs):
    """
    Aggregate the bureau dataframe by:
      1. Aggregating numeric and categorical features.
      2. Merging in additional engineered feature tables.
    
    Parameters:
      - df_bureau: the original bureau DataFrame (must include SK_ID_CURR).
      - additional_feature_dfs: a list of tuples (df, name) for each additional engineered feature DataFrame.
    
    Returns:
      The aggregated bureau features DataFrame, with all additional engineered features merged in.
    """
    # Step 1: Aggregate raw numeric and categorical features.
    agg_numeric = aggregate_numeric_features(df_bureau)
    agg_categorical = aggregate_categorical_features(df_bureau)
    aggregated = merge_aggregated_features(agg_numeric, agg_categorical)

    # Step 2: Merge in each additional engineered feature DataFrame.
    # Here we assume additional_feature_dfs is provided and non-empty.
    for df_new, name in additional_feature_dfs:
        aggregated = safe_merge(aggregated, df_new, merge_on="SK_ID_CURR", name=name)
    
    return aggregated
