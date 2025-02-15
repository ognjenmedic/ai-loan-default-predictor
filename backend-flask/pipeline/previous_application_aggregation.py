import pandas as pd
import numpy as np

def aggregate_numeric_features(df_previous_application):
    """
    Aggregate numeric columns from the previous application DataFrame (excluding SK_ID_PREV)
    using mean, sum, max, and min, grouping by SK_ID_CURR.
    """
    # Exclude the loan-level unique identifier SK_ID_PREV
    numeric_df = df_previous_application.select_dtypes(include=['number']).drop(columns=['SK_ID_PREV'], errors='ignore')


    if 'DAYS_DECISION_BIN' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['DAYS_DECISION_BIN'])


    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('SK_ID_CURR').agg(agg_funcs)
    
    # Flatten MultiIndex columns and add a prefix
    agg_numeric.columns = ['previous_app_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    
    return agg_numeric

def aggregate_categorical_features(df_previous_application):
    """
    Aggregate categorical columns from the previous application DataFrame (excluding SK_ID_PREV)
    by taking the most frequent value for each column, grouping by SK_ID_CURR.
    """
    categorical_df = df_previous_application.select_dtypes(include=['object', 'category']).drop(columns=['SK_ID_PREV'], errors='ignore')
    
    if not categorical_df.empty:
        # Re-attach SK_ID_CURR for grouping
        categorical_df = df_previous_application[['SK_ID_CURR']].join(categorical_df)
        agg_categorical = categorical_df.groupby('SK_ID_CURR').agg(
            lambda x: x.value_counts().idxmax() if not x.empty else "Unknown"
        )
        agg_categorical.columns = ['previous_app_agg_' + col + '_most_frequent' for col in agg_categorical.columns]
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
    Merge two DataFrames on the specified key and print debugging information.
    """
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how='left')
    
    print(f"✅ Merged {name}: {prev_shape} -> {df_main.shape}")
    missing_values = df_main.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print(f"🛠️ Missing Values in {name} After Merge:\n{missing_values}")
    print("-" * 50)
    
    return df_main

def aggregate_previous_application_features(df_previous_application, additional_feature_dfs):
    """
    Aggregate the previous application DataFrame by:
      1. Aggregating numeric features (excluding SK_ID_PREV) by SK_ID_CURR.
      2. Aggregating categorical features (excluding SK_ID_PREV) by SK_ID_CURR.
      3. Merging the numeric and categorical aggregates.
      4. Merging additional engineered feature tables via safe_merge.
      5. Handling missing values in standard deviation columns.
      6. Performing sanity checks for missing values, hidden NaNs, and infinite values.
    
    Parameters:
      - df_previous_application: Raw previous application DataFrame.
      - additional_feature_dfs: List of tuples (df, descriptive_name) for additional engineered feature DataFrames.
    
    Returns:
      The final aggregated previous application DataFrame.
    """
    # Step 1: Aggregate numeric features
    agg_numeric = aggregate_numeric_features(df_previous_application)
    
    # Step 2: Aggregate categorical features
    agg_categorical = aggregate_categorical_features(df_previous_application)
    
    # Step 3: Merge numeric and categorical aggregates
    df_aggregated = merge_aggregated_features(agg_numeric, agg_categorical)
    print(f"✅ Aggregation complete. New df_previous_application_aggregated shape: {df_aggregated.shape}")
    
    # Step 4: Merge in each additional engineered feature DataFrame
    for df_new, name in additional_feature_dfs:
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="SK_ID_CURR", name=name)
    
    # Step 5: Handle NaNs in STD columns (if they exist)
    if "previous_app_STD_APPLICATION_AMOUNT" in df_aggregated.columns:
        df_aggregated["previous_app_STD_APPLICATION_AMOUNT"] = df_aggregated["previous_app_STD_APPLICATION_AMOUNT"].fillna(0)
    if "previous_app_STD_APPROVED_AMOUNT" in df_aggregated.columns:
        df_aggregated["previous_app_STD_APPROVED_AMOUNT"] = df_aggregated["previous_app_STD_APPROVED_AMOUNT"].fillna(0)
    
    # Step 6: Sanity checks
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    print("\n🔍 Standard Missing Values in Aggregated Previous Application Features After Merging:")
    if missing_values.empty:
        print("✅ No standard NaN values detected.")
    else:
        print(missing_values)
    
    hidden_nans = (df_aggregated == "").sum() + (df_aggregated == "nan").sum()
    hidden_nans = hidden_nans[hidden_nans > 0]
    print("\n🔍 Hidden NaNs in Aggregated Previous Application Features After Merging:")
    if hidden_nans.empty:
        print("✅ No hidden NaNs detected.")
    else:
        print(hidden_nans)
    
    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    print("\n🔍 Infinite Values in Aggregated Previous Application Features After Merging:")
    if inf_values.empty:
        print("✅ No Inf values detected.")
    else:
        print(inf_values)
    
    return df_aggregated
