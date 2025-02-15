import pandas as pd
import numpy as np

def aggregate_numeric_features(df_installments_payments):
    """
    Aggregate numeric columns (excluding SK_ID_PREV) with mean, sum, max, min by SK_ID_CURR.
    If 'MISSED_PAYMENT' is present and numeric, it will be included automatically.
    """
    print("\n🔍 [DEBUG] Installments DataFrame Shape Before Aggregation:", df_installments_payments.shape)
    
    numeric_df = df_installments_payments.select_dtypes(include=['number']).drop(columns=['SK_ID_PREV'], errors='ignore')
    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('SK_ID_CURR').agg(agg_funcs)
    
    # Flatten MultiIndex
    agg_numeric.columns = ['installments_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)

    print("\n🔍 [DEBUG] Aggregated Installments Columns Before Merging:")
    print(agg_numeric.columns.tolist())
    print("\n🔍 [DEBUG] Unique SK_ID_CURR Count After Aggregation:", agg_numeric["SK_ID_CURR"].nunique())

    return agg_numeric


def safe_merge(df_main, df_new, merge_on="SK_ID_CURR", name=""):
    """
    Merge two DataFrames on the specified key and print debugging information.
    """
    if df_new is None:
        print(f"\n❌ [ERROR] DataFrame '{name}' is None! Skipping merge.")
        return df_main

    print(f"\n🔍 [DEBUG] Merging '{name}' - Incoming shape: {df_new.shape}")
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how="left")
    print(f"✅ [DEBUG] Merged '{name}': {prev_shape} -> {df_main.shape}")

    missing_cols = set(df_new.columns) - set(df_main.columns)
    if missing_cols:
        print(f"❌ [DEBUG] Columns missing after merging '{name}': {missing_cols}")
    return df_main


def aggregate_installments_payments_features(df_installments_payments, additional_feature_dfs=None):
    """
    1) Aggregates numeric features in df_installments_payments (including MISSED_PAYMENT).
    2) Merges in any additional DataFrames in 'additional_feature_dfs'.
    3) Cleans up NaNs in standard deviation and Payment Ratio columns.
    4) Prints debug info.
    """
    if additional_feature_dfs is None:
        additional_feature_dfs = []
    
    # 1) Basic numeric aggregations
    agg_numeric = aggregate_numeric_features(df_installments_payments)
    print(f"✅ [DEBUG] Aggregation complete. New shape: {agg_numeric.shape}")

    # 2) Merge with additional DataFrames
    df_aggregated = agg_numeric
    for df_new, name in additional_feature_dfs:
        print(f"🔍 [DEBUG] Incoming DataFrame '{name}' shape: {df_new.shape}")
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="SK_ID_CURR", name=name)
    
    # 3) Clean up standard dev columns and payment ratio columns if present
    for col in ["installments_STD_DAYS_ENTRY_PAYMENT", "installments_STD_PAYMENT_DELAY"]:
        if col in df_aggregated.columns:
            df_aggregated[col] = df_aggregated[col].fillna(0)

    for col in df_aggregated.columns:
        # If aggregator created a '..._PAYMENT_RATIO_' column, fill NaN with 1
        if "PAYMENT_RATIO" in col:
            df_aggregated[col] = df_aggregated[col].fillna(1)

    # 4) Debugging: missing values, infinite values, etc.
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    print("\n🔍 Missing Values in Aggregated Installments:")
    if missing_values.empty:
        print("✅ No standard NaN values.")
    else:
        print(missing_values)

    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    print("\n🔍 Infinite Values in Aggregated Installments:")
    if inf_values.empty:
        print("✅ No inf values.")
    else:
        print(inf_values)

    return df_aggregated
