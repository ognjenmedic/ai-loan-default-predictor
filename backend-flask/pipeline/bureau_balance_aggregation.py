import pandas as pd
import numpy as np

def aggregate_numeric_features(df_bureau_balance):
    """
    Aggregate numeric columns from the bureau_balance dataframe (keeping SK_ID_BUREAU for grouping)
    using the functions: mean, sum, max, and min.
    """
    # Select numeric columns (SK_ID_BUREAU remains for grouping)
    numeric_df = df_bureau_balance.select_dtypes(include=['number'])
    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('SK_ID_BUREAU').agg(agg_funcs)
    
    # Rename columns to avoid a MultiIndex (prefix with 'bureau_balance_agg_')
    agg_numeric.columns = ['bureau_balance_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    
    return agg_numeric

def aggregate_categorical_features(df_bureau_balance):
    """
    Aggregate categorical columns from the bureau_balance dataframe (excluding SK_ID_BUREAU)
    by taking the most frequent value.
    """
    # Select categorical columns and drop SK_ID_BUREAU from them
    cat_df = df_bureau_balance.select_dtypes(include=['object', 'category']).drop(columns=['SK_ID_BUREAU'], errors='ignore')
    
    if not cat_df.empty:
        # Join SK_ID_BUREAU for grouping
        cat_df = df_bureau_balance[['SK_ID_BUREAU']].join(cat_df)
        # Group by SK_ID_BUREAU using value_counts().idxmax() for speed
        agg_categorical = cat_df.groupby('SK_ID_BUREAU').agg(lambda x: x.value_counts().idxmax() if not x.empty else "Unknown")
        # Rename columns with a prefix and a suffix
        agg_categorical.columns = ['bureau_balance_agg_' + col + '_most_frequent' for col in agg_categorical.columns]
        agg_categorical.reset_index(inplace=True)
        return agg_categorical
    else:
        return None

def merge_aggregated_features(agg_numeric, agg_categorical):
    """
    Merge aggregated numeric and categorical features on SK_ID_BUREAU.
    """
    if agg_categorical is not None:
        merged = agg_numeric.merge(agg_categorical, on="SK_ID_BUREAU", how="left")
    else:
        merged = agg_numeric
    return merged

def safe_merge(df_main, df_new, merge_on="SK_ID_BUREAU", name=""):
    """
    Merge two DataFrames on the given key and print debugging information.
    """
    prev_shape = df_main.shape
    df_main = df_main.merge(df_new, on=merge_on, how="left")
    
    print(f"✅ Merged {name}: {prev_shape} -> {df_main.shape}")
    missing = df_main.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"🛠️ Missing Values in {name} After Merge:\n{missing}")
    print("-" * 50)
    
    return df_main

def aggregate_bureau_balance_features(df_bureau_balance, additional_feature_dfs):
    """
    Aggregate the bureau_balance dataframe by performing the following steps:
      1. Aggregate numeric features (grouped by SK_ID_BUREAU).
      2. Aggregate categorical features (grouped by SK_ID_BUREAU).
      3. Merge the numeric and categorical aggregates.
      4. Merge additional engineered feature DataFrames (provided as a list of tuples: (df, name))
         using the safe_merge function.
      5. Log some sanity checks for missing, hidden, and infinite values.
    
    Parameters:
      - df_bureau_balance: Raw bureau_balance DataFrame.
      - additional_feature_dfs: List of tuples, where each tuple is (engineered_df, descriptive_name).
    
    Returns:
      The final aggregated bureau_balance DataFrame.
    """
    # Step 1: Aggregate numeric features.
    agg_numeric = aggregate_numeric_features(df_bureau_balance)
    
    # Step 2: Aggregate categorical features.
    agg_categorical = aggregate_categorical_features(df_bureau_balance)
    
    # Step 3: Merge numeric and categorical aggregates.
    df_aggregated = merge_aggregated_features(agg_numeric, agg_categorical)
    print(f"✅ Aggregation complete. New df_bureau_balance_aggregated shape: {df_aggregated.shape}")
    
    # Step 4: Merge in each additional engineered feature DataFrame.
    for df_new, name in additional_feature_dfs:
        df_aggregated = safe_merge(df_aggregated, df_new, merge_on="SK_ID_BUREAU", name=name)
    
    # Step 5: Sanity checks for missing values, hidden NaNs, and infinite values.
    missing_values = df_aggregated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    print("\n🔍 Standard Missing Values in Aggregated Bureau Balance Features After Merging:")
    print(missing_values if not missing_values.empty else "✅ No standard NaN values detected.")
    
    hidden_nans = (df_aggregated == "").sum() + (df_aggregated == "nan").sum()
    hidden_nans = hidden_nans[hidden_nans > 0]
    print("\n🔍 Hidden NaNs in Aggregated Bureau Balance Features After Merging:")
    print(hidden_nans if not hidden_nans.empty else "✅ No hidden NaNs detected.")
    
    inf_values = df_aggregated.replace([np.inf, -np.inf], np.nan).isna().sum()
    inf_values = inf_values[inf_values > 0]
    print("\n🔍 Infinite Values in Aggregated Bureau Balance Features After Merging:")
    if inf_values.empty:
        print("✅ No Inf values detected.")
    else:
        print(inf_values)
    
    return df_aggregated


def final_aggregate_bureau_balance_by_sk_id_curr(df_bureau_balance_aggregated_with_curr):
    """
    1. Takes a DataFrame keyed by SK_ID_BUREAU but already containing SK_ID_CURR.
    2. Aggregates to a single row per SK_ID_CURR using min, max, mean, sum, etc.
    3. Returns a DataFrame keyed by SK_ID_CURR.
    """
    print("Before final aggregation:")
    print("Unique SK_ID_CURR:", df_bureau_balance_aggregated_with_curr['SK_ID_CURR'].nunique())
    print("Total rows:", df_bureau_balance_aggregated_with_curr.shape[0])
    
    # Example: convert columns that should be numeric
    cols_to_convert = ['bureau_balance_MAX_OVERDUE_STATUS']  # if you have a column like this
    for col in cols_to_convert:
        if col in df_bureau_balance_aggregated_with_curr.columns:
            df_bureau_balance_aggregated_with_curr[col] = pd.to_numeric(
                df_bureau_balance_aggregated_with_curr[col], errors='coerce'
            )

    # Define your final aggregation dictionary (adjust as needed):
    agg_dict_final = {
        # EXAMPLE: aggregated columns from the SK_ID_BUREAU-level step
        'bureau_balance_agg_MONTHS_BALANCE_mean': ['min', 'max', 'mean', 'sum'],
        'bureau_balance_agg_MONTHS_BALANCE_sum':  ['min', 'max', 'mean', 'sum'],
        'bureau_balance_agg_MONTHS_BALANCE_max':  ['min', 'max', 'mean', 'sum'],
        'bureau_balance_agg_MONTHS_BALANCE_min':  ['min', 'max', 'mean', 'sum'],
        
        # If you have categorical columns aggregated at SK_ID_BUREAU level:
        'bureau_balance_agg_STATUS_most_frequent': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],

        # Example numeric columns from earlier feature engineering
        'bureau_balance_NUM_MONTHS':              ['min', 'max', 'mean', 'sum'],
        'bureau_balance_NUM_CLOSED_MONTHS':       ['min', 'max', 'mean', 'sum'],
        'bureau_balance_NUM_ACTIVE_MONTHS':       ['min', 'max', 'mean', 'sum'],
        'bureau_balance_TOTAL_OVERDUE_MONTHS':    ['min', 'max', 'mean', 'sum'],
        'bureau_balance_OLDEST_RECORD':           ['min', 'max', 'mean', 'sum'],
        'bureau_balance_MOST_RECENT_RECORD':      ['min', 'max', 'mean', 'sum'],
        'bureau_balance_CREDIT_HISTORY_LENGTH':   ['min', 'max', 'mean', 'sum'],
        'bureau_balance_RECENT_OVERDUE_MONTHS':   ['min', 'max', 'mean', 'sum'],
        'bureau_balance_MAX_OVERDUE_STATUS':      ['min', 'max', 'mean', 'sum'],
        'bureau_balance_PERCENT_CLOSED':          ['min', 'max', 'mean', 'sum'],
        'bureau_balance_PERCENT_ACTIVE':          ['min', 'max', 'mean', 'sum'],
        'bureau_balance_MOST_COMMON_STATUS':      lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
    }

    # Drop SK_ID_BUREAU, since we only need to group by SK_ID_CURR now
    df_temp = df_bureau_balance_aggregated_with_curr.drop('SK_ID_BUREAU', axis=1)

    # Group by SK_ID_CURR
    df_bureau_balance_aggregated_with_curr_final = df_temp.groupby('SK_ID_CURR', as_index=False).agg(agg_dict_final)

    # Flatten multi-index columns
    df_bureau_balance_aggregated_with_curr_final.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in df_bureau_balance_aggregated_with_curr_final.columns
    ]

    # Ensure SK_ID_CURR is integer
    df_bureau_balance_aggregated_with_curr_final['SK_ID_CURR'] = df_bureau_balance_aggregated_with_curr_final['SK_ID_CURR'].astype('int64')

    print("\nAfter final aggregation:")
    print("Unique SK_ID_CURR:", df_bureau_balance_aggregated_with_curr_final['SK_ID_CURR'].nunique())
    print("Total rows:", df_bureau_balance_aggregated_with_curr_final.shape[0])
    
    return df_bureau_balance_aggregated_with_curr_final