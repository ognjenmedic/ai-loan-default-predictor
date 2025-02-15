import pandas as pd
import numpy as np

def compute_bureau_balance_credit_activity(df_bureau_balance):
    """
    Compute credit activity features from the bureau_balance dataset.
    Groups by SK_ID_BUREAU and calculates:
      - bureau_balance_NUM_MONTHS: total number of records (active months)
      - bureau_balance_NUM_CLOSED_MONTHS: count where STATUS equals "C"
      - bureau_balance_NUM_ACTIVE_MONTHS: count where STATUS equals "0"
      - bureau_balance_NUM_OVERDUE_MONTHS: count where STATUS is in ["1", "2", "3", "4", "5"]
    """
    bureau_balance_credit_activity = df_bureau_balance.groupby("SK_ID_BUREAU").agg(
        bureau_balance_NUM_MONTHS=("MONTHS_BALANCE", "count"),
        bureau_balance_NUM_CLOSED_MONTHS=("STATUS", lambda x: (x == "C").sum()),
        bureau_balance_NUM_ACTIVE_MONTHS=("STATUS", lambda x: (x == "0").sum()),
        bureau_balance_NUM_OVERDUE_MONTHS=("STATUS", lambda x: x.isin(["1", "2", "3", "4", "5"]).sum()),
    ).reset_index()
    return bureau_balance_credit_activity

def compute_bureau_balance_time_features(df_bureau_balance):
    """
    Compute time-related features from the bureau_balance dataset.
    Groups by SK_ID_BUREAU and calculates:
      - bureau_balance_OLDEST_RECORD: the earliest MONTHS_BALANCE
      - bureau_balance_MOST_RECENT_RECORD: the most recent MONTHS_BALANCE
      - bureau_balance_CREDIT_HISTORY_LENGTH: difference between max and min MONTHS_BALANCE
    """
    bureau_balance_time_features = df_bureau_balance.groupby("SK_ID_BUREAU").agg(
        bureau_balance_OLDEST_RECORD=("MONTHS_BALANCE", "min"),
        bureau_balance_MOST_RECENT_RECORD=("MONTHS_BALANCE", "max"),
        bureau_balance_CREDIT_HISTORY_LENGTH=("MONTHS_BALANCE", lambda x: x.max() - x.min())
    ).reset_index()
    return bureau_balance_time_features

def compute_bureau_balance_overdue_features(df_bureau_balance):
    """
    Compute overdue-related features from the bureau_balance dataset.
    First, convert STATUS to an ordered categorical variable, then group by SK_ID_BUREAU.
    Calculates:
      - bureau_balance_NUM_OVERDUE_MONTHS: count of months where STATUS is in ["1", "2", "3", "4", "5"]
      - bureau_balance_MAX_OVERDUE_STATUS: the highest overdue level among ["1", "2", "3", "4", "5"] (or "0" if none)
    """
    # Define the correct order of STATUS categories.
    status_order = ["0", "1", "2", "3", "4", "5", "C", "X"]
    df_tmp = df_bureau_balance.copy()
    df_tmp["STATUS"] = pd.Categorical(df_tmp["STATUS"], categories=status_order, ordered=True)
    
    bureau_balance_overdue_features = df_tmp.groupby("SK_ID_BUREAU").agg(
        bureau_balance_NUM_OVERDUE_MONTHS=("STATUS", lambda x: x.isin(["1", "2", "3", "4", "5"]).sum()),
        bureau_balance_MAX_OVERDUE_STATUS=("STATUS", 
            lambda x: x[x.isin(["1", "2", "3", "4", "5"])].max() if x.isin(["1", "2", "3", "4", "5"]).any() else "0")
    ).reset_index()
    return bureau_balance_overdue_features

def compute_bureau_balance_categorical_features(df_bureau_balance):
    """
    Compute categorical aggregations from the bureau_balance dataset.
    Groups by SK_ID_BUREAU and calculates:
      - bureau_balance_PERCENT_CLOSED: percentage of records with STATUS equal to "C"
      - bureau_balance_PERCENT_ACTIVE: percentage of records with STATUS equal to "0"
      - bureau_balance_MOST_COMMON_STATUS: most frequent STATUS value (or "Unknown" if none)
    """
    bureau_balance_categorical_features = df_bureau_balance.groupby("SK_ID_BUREAU").agg(
        bureau_balance_PERCENT_CLOSED=("STATUS", lambda x: (x == "C").sum() / len(x) if len(x) > 0 else 0),
        bureau_balance_PERCENT_ACTIVE=("STATUS", lambda x: (x == "0").sum() / len(x) if len(x) > 0 else 0),
        bureau_balance_MOST_COMMON_STATUS=("STATUS", lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    ).reset_index()
    return bureau_balance_categorical_features

def generate_bureau_balance_features(df_bureau_balance):
    """
    Generate all engineered bureau_balance features.
    This function computes individual feature groups and then merges them sequentially on SK_ID_BUREAU.
    
    Returns:
      A DataFrame with all engineered bureau_balance features keyed by SK_ID_BUREAU.
    """
    # Compute each feature set
    credit_activity = compute_bureau_balance_credit_activity(df_bureau_balance)
    time_features = compute_bureau_balance_time_features(df_bureau_balance)
    overdue_features = compute_bureau_balance_overdue_features(df_bureau_balance)
    categorical_features = compute_bureau_balance_categorical_features(df_bureau_balance)
    
    # Merge all features on SK_ID_BUREAU
    features_list = [credit_activity, time_features, overdue_features, categorical_features]
    df_features = features_list[0]
    for features in features_list[1:]:
        df_features = df_features.merge(features, on="SK_ID_BUREAU", how="left")
    
    return df_features
