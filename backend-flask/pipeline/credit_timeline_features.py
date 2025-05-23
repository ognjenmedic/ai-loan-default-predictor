import pandas as pd
import numpy as np
import logging
from tools.feature_config import RAW_SCHEMA, DTYPE_MAP, DEFAULT_CATS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Use synthetic ACCOUNT_STATUS values from config
ACCOUNT_STATUS_VALUES = DEFAULT_CATS["ACCOUNT_STATUS"]
ACCOUNT_STATUS_ON_TIME = ACCOUNT_STATUS_VALUES[0]     
ACCOUNT_STATUS_DELAYED = ACCOUNT_STATUS_VALUES[1]     
ACCOUNT_STATUS_DEFAULTED = ACCOUNT_STATUS_VALUES[2]   

OVERDUE_ACCOUNT_STATUS_VALUES = ACCOUNT_STATUS_VALUES[1:]  


def compute_credit_timeline_activity(df_credit_timeline):
    """
    Compute synthetic credit activity indicators from the credit_timeline dataset.
    Aggregates by CREDIT_RECORD_ID and returns:
      - Total number of monthly records (activity span)
      - Number of 'closed' months (ACCOUNT_STATUS == synthetic 'Defaulted')
      - Number of 'on-time' months (ACCOUNT_STATUS == synthetic 'On Time')
      - Number of overdue months (ACCOUNT_STATUS in ['Delayed', 'Defaulted'])
    """
    credit_timeline_activity = df_credit_timeline.groupby("CREDIT_RECORD_ID").agg(
        credit_timeline_NUM_MONTHS=("TIME_SINCE_ENTRY", "count"),
        credit_timeline_NUM_CLOSED_MONTHS=("ACCOUNT_STATUS", lambda x: (x == ACCOUNT_STATUS_DEFAULTED).sum()),
        credit_timeline_NUM_ACTIVE_MONTHS=("ACCOUNT_STATUS", lambda x: (x == ACCOUNT_STATUS_ON_TIME).sum()),
        credit_timeline_TOTAL_OVERDUE_MONTHS=("ACCOUNT_STATUS", lambda x: x.isin(OVERDUE_ACCOUNT_STATUS_VALUES).sum()),
    ).reset_index()
    return credit_timeline_activity

def compute_credit_timeline_time_features(df_credit_timeline):
    """
    Compute time span statistics from credit_timeline.
    Returns:
      - Earliest record
      - Most recent record
      - Time range (duration between oldest and most recent records)
    """
    credit_timeline_time_features = df_credit_timeline.groupby("CREDIT_RECORD_ID").agg(
        credit_timeline_OLDEST_RECORD=("TIME_SINCE_ENTRY", "min"),
        credit_timeline_MOST_RECENT_RECORD=("TIME_SINCE_ENTRY", "max"),
        credit_timeline_CREDIT_HISTORY_LENGTH=("TIME_SINCE_ENTRY", lambda x: x.max() - x.min())
    ).reset_index()
    return credit_timeline_time_features

def compute_credit_timeline_overdue_features(df_credit_timeline):
    """
    Compute overdue metrics from ACCOUNT_STATUS column.
    Returns:
      - Count of months marked overdue (synthetic values like 'Delayed', 'Defaulted')
      - Maximum severity (highest label in the defined category order)
    """
    # Create an ordered synthetic category type
    df_tmp = df_credit_timeline.copy()
    df_tmp["ACCOUNT_STATUS"] = pd.Categorical(df_tmp["ACCOUNT_STATUS"], categories=ACCOUNT_STATUS_VALUES, ordered=True)

    credit_timeline_overdue_features = df_tmp.groupby("CREDIT_RECORD_ID").agg(
        credit_timeline_RECENT_OVERDUE_MONTHS=("ACCOUNT_STATUS", lambda x: x.isin(OVERDUE_ACCOUNT_STATUS_VALUES).sum()),
        credit_timeline_MAX_OVERDUE_STATUS=("ACCOUNT_STATUS", 
            lambda x: x[x.isin(OVERDUE_ACCOUNT_STATUS_VALUES)].max() if x.isin(OVERDUE_ACCOUNT_STATUS_VALUES).any() else ACCOUNT_STATUS_ON_TIME)
    ).reset_index()
    return credit_timeline_overdue_features

def compute_credit_timeline_categorical_features(df_credit_timeline):
    """
    Compute categorical frequency-based indicators.
    Returns:
      - % of months labeled 'Defaulted'
      - % of months labeled 'On Time'
      - Most common ACCOUNT_STATUS overall
    """
    credit_timeline_categorical_features = df_credit_timeline.groupby("CREDIT_RECORD_ID").agg(
        credit_timeline_PERCENT_CLOSED=("ACCOUNT_STATUS", lambda x: (x == ACCOUNT_STATUS_DEFAULTED).sum() / len(x) if len(x) > 0 else 0),
        credit_timeline_PERCENT_ACTIVE=("ACCOUNT_STATUS", lambda x: (x == ACCOUNT_STATUS_ON_TIME).sum() / len(x) if len(x) > 0 else 0),
        credit_timeline_MOST_COMMON_STATUS=("ACCOUNT_STATUS", lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    ).reset_index()
    return credit_timeline_categorical_features

def generate_credit_timeline_features(df_credit_timeline):
    """
    Generate full synthetic credit_timeline feature set.
    Computes individual feature groups and merges them on CREDIT_RECORD_ID.
    """
    logging.info("Starting credit timeline feature computation...")

    activity = compute_credit_timeline_activity(df_credit_timeline)
    time_features = compute_credit_timeline_time_features(df_credit_timeline)
    overdue_features = compute_credit_timeline_overdue_features(df_credit_timeline)
    categorical_features = compute_credit_timeline_categorical_features(df_credit_timeline)
    
    features_list = [activity, time_features, overdue_features, categorical_features]
    df_features = features_list[0]
    for features in features_list[1:]:
        df_features = df_features.merge(features, on="CREDIT_RECORD_ID", how="left")

    logging.info(f"✅ Credit timeline features generated. Shape: {df_features.shape}")
    
    return df_features
