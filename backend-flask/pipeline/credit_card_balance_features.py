import pandas as pd
import numpy as np

def compute_credit_card_activity_features(df_credit_card_balance):
    """
    Compute credit activity features from the credit card balance dataset.
    Groups by SK_ID_CURR and calculates:
      - credit_card_NUM_CREDIT_CARDS: total number of credit cards per client (count of SK_ID_PREV)
      - credit_card_NUM_ACTIVE_CARDS: count of active credit cards (where NAME_CONTRACT_STATUS == "Active")
      - credit_card_NUM_CLOSED_CARDS: count of closed credit cards (where NAME_CONTRACT_STATUS == "Completed")
    """
    activity = df_credit_card_balance.groupby("SK_ID_CURR").agg(
        credit_card_NUM_CREDIT_CARDS=("SK_ID_PREV", "count"),
        credit_card_NUM_ACTIVE_CARDS=("NAME_CONTRACT_STATUS", lambda x: (x == "Active").sum()),
        credit_card_NUM_CLOSED_CARDS=("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").sum())
    ).reset_index()
    return activity

def compute_credit_card_loan_amount_features(df_credit_card_balance):
    """
    Compute loan amount features from the credit card balance dataset.
    Groups by SK_ID_CURR and calculates:
      - credit_card_TOTAL_CREDIT_LIMIT: sum of AMT_CREDIT_LIMIT_ACTUAL
      - credit_card_TOTAL_CREDIT_BALANCE: sum of AMT_BALANCE
      - credit_card_TOTAL_DEBT: sum of AMT_RECIVABLE
    Then computes the credit utilization ratio.
    """
    loan_amounts = df_credit_card_balance.groupby("SK_ID_CURR").agg(
        credit_card_TOTAL_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "sum"),
        credit_card_TOTAL_CREDIT_BALANCE=("AMT_BALANCE", "sum"),
        credit_card_TOTAL_DEBT=("AMT_RECIVABLE", "sum")
    ).reset_index()
    
    # Compute credit utilization ratio (avoid division by zero)
    loan_amounts["credit_card_CREDIT_UTILIZATION_RATIO"] = (
        loan_amounts["credit_card_TOTAL_CREDIT_BALANCE"] / loan_amounts["credit_card_TOTAL_CREDIT_LIMIT"]
    )
    loan_amounts["credit_card_CREDIT_UTILIZATION_RATIO"] = (
        loan_amounts["credit_card_CREDIT_UTILIZATION_RATIO"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return loan_amounts

def compute_credit_card_time_features(df_credit_card_balance):
    """
    Compute time-based features from the credit card balance dataset.
    Groups by SK_ID_CURR and calculates:
      - credit_card_AVG_BALANCE_LAST_6M: average AMT_BALANCE over the last 6 months
      - credit_card_AVG_BALANCE_LAST_12M: average AMT_BALANCE over the last 12 months
      - credit_card_AVG_PAYMENT_LAST_6M: average AMT_PAYMENT_CURRENT over the last 6 months
      - credit_card_AVG_PAYMENT_LAST_12M: average AMT_PAYMENT_CURRENT over the last 12 months
    """
    time_features = df_credit_card_balance.groupby("SK_ID_CURR").agg(
        credit_card_AVG_BALANCE_LAST_6M=("AMT_BALANCE", lambda x: x.tail(6).mean()),
        credit_card_AVG_BALANCE_LAST_12M=("AMT_BALANCE", lambda x: x.tail(12).mean()),
        credit_card_AVG_PAYMENT_LAST_6M=("AMT_PAYMENT_CURRENT", lambda x: x.tail(6).mean()),
        credit_card_AVG_PAYMENT_LAST_12M=("AMT_PAYMENT_CURRENT", lambda x: x.tail(12).mean())
    ).reset_index()
    return time_features

def compute_credit_card_overdue_features(df_credit_card_balance):
    """
    Compute overdue-related features from the credit card balance dataset.
    Groups by SK_ID_CURR and calculates:
      - credit_card_TOTAL_OVERDUE: total overdue principal amount (sum of AMT_RECEIVABLE_PRINCIPAL)
      - credit_card_NUM_DPD_0_30: count of SK_DPD between 0 and 30
      - credit_card_NUM_DPD_30_90: count of SK_DPD between 30 and 90
      - credit_card_NUM_DPD_90_PLUS: count of SK_DPD greater than 90
    """
    overdue = df_credit_card_balance.groupby("SK_ID_CURR").agg(
        credit_card_TOTAL_OVERDUE=("AMT_RECEIVABLE_PRINCIPAL", "sum"),
        credit_card_NUM_DPD_0_30=("SK_DPD", lambda x: ((x > 0) & (x <= 30)).sum()),
        credit_card_NUM_DPD_30_90=("SK_DPD", lambda x: ((x > 30) & (x <= 90)).sum()),
        credit_card_NUM_DPD_90_PLUS=("SK_DPD", lambda x: (x > 90).sum())
    ).reset_index()
    return overdue

def compute_credit_card_categorical_features(df_credit_card_balance):
    """
    Compute categorical aggregation features from the credit card balance dataset.
    Groups by SK_ID_CURR and calculates:
      - credit_card_STATUS_ACTIVE_RATIO: percentage of records where NAME_CONTRACT_STATUS is "Active"
      - credit_card_STATUS_COMPLETED_RATIO: percentage of records where NAME_CONTRACT_STATUS is "Completed"
    """
    categorical = df_credit_card_balance.groupby("SK_ID_CURR").agg(
        credit_card_STATUS_ACTIVE_RATIO=("NAME_CONTRACT_STATUS", lambda x: (x == "Active").sum() / len(x)),
        credit_card_STATUS_COMPLETED_RATIO=("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").sum() / len(x))
    ).reset_index()
    return categorical

def generate_credit_card_balance_features(df_credit_card_balance):
    """
    Given a raw credit card balance DataFrame, compute and merge all engineered features.
    This function sequentially computes:
      - Credit activity features
      - Loan amount features (with credit utilization ratio)
      - Time-based features
      - Overdue features
      - Categorical aggregation features
    Then it merges these DataFrames on SK_ID_CURR and returns the final feature DataFrame.
    """
    # Compute individual feature sets
    activity_features = compute_credit_card_activity_features(df_credit_card_balance)
    loan_amount_features = compute_credit_card_loan_amount_features(df_credit_card_balance)
    time_features = compute_credit_card_time_features(df_credit_card_balance)
    overdue_features = compute_credit_card_overdue_features(df_credit_card_balance)
    categorical_features = compute_credit_card_categorical_features(df_credit_card_balance)
    
    # Merge all feature DataFrames sequentially on SK_ID_CURR
    features_list = [
        activity_features,
        loan_amount_features,
        time_features,
        overdue_features,
        categorical_features
    ]
    df_features = features_list[0]
    for df in features_list[1:]:
        df_features = df_features.merge(df, on="SK_ID_CURR", how="left")
    
    return df_features
