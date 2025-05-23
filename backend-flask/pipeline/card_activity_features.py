import pandas as pd
import numpy as np
import logging
from tools.feature_config import RAW_SCHEMA, DTYPE_MAP, DEFAULT_CATS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Extract synthetic contract statuses
CONTRACT_STATUSES = DEFAULT_CATS["CONTRACT_STATUS"]
STATUS_ACTIVE = CONTRACT_STATUSES[0] 
STATUS_COMPLETED = CONTRACT_STATUSES[1]

def compute_card_activity_features(df_card_activity):
    """
    Compute credit activity features from the card activity dataset.
    Groups by CUSTOMER_ID and calculates:
      - card_activity_NUM_CREDIT_CARDS: count of PRIOR_LOAN_ID entries per customer
      - card_activity_NUM_ACTIVE_CARDS: where CONTRACT_STATUS == synthetic 'Status A'
      - card_activity_NUM_CLOSED_CARDS: where CONTRACT_STATUS == synthetic 'Status B'
    """
    activity = df_card_activity.groupby("CUSTOMER_ID").agg(
        card_activity_NUM_CREDIT_CARDS=("PRIOR_LOAN_ID", "count"),
        card_activity_NUM_ACTIVE_CARDS=("CONTRACT_STATUS", lambda x: (x == STATUS_ACTIVE).sum()),
        card_activity_NUM_CLOSED_CARDS=("CONTRACT_STATUS", lambda x: (x == STATUS_COMPLETED).sum())
    ).reset_index()
    return activity

def compute_card_loan_amount_features(df_card_activity):
    """
    Compute loan amount-related features from the card activity dataset.
    Groups by CUSTOMER_ID and calculates:
      - card_activity_TOTAL_CREDIT_LIMIT: total limit across cards
      - card_activity_TOTAL_CREDIT_BALANCE: total balance outstanding
      - card_activity_TOTAL_DEBT: sum of receivables
      - card_activity_CREDIT_UTILIZATION_RATIO: balance-to-limit ratio
    """
    loan_amounts = df_card_activity.groupby("CUSTOMER_ID").agg(
        card_activity_TOTAL_CREDIT_LIMIT=("CARD_CREDIT_LIMIT", "sum"),
        card_activity_TOTAL_CREDIT_BALANCE=("CURRENT_CARD_BALANCE", "sum"),
        card_activity_TOTAL_DEBT=("RECEIVABLE_BALANCE", "sum")
    ).reset_index()

    # Compute utilization ratio (avoid div by zero or inf)
    loan_amounts["card_activity_CREDIT_UTILIZATION_RATIO"] = (
        loan_amounts["card_activity_TOTAL_CREDIT_BALANCE"] / loan_amounts["card_activity_TOTAL_CREDIT_LIMIT"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return loan_amounts

def compute_card_time_features(df_card_activity):
    """
    Compute time-windowed averages from recent card activity history.
    Groups by CUSTOMER_ID and calculates:
      - avg balance/payment over last 6 and 12 months
    """
    time_features = df_card_activity.groupby("CUSTOMER_ID").agg(
        card_activity_AVG_BALANCE_LAST_6M=("CURRENT_CARD_BALANCE", lambda x: x.tail(6).mean()),
        card_activity_AVG_BALANCE_LAST_12M=("CURRENT_CARD_BALANCE", lambda x: x.tail(12).mean()),
        card_activity_AVG_PAYMENT_LAST_6M=("CURRENT_PAYMENT", lambda x: x.tail(6).mean()),
        card_activity_AVG_PAYMENT_LAST_12M=("CURRENT_PAYMENT", lambda x: x.tail(12).mean())
    ).reset_index()
    return time_features

def compute_card_overdue_features(df_card_activity):
    """
    Compute overdue metrics from the card activity dataset.
    Groups by CUSTOMER_ID and calculates:
      - Total overdue principal
      - Number of months in various DPD (days past due) buckets
    """
    overdue = df_card_activity.groupby("CUSTOMER_ID").agg(
        card_activity_TOTAL_OVERDUE=("PRINCIPAL_DUE", "sum"),
        card_activity_NUM_DPD_0_30=("DAYS_PAST_DUE", lambda x: ((x > 0) & (x <= 30)).sum()),
        card_activity_NUM_DPD_30_90=("DAYS_PAST_DUE", lambda x: ((x > 30) & (x <= 90)).sum()),
        card_activity_NUM_DPD_90_PLUS=("DAYS_PAST_DUE", lambda x: (x > 90).sum())
    ).reset_index()
    return overdue

def compute_card_categorical_features(df_card_activity):
    """
    Compute categorical ratios from CONTRACT_STATUS using synthetic values.
    Groups by CUSTOMER_ID and calculates:
      - Ratio of records with synthetic 'Status A' (active)
      - Ratio with synthetic 'Status B' (completed/closed)
    """
    categorical = df_card_activity.groupby("CUSTOMER_ID").agg(
        card_activity_STATUS_ACTIVE_RATIO=("CONTRACT_STATUS", lambda x: (x == STATUS_ACTIVE).sum() / len(x)),
        card_activity_STATUS_COMPLETED_RATIO=("CONTRACT_STATUS", lambda x: (x == STATUS_COMPLETED).sum() / len(x))
    ).reset_index()
    return categorical

def generate_card_activity_features(df_card_activity):
    """
    Generate all engineered features from the card_activity table.
    Includes:
      - Activity stats
      - Loan amount summaries
      - Time-windowed means
      - Overdue indicators
      - Categorical status ratios
    Merges all components into one final DataFrame by CUSTOMER_ID.
    """
    logging.info("Starting card activity feature computation...")

    # Compute each component
    activity_features = compute_card_activity_features(df_card_activity)
    loan_amount_features = compute_card_loan_amount_features(df_card_activity)
    time_features = compute_card_time_features(df_card_activity)
    overdue_features = compute_card_overdue_features(df_card_activity)
    categorical_features = compute_card_categorical_features(df_card_activity)

    # Merge all together
    features_list = [
        activity_features,
        loan_amount_features,
        time_features,
        overdue_features,
        categorical_features
    ]
    df_features = features_list[0]
    for df in features_list[1:]:
        df_features = df_features.merge(df, on="CUSTOMER_ID", how="left")

    logging.info(f"✅ Card activity features generated. Shape: {df_features.shape}")
    return df_features
