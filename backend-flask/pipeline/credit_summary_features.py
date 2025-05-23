import pandas as pd
import numpy as np
from functools import reduce
import logging

from tools.feature_config import RAW_SCHEMA, DTYPE_MAP, DEFAULT_CATS

# Unpack synthetic values for CREDIT_STATUS and TYPE_OF_CREDIT
CREDIT_STATUS_VALUES = DEFAULT_CATS["CREDIT_STATUS"]
CREDIT_STATUS_ACTIVE, CREDIT_STATUS_REPAID, CREDIT_STATUS_WRITTEN_OFF = CREDIT_STATUS_VALUES

TYPE_OF_CREDIT_VALUES = DEFAULT_CATS["TYPE_OF_CREDIT"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def compute_credit_summary_activity(df_credit_summary):
    """
    Compute credit activity features from the credit_summary dataset.
    Groups by CUSTOMER_ID and calculates:
      - Total number of loans (count of CREDIT_RECORD_ID)
      - Number of active loans (where CREDIT_STATUS == synthetic "Active")
      - Number of closed loans (where CREDIT_STATUS == synthetic "Repaid")
    """
    credit_summary_activity = df_credit_summary.groupby("CUSTOMER_ID").agg(
        credit_summary_NUM_LOANS=("CREDIT_RECORD_ID", "count"),
        credit_summary_NUM_ACTIVE_LOANS=("CREDIT_STATUS", lambda x: (x == CREDIT_STATUS_ACTIVE).sum()),
        credit_summary_NUM_CLOSED_LOANS=("CREDIT_STATUS", lambda x: (x == CREDIT_STATUS_REPAID).sum())
    ).reset_index()
    return credit_summary_activity

def compute_credit_summary_loan_amounts(df_credit_summary):
    """
    Compute credit-related statistics from the credit_summary dataset.
    Groups by CUSTOMER_ID and calculates:
      - Total credit amount
      - Average credit amount per loan
      - Total outstanding debt
      - Debt-to-Credit Ratio (with safe division)
    """
    credit_summary_loan_amounts = df_credit_summary.groupby("CUSTOMER_ID").agg(
        credit_summary_TOTAL_CREDIT_AMOUNT=("TOTAL_CREDIT_AMOUNT", "sum"),
        credit_summary_AVG_CREDIT_AMOUNT=("TOTAL_CREDIT_AMOUNT", "mean"),
        credit_summary_TOTAL_DEBT=("OUTSTANDING_DEBT", "sum")
    ).reset_index()

    credit_summary_loan_amounts["credit_summary_DEBT_CREDIT_RATIO"] = (
        credit_summary_loan_amounts["credit_summary_TOTAL_DEBT"] / credit_summary_loan_amounts["credit_summary_TOTAL_CREDIT_AMOUNT"]
    )
    credit_summary_loan_amounts["credit_summary_DEBT_CREDIT_RATIO"] = (
        credit_summary_loan_amounts["credit_summary_DEBT_CREDIT_RATIO"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return credit_summary_loan_amounts

def compute_credit_summary_time_features(df_credit_summary):
    """
    Compute time-related statistics from the credit_summary dataset.
    Groups by CUSTOMER_ID and calculates:
      - Average loan age in years (absolute mean of DAYS_SINCE_CREDIT_ISSUED)
      - Time since last update (absolute max of DAYS_SINCE_CREDIT_UPDATE in years)
      - Average remaining credit time (absolute mean of DAYS_UNTIL_CREDIT_END in years)
    """
    credit_summary_time_features = df_credit_summary.groupby("CUSTOMER_ID").agg(
        credit_summary_AVG_CREDIT_AGE=("DAYS_SINCE_CREDIT_ISSUED", lambda x: abs(x.mean()) / 365),
        credit_summary_TIME_SINCE_LAST_UPDATE=("DAYS_SINCE_CREDIT_UPDATE", lambda x: abs(x.max()) / 365),
        credit_summary_AVG_REMAINING_CREDIT_TIME=("DAYS_UNTIL_CREDIT_END", lambda x: abs(x.mean()) / 365)
    ).reset_index()
    return credit_summary_time_features

def compute_credit_summary_overdue_features(df_credit_summary):
    """
    Compute overdue-related statistics from the credit_summary dataset.
    Groups by CUSTOMER_ID and calculates:
      - Number of overdue loans (count where DAYS_OVERDUE > 0)
      - Total overdue amount
    """
    credit_summary_overdue_features = df_credit_summary.groupby("CUSTOMER_ID").agg(
        credit_summary_NUM_OVERDUE_LOANS=("DAYS_OVERDUE", lambda x: (x > 0).sum()),
        credit_summary_TOTAL_OVERDUE_AMOUNT=("OVERDUE_BALANCE", "sum")
    ).reset_index()
    return credit_summary_overdue_features

def compute_credit_summary_categorical_features(df_credit_summary):
    """
    Compute categorical aggregations from the credit_summary dataset.
    Groups by CUSTOMER_ID and calculates:
      - Percentage of active loans (where CREDIT_STATUS == synthetic "Active")
      - Most common credit type (mode of TYPE_OF_CREDIT)
      - Flag for bad debt history (1 if any loan has CREDIT_STATUS == synthetic "Written-Off", else 0)
    """
    credit_summary_categorical_features = df_credit_summary.groupby("CUSTOMER_ID").agg(
        credit_summary_PERCENT_ACTIVE=("CREDIT_STATUS", lambda x: (x == CREDIT_STATUS_ACTIVE).sum() / len(x)),
        credit_summary_MOST_COMMON_TYPE_OF_CREDIT=("TYPE_OF_CREDIT", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        credit_summary_HAS_BAD_DEBT=("CREDIT_STATUS", lambda x: 1 if (x == CREDIT_STATUS_WRITTEN_OFF).any() else 0)
    ).reset_index()
    return credit_summary_categorical_features

def generate_credit_summary_features(df_credit_summary):
    """
    Given a raw credit_summary dataframe, compute and merge all engineered features.
    This function sequentially computes:
      - Credit activity features
      - Loan amount statistics
      - Time-related statistics
      - Overdue-related statistics
      - Categorical aggregations
    Then it merges these DataFrames on CUSTOMER_ID and returns the final feature DataFrame.
    """
    logging.info("Starting credit summary feature computation...")

    # Compute features from different aspects
    activity = compute_credit_summary_activity(df_credit_summary)
    loan_amounts = compute_credit_summary_loan_amounts(df_credit_summary)
    time_features = compute_credit_summary_time_features(df_credit_summary)
    overdue_features = compute_credit_summary_overdue_features(df_credit_summary)
    categorical_features = compute_credit_summary_categorical_features(df_credit_summary)

    # Merge all features sequentially on 'CUSTOMER_ID'
    features_list = [activity, loan_amounts, time_features, overdue_features, categorical_features]
    credit_summary_features = reduce(lambda left, right: pd.merge(left, right, on="CUSTOMER_ID", how="left"), features_list)

    logging.info(f"✅ Credit summary features generated. Final shape: {credit_summary_features.shape}")

    return credit_summary_features
