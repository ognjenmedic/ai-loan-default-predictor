import pandas as pd
import numpy as np
from functools import reduce
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def compute_bureau_credit_activity(df_bureau):
    """
    Compute credit activity features from the bureau dataset.
    Groups by SK_ID_CURR and calculates:
      - Total number of loans (count of SK_ID_BUREAU)
      - Number of active loans (where CREDIT_ACTIVE == "Active")
      - Number of closed loans (where CREDIT_ACTIVE == "Closed")
    """
    bureau_credit_activity = df_bureau.groupby("SK_ID_CURR").agg(
        bureau_NUM_LOANS=("SK_ID_BUREAU", "count"),
        bureau_NUM_ACTIVE_LOANS=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_NUM_CLOSED_LOANS=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum())
    ).reset_index()
    return bureau_credit_activity

def compute_bureau_loan_amounts(df_bureau):
    """
    Compute credit-related statistics from the bureau dataset.
    Groups by SK_ID_CURR and calculates:
      - Total credit amount
      - Average credit amount per loan
      - Total outstanding debt
      - Debt-to-Credit Ratio (with safe division)
    """
    bureau_loan_amounts = df_bureau.groupby("SK_ID_CURR").agg(
        bureau_TOTAL_CREDIT_AMOUNT=("AMT_CREDIT_SUM", "sum"),
        bureau_AVG_CREDIT_AMOUNT=("AMT_CREDIT_SUM", "mean"),
        bureau_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum")
    ).reset_index()

    bureau_loan_amounts["bureau_DEBT_CREDIT_RATIO"] = (
        bureau_loan_amounts["bureau_TOTAL_DEBT"] / bureau_loan_amounts["bureau_TOTAL_CREDIT_AMOUNT"]
    )
    bureau_loan_amounts["bureau_DEBT_CREDIT_RATIO"] = (
        bureau_loan_amounts["bureau_DEBT_CREDIT_RATIO"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    return bureau_loan_amounts

def compute_bureau_time_features(df_bureau):
    """
    Compute time-related statistics from the bureau dataset.
    Groups by SK_ID_CURR and calculates:
      - Average loan age in years (absolute mean of DAYS_CREDIT)
      - Time since last update (absolute max of DAYS_CREDIT_UPDATE in years)
      - Average remaining credit time (absolute mean of DAYS_CREDIT_ENDDATE in years)
    """
    bureau_time_features = df_bureau.groupby("SK_ID_CURR").agg(
        bureau_AVG_CREDIT_AGE=("DAYS_CREDIT", lambda x: abs(x.mean()) / 365),
        bureau_TIME_SINCE_LAST_UPDATE=("DAYS_CREDIT_UPDATE", lambda x: abs(x.max()) / 365),
        bureau_AVG_REMAINING_CREDIT_TIME=("DAYS_CREDIT_ENDDATE", lambda x: abs(x.mean()) / 365)
    ).reset_index()
    return bureau_time_features

def compute_bureau_overdue_features(df_bureau):
    """
    Compute overdue-related statistics from the bureau dataset.
    Groups by SK_ID_CURR and calculates:
      - Number of overdue loans (count where CREDIT_DAY_OVERDUE > 0)
      - Total overdue amount
    """
    bureau_overdue_features = df_bureau.groupby("SK_ID_CURR").agg(
        bureau_NUM_OVERDUE_LOANS=("CREDIT_DAY_OVERDUE", lambda x: (x > 0).sum()),
        bureau_TOTAL_OVERDUE_AMOUNT=("AMT_CREDIT_SUM_OVERDUE", "sum")
    ).reset_index()
    return bureau_overdue_features

def compute_bureau_categorical_features(df_bureau):
    """
    Compute categorical aggregations from the bureau dataset.
    Groups by SK_ID_CURR and calculates:
      - Percentage of active loans (where CREDIT_ACTIVE == "Active")
      - Most common credit type (mode of CREDIT_TYPE)
      - Flag for bad debt history (1 if any loan has CREDIT_ACTIVE == "Bad debt", else 0)
    """
    bureau_categorical_features = df_bureau.groupby("SK_ID_CURR").agg(
        bureau_PERCENT_ACTIVE=("CREDIT_ACTIVE", lambda x: (x == "Active").sum() / len(x)),
        bureau_MOST_COMMON_CREDIT_TYPE=("CREDIT_TYPE", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        bureau_HAS_BAD_DEBT=("CREDIT_ACTIVE", lambda x: 1 if (x == "Bad debt").any() else 0)
    ).reset_index()
    return bureau_categorical_features

def generate_bureau_features(df_bureau):
    """
    Given a raw bureau dataframe, compute and merge all engineered features.
    This function sequentially computes:
      - Credit activity features
      - Loan amount statistics
      - Time-related statistics
      - Overdue-related statistics
      - Categorical aggregations
    Then it merges these DataFrames on SK_ID_CURR and returns the final feature DataFrame.
    """
    logging.info("Starting bureau feature computation...")
    
    # Compute features from different aspects
    credit_activity = compute_bureau_credit_activity(df_bureau)
    loan_amounts = compute_bureau_loan_amounts(df_bureau)
    time_features = compute_bureau_time_features(df_bureau)
    overdue_features = compute_bureau_overdue_features(df_bureau)
    categorical_features = compute_bureau_categorical_features(df_bureau)
    
    # Merge all features sequentially on 'SK_ID_CURR'
    features_list = [credit_activity, loan_amounts, time_features, overdue_features, categorical_features]
    bureau_features = reduce(lambda left, right: pd.merge(left, right, on="SK_ID_CURR", how="left"), features_list)

    logging.info(f"✅ Bureau features generated. Final shape: {bureau_features.shape}")
    
    return bureau_features
