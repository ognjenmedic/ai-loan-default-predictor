import pandas as pd
import numpy as np
import logging
from tools.feature_config import RAW_SCHEMA, DTYPE_MAP, DEFAULT_CATS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Pull synthetic values from config
STATUS_APPROVED   = DEFAULT_CATS["CONTRACT_STATUS"][0]
STATUS_REFUSED    = DEFAULT_CATS["CONTRACT_STATUS"][1]
STATUS_CANCELED   = DEFAULT_CATS["CONTRACT_STATUS"][2]

CONTRACT_REVOLVING = DEFAULT_CATS["LOAN_CONTRACT_TYPE"][0]
CONTRACT_CASH      = DEFAULT_CATS["LOAN_CONTRACT_TYPE"][1]

CLIENT_REPEATER = DEFAULT_CATS["CLIENT_TYPE"][0]

# ————————————————————————————————————————
# Row-Level: Create DAYS_SINCE_DECISION_BIN
# ————————————————————————————————————————
def add_days_since_decision_bin(df):
    """
    Create DAYS_SINCE_DECISION_BIN: 20-bin numeric encoding of DAYS_SINCE_DECISION.
    Bins are integers from 0 to 19.
    """
    df = df.copy()
    df["DAYS_SINCE_DECISION_BIN"] = pd.cut(
        df["DAYS_SINCE_DECISION"],
        bins=20,
        labels=False
    )
    return df

def compute_days_since_decision_bin_aggregates(df):
    """
    Aggregate DAYS_SINCE_DECISION_BIN into mean, sum, max, min.
    """
    agg_df = (
        df.groupby("CUSTOMER_ID")["DAYS_SINCE_DECISION_BIN"]
          .agg(["mean", "sum", "max", "min"])
          .reset_index()
    )
    agg_df.rename(columns={
        "mean": "prior_loan_agg_DAYS_SINCE_DECISION_BIN_mean",
        "sum":  "prior_loan_agg_DAYS_SINCE_DECISION_BIN_sum",
        "max":  "prior_loan_agg_DAYS_SINCE_DECISION_BIN_max",
        "min":  "prior_loan_agg_DAYS_SINCE_DECISION_BIN_min"
    }, inplace=True)
    return agg_df

# ————————————————————————————————————————
# Aggregators
# ————————————————————————————————————————
def compute_prior_loan_credit_activity(df):
    return df.groupby("CUSTOMER_ID").agg(
        prior_loan_NUM_APPLICATIONS=("PRIOR_LOAN_ID", "count"),
        prior_loan_NUM_APPROVED_LOANS=("CONTRACT_STATUS", lambda x: (x == STATUS_APPROVED).sum()),
        prior_loan_NUM_REFUSED_LOANS=("CONTRACT_STATUS", lambda x: (x == STATUS_REFUSED).sum()),
        prior_loan_NUM_CANCELED_LOANS=("CONTRACT_STATUS", lambda x: (x == STATUS_CANCELED).sum()),
        prior_loan_APPROVAL_RATE=("CONTRACT_STATUS", lambda x: (x == STATUS_APPROVED).sum() / len(x)),
        prior_loan_NUM_REVOLVING_LOANS=("LOAN_CONTRACT_TYPE", lambda x: (x == CONTRACT_REVOLVING).sum()),
        prior_loan_NUM_CASH_LOANS=("LOAN_CONTRACT_TYPE", lambda x: (x == CONTRACT_CASH).sum()),
        prior_loan_NUM_REPEAT_LOANS=("CLIENT_TYPE", lambda x: (x == CLIENT_REPEATER).sum())
    ).reset_index()

def compute_prior_loan_amounts(df):
    amounts = df.groupby("CUSTOMER_ID").agg(
        prior_loan_TOTAL_APPLICATION_AMOUNT=("REQUESTED_LOAN_AMOUNT", "sum"),
        prior_loan_TOTAL_APPROVED_AMOUNT=("LOAN_AMOUNT", "sum"),
        prior_loan_AVG_APPLICATION_AMOUNT=("REQUESTED_LOAN_AMOUNT", "mean"),
        prior_loan_AVG_APPROVED_AMOUNT=("LOAN_AMOUNT", "mean"),
        prior_loan_MAX_APPLICATION_AMOUNT=("REQUESTED_LOAN_AMOUNT", "max"),
        prior_loan_MAX_APPROVED_AMOUNT=("LOAN_AMOUNT", "max"),
        prior_loan_MIN_APPLICATION_AMOUNT=("REQUESTED_LOAN_AMOUNT", "min"),
        prior_loan_MIN_APPROVED_AMOUNT=("LOAN_AMOUNT", "min"),
        prior_loan_STD_APPLICATION_AMOUNT=("REQUESTED_LOAN_AMOUNT", "std"),
        prior_loan_STD_APPROVED_AMOUNT=("LOAN_AMOUNT", "std"),
        prior_loan_TOTAL_PAYMENT_AMOUNT=("PLANNED_NUM_PAYMENTS", lambda x: x.fillna(0).sum())
    ).reset_index()

    amounts["prior_loan_APPROVAL_AMOUNT_RATIO"] = (
        amounts["prior_loan_TOTAL_APPROVED_AMOUNT"] /
        amounts["prior_loan_TOTAL_APPLICATION_AMOUNT"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    amounts["prior_loan_APPLICATION_TO_CREDIT_RATIO"] = (
        amounts["prior_loan_TOTAL_APPLICATION_AMOUNT"] /
        amounts["prior_loan_TOTAL_APPROVED_AMOUNT"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return amounts

def compute_prior_loan_time_features(df):
    return df.groupby("CUSTOMER_ID").agg(
        prior_loan_AVG_TIME_SINCE_APPLICATION=("DAYS_SINCE_DECISION", lambda x: abs(x.mean()) / 365),
        prior_loan_MAX_TIME_SINCE_APPLICATION=("DAYS_SINCE_DECISION", lambda x: abs(x.min()) / 365),
        prior_loan_TIME_SINCE_LAST_APPLICATION=("DAYS_SINCE_DECISION", lambda x: abs(x.max()) / 365),
        prior_loan_AVG_LOAN_DURATION=("DAYS_UNTIL_TERMINATION", lambda x: abs(x.mean()) / 365),
        prior_loan_MAX_LOAN_DURATION=("DAYS_UNTIL_TERMINATION", lambda x: abs(x.max()) / 365),
        prior_loan_AVG_TIME_TO_FIRST_PAYMENT=("DAYS_UNTIL_FIRST_PAYMENT", lambda x: abs(x.mean()) / 365),
        prior_loan_AVG_TIME_REMAINING=("DAYS_UNTIL_FINAL_PAYMENT", lambda x: abs(x.mean()) / 365)
    ).reset_index()

def compute_prior_loan_credit_overdue(df):
    return df.groupby("CUSTOMER_ID").agg(
        prior_loan_NUM_OVERDUE_APPLICATIONS=("DAYS_UNTIL_FIRST_PAYMENT", lambda x: (x < 0).sum()),
        prior_loan_TOTAL_OVERDUE_AMOUNT=("DAYS_UNTIL_FINAL_PAYMENT", lambda x: (x < 0).sum()),
        prior_loan_PROPORTION_OVERDUE_APPLICATIONS=("DAYS_UNTIL_FINAL_PAYMENT", lambda x: (x < 0).mean())
    ).reset_index()

def compute_prior_loan_categorical_features(df):
    return df.groupby("CUSTOMER_ID").agg(
        prior_loan_PERCENT_APPROVED=("CONTRACT_STATUS", lambda x: (x == STATUS_APPROVED).sum() / len(x)),
        prior_loan_MOST_COMMON_CONTRACT_TYPE=("LOAN_CONTRACT_TYPE", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        prior_loan_MOST_COMMON_LOAN_PURPOSE=("LOAN_PURPOSE_CATEGORY", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        prior_loan_HAS_REFUSALS=("CONTRACT_STATUS", lambda x: 1 if (x == STATUS_REFUSED).any() else 0)
    ).reset_index()

# ————————————————————————————————————————
# Orchestration
# ————————————————————————————————————————
def generate_prior_loan_history_features(df_prior_loan_history):
    """
    Generate synthetic-safe features from prior loan history data.
    """
    logging.info("Starting prior loan history feature computation...")

    df_temp = add_days_since_decision_bin(df_prior_loan_history)

    days_bin_agg       = compute_days_since_decision_bin_aggregates(df_temp)
    credit_activity    = compute_prior_loan_credit_activity(df_temp)
    loan_amounts       = compute_prior_loan_amounts(df_temp)
    time_features      = compute_prior_loan_time_features(df_temp)
    credit_overdue     = compute_prior_loan_credit_overdue(df_temp)
    categorical_feats  = compute_prior_loan_categorical_features(df_temp)

    features_list = [
        days_bin_agg,
        credit_activity,
        loan_amounts,
        time_features,
        credit_overdue,
        categorical_feats
    ]

    df_final = features_list[0]
    for feat_df in features_list[1:]:
        df_final = df_final.merge(feat_df, on="CUSTOMER_ID", how="left")

    logging.info(f"✅ Prior loan history features generated. Final shape: {df_final.shape}")
    return df_final
