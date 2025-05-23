import pandas as pd
import numpy as np
import logging
from tools.feature_config import RAW_SCHEMA, DTYPE_MAP, DEFAULT_CATS

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Synthetic label references
CONTRACT_STATUSES = DEFAULT_CATS["CONTRACT_STATUS"]
STATUS_ACTIVE = CONTRACT_STATUSES[0]     
STATUS_COMPLETED = CONTRACT_STATUSES[1]  
STATUS_RETURNED = CONTRACT_STATUSES[2]  

LOAN_AGE_GROUP_LABELS = ["Loan Tier A", "Loan Tier B", "Loan Tier C"]

# Row-Level Feature Creation
def add_loan_age_group(df):
    """
    Segment loans into synthetic age tiers based on TIME_SINCE_ENTRY:
      - Loan Tier A: oldest loans (e.g., >4 years)
      - Loan Tier B: medium age
      - Loan Tier C: recent loans
    """
    df = df.copy()
    df["LOAN_AGE_GROUP"] = pd.cut(
        df["TIME_SINCE_ENTRY"],
        bins=[-100, -48, -24, 0], 
        labels=LOAN_AGE_GROUP_LABELS
    )
    return df

def compute_loan_age_group_mode(df):
    """
    Compute most common LOAN_AGE_GROUP per CUSTOMER_ID.
    """
    df_grp = df.groupby("CUSTOMER_ID")["LOAN_AGE_GROUP"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    ).reset_index()
    df_grp.rename(columns={"LOAN_AGE_GROUP": "cash_pos_agg_LOAN_AGE_GROUP_most_frequent"}, inplace=True)
    return df_grp

def compute_cash_pos_credit_activity(df):
    """
    Aggregates synthetic loan activity:
      - Count of POS loans
      - Count of active and completed loans by synthetic category
    """
    df_grp = df.groupby("CUSTOMER_ID").agg(
        cash_pos_NUM_POS_LOANS=("PRIOR_LOAN_ID", "nunique"),
        cash_pos_NUM_ACTIVE_POS_LOANS=("CONTRACT_STATUS", lambda x: (x == STATUS_ACTIVE).sum()),
        cash_pos_NUM_COMPLETED_POS_LOANS=("CONTRACT_STATUS", lambda x: (x == STATUS_COMPLETED).sum())
    ).reset_index()
    return df_grp

def compute_cash_pos_time_features(df):
    """
    Time-related loan summaries:
      - Average loan age
      - Time since last POS update
      - Avg remaining installment duration
    """
    df_grp = df.groupby("CUSTOMER_ID").agg(
        cash_pos_AVG_POS_CREDIT_DURATION=("TIME_SINCE_ENTRY", lambda x: abs(x.mean()) / 12),
        cash_pos_TIME_SINCE_LAST_POS_UPDATE=("TIME_SINCE_ENTRY", lambda x: abs(x.max()) / 12),
        cash_pos_AVG_REMAINING_INSTALLMENT_TIME=("REMAINING_TERM_MONTHS", lambda x: abs(x.mean()))
    ).reset_index()
    return df_grp

def compute_cash_pos_overdue_features(df):
    """
    Overdue POS loan features:
      - Count and severity of overdue records (DAYS_PAST_DUE, DAYS_PAST_DUE_90)
    """
    df_grp = df.groupby("CUSTOMER_ID").agg(
        cash_pos_NUM_OVERDUE_POS_CREDITS=("DAYS_PAST_DUE", lambda x: (x > 0).sum()),
        cash_pos_TOTAL_OVERDUE_DAYS_POS=("DAYS_PAST_DUE", "sum"),
        cash_pos_MAX_OVERDUE_DAYS_POS=("DAYS_PAST_DUE", "max"),
        cash_pos_NUM_SEVERE_OVERDUE_POS=("DAYS_PAST_DUE_90", lambda x: (x > 0).sum())
    ).reset_index()
    return df_grp

def compute_cash_pos_categorical_features(df):
    """
    Categorical indicators from CONTRACT_STATUS:
      - % of time contracts are 'Active'
      - Most common status overall
      - Binary flag if ever marked 'Returned'
    """
    df_grp = df.groupby("CUSTOMER_ID").agg(
        cash_pos_PERCENT_ACTIVE_MONTHS=("CONTRACT_STATUS", lambda x: (x == STATUS_ACTIVE).sum() / len(x)),
        cash_pos_MOST_COMMON_CONTRACT_STATUS=("CONTRACT_STATUS", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        cash_pos_HAS_RETURNED_CONTRACT=("CONTRACT_STATUS", lambda x: 1 if (x == STATUS_RETURNED).any() else 0)
    ).reset_index()
    return df_grp

def generate_cash_pos_records_features(df_cash_pos_records):
    """
    Complete cash_pos_records feature engineering:
      1. Tag each loan with a synthetic age group.
      2. Compute aggregations for loan age, activity, timing, overdue, and status.
      3. Merge all into a unified feature set per CUSTOMER_ID.
    """
    logging.info("Generating cash_pos_records features...")

    # Row-level transform
    df_transformed = add_loan_age_group(df_cash_pos_records)

    # Aggregators
    loan_age_mode_df = compute_loan_age_group_mode(df_transformed)
    credit_activity = compute_cash_pos_credit_activity(df_transformed)
    time_features = compute_cash_pos_time_features(df_transformed)
    overdue_features = compute_cash_pos_overdue_features(df_transformed)
    categorical_features = compute_cash_pos_categorical_features(df_transformed)

    # Merge all
    features_list = [
        loan_age_mode_df,
        credit_activity,
        time_features,
        overdue_features,
        categorical_features
    ]
    df_final = features_list[0]
    for feat_df in features_list[1:]:
        df_final = df_final.merge(feat_df, on="CUSTOMER_ID", how="left")

    logging.info(f"✅ Cash POS Records feature generation complete. Final shape: {df_final.shape}")
    return df_final
