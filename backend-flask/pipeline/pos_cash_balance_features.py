import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Row-Level Feature Creation
def add_loan_age_group(df):
    """
    Row-level transformation:
    Segment loans into old vs. recent via:
    bins=[-100, -48, -24, 0]
    labels=["Old (>4 years)", "Mid (2-4 years)", "Recent (<2 years)"]
    """
    df = df.copy()
    df["LOAN_AGE_GROUP"] = pd.cut(
        df["MONTHS_BALANCE"],
        bins=[-100, -48, -24, 0], 
        labels=["Old (>4 years)", "Mid (2-4 years)", "Recent (<2 years)"]
    )
    return df

def compute_loan_age_group_mode(df):
    """
    Aggregates LOAN_AGE_GROUP by SK_ID_CURR,
    creating 'pos_cash_agg_LOAN_AGE_GROUP_most_frequent'.
    """
    df_grp = df.groupby("SK_ID_CURR")["LOAN_AGE_GROUP"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else "Unknown"
    ).reset_index()
    df_grp.rename(columns={"LOAN_AGE_GROUP": "pos_cash_agg_LOAN_AGE_GROUP_most_frequent"}, inplace=True)
    return df_grp

def compute_pos_cash_credit_activity(df):
    """
    Aggregates loan activity features based on SK_ID_CURR.
    It calculates:
      - pos_cash_balance_NUM_POS_LOANS: Total number of POS loans per client (unique SK_ID_PREV values).
      - pos_cash_balance_NUM_ACTIVE_POS_LOANS: Number of active POS loans (where NAME_CONTRACT_STATUS == "Active").
      - pos_cash_balance_NUM_COMPLETED_POS_LOANS: Number of completed POS loans (where NAME_CONTRACT_STATUS == "Completed").
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        pos_cash_balance_NUM_POS_LOANS=("SK_ID_PREV", "nunique"),
        pos_cash_balance_NUM_ACTIVE_POS_LOANS=("NAME_CONTRACT_STATUS", lambda x: (x == "Active").sum()),
        pos_cash_balance_NUM_COMPLETED_POS_LOANS=("NAME_CONTRACT_STATUS", lambda x: (x == "Completed").sum())
    ).reset_index()
    return df_grp

def compute_pos_cash_time_features(df):
    """
    Calculates the duration and recency of POS credit activity per client (SK_ID_CURR):
      - pos_cash_balance_AVG_POS_CREDIT_DURATION: Average loan age in years (absolute mean of MONTHS_BALANCE / 12).
      - pos_cash_balance_TIME_SINCE_LAST_POS_UPDATE: Time in years since the most recent update (max MONTHS_BALANCE / 12).
      - pos_cash_balance_AVG_REMAINING_INSTALLMENT_TIME: Average number of remaining installments (CNT_INSTALMENT_FUTURE mean).
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        pos_cash_balance_AVG_POS_CREDIT_DURATION=("MONTHS_BALANCE", lambda x: abs(x.mean()) / 12),
        pos_cash_balance_TIME_SINCE_LAST_POS_UPDATE=("MONTHS_BALANCE", lambda x: abs(x.max()) / 12),
        pos_cash_balance_AVG_REMAINING_INSTALLMENT_TIME=("CNT_INSTALMENT_FUTURE", lambda x: abs(x.mean()))
    ).reset_index()
    return df_grp

def compute_pos_cash_overdue_features(df):
    """
    Calculates overdue statistics for each client (SK_ID_CURR):
      - pos_cash_balance_NUM_OVERDUE_POS_CREDITS: Number of loans with overdue days (SK_DPD > 0).
      - pos_cash_balance_TOTAL_OVERDUE_DAYS_POS: Total overdue days across all POS loans (sum of SK_DPD).
      - pos_cash_balance_MAX_OVERDUE_DAYS_POS: Maximum overdue days for a single loan (max of SK_DPD).
      - pos_cash_balance_NUM_SEVERE_OVERDUE_POS: Number of loans with severe overdue status (SK_DPD_DEF > 0).
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        pos_cash_balance_NUM_OVERDUE_POS_CREDITS=("SK_DPD", lambda x: (x > 0).sum()),
        pos_cash_balance_TOTAL_OVERDUE_DAYS_POS=("SK_DPD", "sum"),
        pos_cash_balance_MAX_OVERDUE_DAYS_POS=("SK_DPD", "max"),
        pos_cash_balance_NUM_SEVERE_OVERDUE_POS=("SK_DPD_DEF", lambda x: (x > 0).sum())
    ).reset_index()
    return df_grp

def compute_pos_cash_balance_categorical_features(df):
    """
    Calculates categorical features related to loan activity status:
      - pos_cash_balance_PERCENT_ACTIVE_MONTHS: Percentage of records where contract status is "Active".
      - pos_cash_balance_MOST_COMMON_CONTRACT_STATUS: Most frequent contract status per client.
      - pos_cash_balance_HAS_RETURNED_CONTRACT: Flag indicating if the client ever had a contract returned to the store (1 if "Returned to the store" appears, otherwise 0).
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        pos_cash_balance_PERCENT_ACTIVE_MONTHS=("NAME_CONTRACT_STATUS", lambda x: (x == "Active").sum() / len(x)),
        pos_cash_balance_MOST_COMMON_CONTRACT_STATUS=("NAME_CONTRACT_STATUS", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        pos_cash_balance_HAS_RETURNED_CONTRACT=("NAME_CONTRACT_STATUS", lambda x: 1 if (x == "Returned to the store").any() else 0)
    ).reset_index()
    return df_grp

def generate_pos_cash_balance_features(df_pos_cash_balance):
    """
    1) Add LOAN_AGE_GROUP row-level
    2) Compute aggregator for LOAN_AGE_GROUP => pos_cash_agg_LOAN_AGE_GROUP_most_frequent
    3) Compute aggregator sets => credit_activity, time_features, overdue_features, categorical_features
    4) Merge all => final DataFrame with SK_ID_CURR + all columns
    """

    logging.info("Generating POS Cash Balance features...")
    
    # 1) Row-level transform
    df_transformed = add_loan_age_group(df_pos_cash_balance)
    
    # 2) LOAN_AGE_GROUP aggregator
    loan_age_mode_df = compute_loan_age_group_mode(df_transformed)
    
    # 3) Other aggregator sets
    credit_activity = compute_pos_cash_credit_activity(df_transformed)
    time_features = compute_pos_cash_time_features(df_transformed)
    overdue_features = compute_pos_cash_overdue_features(df_transformed)
    categorical_features = compute_pos_cash_balance_categorical_features(df_transformed)
    
    # 4) Merge them all
    features_list = [
        loan_age_mode_df,
        credit_activity,
        time_features,
        overdue_features,
        categorical_features
    ]
    df_final = features_list[0]
    for feat_df in features_list[1:]:
        df_final = df_final.merge(feat_df, on="SK_ID_CURR", how="left")

    logging.info(f"✅ POS Cash Balance feature generation complete. Final shape: {df_final.shape}")

    return df_final
