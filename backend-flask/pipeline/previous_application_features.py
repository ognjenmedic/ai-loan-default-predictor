import pandas as pd
import numpy as np

###############################################################################
# 1) Row-Level: Creating DAYS_DECISION_BIN
###############################################################################
def add_days_decision_bin(df):
    """
    Replicates EDA step of binning DAYS_DECISION into 20 discrete bins.
    Using labels=False means the bins are encoded as integer labels [0..19].
    """
    df = df.copy()
    df["DAYS_DECISION_BIN"] = pd.cut(
        df["DAYS_DECISION"],
        bins=20,  
        labels=False  
    )
    return df

###############################################################################
# 2) Aggregator for DAYS_DECISION_BIN => mean, sum, max, min
###############################################################################
def compute_days_decision_bin_aggregates(df):
    """
    Aggregates DAYS_DECISION_BIN by SK_ID_CURR to create:
      - previous_app_agg_DAYS_DECISION_BIN_mean
      - previous_app_agg_DAYS_DECISION_BIN_sum
      - previous_app_agg_DAYS_DECISION_BIN_max
      - previous_app_agg_DAYS_DECISION_BIN_min
    """
    agg_df = (
        df.groupby("SK_ID_CURR")["DAYS_DECISION_BIN"]
          .agg(["mean", "sum", "max", "min"])
          .reset_index()
    )
    agg_df.rename(
        columns={
            "mean": "previous_app_agg_DAYS_DECISION_BIN_mean",
            "sum":  "previous_app_agg_DAYS_DECISION_BIN_sum",
            "max":  "previous_app_agg_DAYS_DECISION_BIN_max",
            "min":  "previous_app_agg_DAYS_DECISION_BIN_min"
        },
        inplace=True
    )
    return agg_df

###############################################################################
# 3) Existing Aggregator Functions
###############################################################################
def compute_previous_application_credit_activity(df):
    """
    Groups by SK_ID_CURR and calculates credit activity features
    (NUM_APPLICATIONS, NUM_APPROVED_LOANS, APPROVAL_RATE, etc.).
    """
    activity = df.groupby("SK_ID_CURR").agg(
        previous_app_NUM_APPLICATIONS=("SK_ID_PREV", "count"),
        previous_app_NUM_APPROVED_LOANS=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        previous_app_NUM_REFUSED_LOANS=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        previous_app_NUM_CANCELED_LOANS=("NAME_CONTRACT_STATUS", lambda x: (x == "Canceled").sum()),
        previous_app_NUM_UNUSED_LOANS=("NAME_CONTRACT_STATUS", lambda x: (x == "Unused offer").sum()),
        previous_app_APPROVAL_RATE=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum() / len(x)),
        previous_app_NUM_REVOLVING_LOANS=("NAME_CONTRACT_TYPE", lambda x: (x == "Revolving loans").sum()),
        previous_app_NUM_CASH_LOANS=("NAME_CONTRACT_TYPE", lambda x: (x == "Cash loans").sum()),
        previous_app_NUM_CONSUMER_LOANS=("NAME_CONTRACT_TYPE", lambda x: (x == "Consumer loans").sum()),
        previous_app_NUM_REPEAT_LOANS=("NAME_CLIENT_TYPE", lambda x: (x == "Repeater").sum())
    ).reset_index()
    return activity

def compute_previous_application_loan_amounts(df):
    """
    Groups by SK_ID_CURR and calculates loan amount stats:
      - sums, means, max, min, std of AMT_APPLICATION, AMT_CREDIT
      - previous_app_APPROVAL_AMOUNT_RATIO
      - previous_app_APPLICATION_TO_CREDIT_RATIO
    """
    amounts = df.groupby("SK_ID_CURR").agg(
        previous_app_TOTAL_APPLICATION_AMOUNT=("AMT_APPLICATION", "sum"),
        previous_app_TOTAL_APPROVED_AMOUNT=("AMT_CREDIT", "sum"),
        previous_app_AVG_APPLICATION_AMOUNT=("AMT_APPLICATION", "mean"),
        previous_app_AVG_APPROVED_AMOUNT=("AMT_CREDIT", "mean"),
        previous_app_MAX_APPLICATION_AMOUNT=("AMT_APPLICATION", "max"),
        previous_app_MAX_APPROVED_AMOUNT=("AMT_CREDIT", "max"),
        previous_app_MIN_APPLICATION_AMOUNT=("AMT_APPLICATION", "min"),
        previous_app_MIN_APPROVED_AMOUNT=("AMT_CREDIT", "min"),
        previous_app_STD_APPLICATION_AMOUNT=("AMT_APPLICATION", "std"),
        previous_app_STD_APPROVED_AMOUNT=("AMT_CREDIT", "std"),
        previous_app_TOTAL_PAYMENT_AMOUNT=("CNT_PAYMENT", lambda x: x.fillna(0).sum())
    ).reset_index()
    
    # Ratios
    amounts["previous_app_APPROVAL_AMOUNT_RATIO"] = (
        amounts["previous_app_TOTAL_APPROVED_AMOUNT"] / amounts["previous_app_TOTAL_APPLICATION_AMOUNT"]
    )
    amounts["previous_app_APPLICATION_TO_CREDIT_RATIO"] = (
        amounts["previous_app_TOTAL_APPLICATION_AMOUNT"] / amounts["previous_app_TOTAL_APPROVED_AMOUNT"]
    )
    # Handle inf/NaN
    amounts["previous_app_APPROVAL_AMOUNT_RATIO"] = (
        amounts["previous_app_APPROVAL_AMOUNT_RATIO"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    amounts["previous_app_APPLICATION_TO_CREDIT_RATIO"] = (
        amounts["previous_app_APPLICATION_TO_CREDIT_RATIO"].replace([np.inf, -np.inf], np.nan).fillna(0)
    )
    return amounts

def compute_previous_application_time_features(df):
    """
    Time-related features:
      - previous_app_AVG_TIME_SINCE_APPLICATION, ...
      - previous_app_TIME_SINCE_LAST_APPLICATION, ...
      - previous_app_AVG_LOAN_DURATION, ...
    """
    time_feats = df.groupby("SK_ID_CURR").agg(
        previous_app_AVG_TIME_SINCE_APPLICATION=("DAYS_DECISION", lambda x: abs(x.mean()) / 365),
        previous_app_MAX_TIME_SINCE_APPLICATION=("DAYS_DECISION", lambda x: abs(x.min()) / 365),
        previous_app_TIME_SINCE_LAST_APPLICATION=("DAYS_DECISION", lambda x: abs(x.max()) / 365),
        previous_app_AVG_LOAN_DURATION=("DAYS_TERMINATION", lambda x: abs(x.mean()) / 365),
        previous_app_MAX_LOAN_DURATION=("DAYS_TERMINATION", lambda x: abs(x.max()) / 365),
        previous_app_AVG_TIME_TO_FIRST_PAYMENT=("DAYS_FIRST_DUE", lambda x: abs(x.mean()) / 365),
        previous_app_AVG_TIME_REMAINING=("DAYS_LAST_DUE", lambda x: abs(x.mean()) / 365)
    ).reset_index()
    return time_feats

def compute_previous_application_credit_overdue(df):
    """
    Overdue-related features:
      - previous_app_NUM_OVERDUE_APPLICATIONS
      - previous_app_TOTAL_OVERDUE_AMOUNT
      - previous_app_PROPORTION_OVERDUE_APPLICATIONS
    """
    overdue = df.groupby("SK_ID_CURR").agg(
        previous_app_NUM_OVERDUE_APPLICATIONS=("DAYS_FIRST_DUE", lambda x: (x < 0).sum()),
        previous_app_TOTAL_OVERDUE_AMOUNT=("DAYS_LAST_DUE", lambda x: (x < 0).sum()),
        previous_app_PROPORTION_OVERDUE_APPLICATIONS=("DAYS_LAST_DUE", lambda x: (x < 0).mean())
    ).reset_index()
    return overdue

def compute_previous_application_categorical_features(df):
    """
    Categorical aggregations:
      - previous_app_PERCENT_APPROVED
      - previous_app_MOST_COMMON_CONTRACT_TYPE
      - previous_app_MOST_COMMON_LOAN_PURPOSE
      - previous_app_HAS_REFUSALS
    """
    cat_feats = df.groupby("SK_ID_CURR").agg(
        previous_app_PERCENT_APPROVED=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum() / len(x)),
        previous_app_MOST_COMMON_CONTRACT_TYPE=("NAME_CONTRACT_TYPE", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        previous_app_MOST_COMMON_LOAN_PURPOSE=("NAME_CASH_LOAN_PURPOSE", lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
        previous_app_HAS_REFUSALS=("NAME_CONTRACT_STATUS", lambda x: 1 if (x == "Refused").any() else 0)
    ).reset_index()
    return cat_feats

###############################################################################
# 4) Master Function => 'generate_previous_application_features'
###############################################################################
def generate_previous_application_features(df_previous_application):
    """
    1) Add DAYS_DECISION_BIN (row-level).
    2) Aggregate DAYS_DECISION_BIN => previous_app_agg_DAYS_DECISION_BIN_{mean,sum,max,min}.
    3) Compute existing aggregator sets:
       - credit_activity
       - loan_amounts
       - time_features
       - credit_overdue
       - categorical_features
    4) Merge all on SK_ID_CURR => final DataFrame
    """
    # (1) Create the DAYS_DECISION_BIN column
    df_temp = add_days_decision_bin(df_previous_application)

    # (2) Aggregate DAYS_DECISION_BIN 
    days_bin_agg = compute_days_decision_bin_aggregates(df_temp)

    # (3) Existing aggregator sets
    credit_activity = compute_previous_application_credit_activity(df_temp)
    loan_amounts = compute_previous_application_loan_amounts(df_temp)
    time_features = compute_previous_application_time_features(df_temp)
    credit_overdue = compute_previous_application_credit_overdue(df_temp)
    categorical_features = compute_previous_application_categorical_features(df_temp)
    
    # (4) Merge everything
    features_list = [
        days_bin_agg,
        credit_activity,
        loan_amounts,
        time_features,
        credit_overdue,
        categorical_features
    ]
    
    df_final = features_list[0]
    for feat in features_list[1:]:
        df_final = df_final.merge(feat, on="SK_ID_CURR", how="left")
    
    return df_final
