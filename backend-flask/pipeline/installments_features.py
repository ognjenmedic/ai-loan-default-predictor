import pandas as pd
import numpy as np

###############################################################################
# 1) Row-Level Feature Creation
###############################################################################
def create_row_level_features(df):
    """
    Replicate row-level transformations from EDA notebook:
      - PAYMENT_RATIO: AMT_PAYMENT / AMT_INSTALMENT (safe division)
      - MISSED_PAYMENT: 1 if AMT_PAYMENT == 0
      - LATE_PAYMENT: 1 if DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
      - PAYMENT_DELAY: DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
    """
    df = df.copy()
    
    # Payment Ratio
    df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)
    
    # Missed Payment if AMT_PAYMENT == 0
    df["MISSED_PAYMENT"] = (df["AMT_PAYMENT"] == 0).astype(int)
    
    # Late Payment if DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT
    df["LATE_PAYMENT"] = (df["DAYS_ENTRY_PAYMENT"] > df["DAYS_INSTALMENT"]).astype(int)
    
    # Payment Delay
    df["PAYMENT_DELAY"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    
    return df

###############################################################################
# 2) Numeric Aggregator => "installments_agg_*"
###############################################################################
def compute_numeric_aggregates(df):
    """
    Aggregates ALL numeric columns (except SK_ID_PREV) 
    using mean, sum, max, min => columns like "installments_agg_...".
    """
    numeric_df = df.select_dtypes(include=['number']).drop(columns=['SK_ID_PREV'], errors='ignore')
    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('SK_ID_CURR').agg(agg_funcs)
    
    # Flatten MultiIndex -> "installments_agg_NUM_INSTALMENT_VERSION_mean", etc.
    agg_numeric.columns = ['installments_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    return agg_numeric

###############################################################################
# 3) Custom Aggregators (Matching Notebook Logic)
###############################################################################
def compute_installments_credit_activity(df):
    """
    # Aggregation from EDA:
      - installments_NUM_PAYMENTS = count of SK_ID_PREV
      - installments_NUM_MISSED_PAYMENTS = sum of MISSED_PAYMENT
      - installments_MISSED_PAYMENT_RATIO = mean of MISSED_PAYMENT
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        installments_NUM_PAYMENTS=("SK_ID_PREV", "count"),
        installments_NUM_MISSED_PAYMENTS=("MISSED_PAYMENT", "sum"),
        installments_MISSED_PAYMENT_RATIO=("MISSED_PAYMENT", "mean")
    ).reset_index()
    return df_grp


def compute_installments_loan_amount(df):
    """
    # Aggregation from EDA:
      - installments_MEAN_INSTALMENT_AMOUNT = mean of AMT_INSTALMENT
      - installments_MEAN_ACTUAL_PAYMENT = mean of AMT_PAYMENT
      - installments_SUM_INSTALMENT_AMOUNT = sum of AMT_INSTALMENT
      - installments_SUM_ACTUAL_PAYMENT = sum of AMT_PAYMENT
      - installments_PAYMENT_COMPLIANCE_RATIO = sum(actual)/sum(instalment)
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        installments_MEAN_INSTALMENT_AMOUNT=("AMT_INSTALMENT", "mean"),
        installments_MEAN_ACTUAL_PAYMENT=("AMT_PAYMENT", "mean"),
        installments_SUM_INSTALMENT_AMOUNT=("AMT_INSTALMENT", "sum"),
        installments_SUM_ACTUAL_PAYMENT=("AMT_PAYMENT", "sum"),
    ).reset_index()
    
    df_grp["installments_PAYMENT_COMPLIANCE_RATIO"] = (
        df_grp["installments_SUM_ACTUAL_PAYMENT"] /
        df_grp["installments_SUM_INSTALMENT_AMOUNT"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df_grp


def compute_installments_time_based(df):
    """
    # Aggregation from EDA:
      - installments_MEAN_DAYS_ENTRY_PAYMENT, installments_STD_DAYS_ENTRY_PAYMENT
      - installments_MIN_DAYS_ENTRY_PAYMENT, installments_MAX_DAYS_ENTRY_PAYMENT
      - installments_MEAN_DAYS_INSTALMENT, installments_MIN_DAYS_INSTALMENT, installments_MAX_DAYS_INSTALMENT
      - installments_TIME_SINCE_LAST_PAYMENT = abs of max DAYS_ENTRY_PAYMENT
      - installments_MEAN_PAYMENT_DELAY = abs(mean(DAYS_ENTRY_PAYMENT) - mean(DAYS_INSTALMENT))
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        installments_MEAN_DAYS_ENTRY_PAYMENT=("DAYS_ENTRY_PAYMENT", "mean"),
        installments_STD_DAYS_ENTRY_PAYMENT=("DAYS_ENTRY_PAYMENT", "std"),
        installments_MIN_DAYS_ENTRY_PAYMENT=("DAYS_ENTRY_PAYMENT", "min"),
        installments_MAX_DAYS_ENTRY_PAYMENT=("DAYS_ENTRY_PAYMENT", "max"),
        installments_MEAN_DAYS_INSTALMENT=("DAYS_INSTALMENT", "mean"),
        installments_MIN_DAYS_INSTALMENT=("DAYS_INSTALMENT", "min"),
        installments_MAX_DAYS_INSTALMENT=("DAYS_INSTALMENT", "max"),
    ).reset_index()
    
    df_grp["installments_TIME_SINCE_LAST_PAYMENT"] = df_grp["installments_MAX_DAYS_ENTRY_PAYMENT"].abs()
    df_grp["installments_MEAN_PAYMENT_DELAY"] = (
        df_grp["installments_MEAN_DAYS_ENTRY_PAYMENT"] - df_grp["installments_MEAN_DAYS_INSTALMENT"]
    ).abs()
    
    return df_grp


def compute_installments_overdue(df):
    """
    # Aggregation from EDA:
      - installments_MEAN_PAYMENT_DELAY = mean of PAYMENT_DELAY
      - installments_STD_PAYMENT_DELAY = std of PAYMENT_DELAY
      - installments_NUM_LATE_PAYMENTS = sum of LATE_PAYMENT
      - installments_LATE_PAYMENT_RATIO = mean of LATE_PAYMENT
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        installments_MEAN_PAYMENT_DELAY=("PAYMENT_DELAY", "mean"),
        installments_STD_PAYMENT_DELAY=("PAYMENT_DELAY", "std"),
        installments_NUM_LATE_PAYMENTS=("LATE_PAYMENT", "sum"),
        installments_LATE_PAYMENT_RATIO=("LATE_PAYMENT", "mean")
    ).reset_index()
    return df_grp


###############################################################################
# 4) Master Function That Produces Final DataFrame
###############################################################################
def generate_installments_features(df_installments_payments):
    """
    Replicates entire EDA/Feature Engineering for 'installments_payments':
      A) Row-Level Features (PAYMENT_RATIO, MISSED_PAYMENT, LATE_PAYMENT, PAYMENT_DELAY)
      B) Numeric Aggregates => "installments_agg_*" columns
      C) Custom Aggregators => credit_activity, loan_amount, time_based, overdue
      D) Merge them all into a single DataFrame keyed by SK_ID_CURR
    """

    # A) Row-level features
    df_temp = create_row_level_features(df_installments_payments)

    # B) Numeric aggregator => "installments_agg_*"
    df_numeric_agg = compute_numeric_aggregates(df_temp)

    # C) Custom aggregator DataFrames
    df_credit_activity = compute_installments_credit_activity(df_temp)
    df_loan_amount = compute_installments_loan_amount(df_temp)
    df_time_based = compute_installments_time_based(df_temp)
    df_overdue = compute_installments_overdue(df_temp)

    # D) Merge everything on 'SK_ID_CURR'
    df_final = df_numeric_agg.merge(df_credit_activity, on="SK_ID_CURR", how="left")
    df_final = df_final.merge(df_loan_amount, on="SK_ID_CURR", how="left")
    df_final = df_final.merge(df_time_based, on="SK_ID_CURR", how="left")
    df_final = df_final.merge(df_overdue, on="SK_ID_CURR", how="left")

    return df_final
