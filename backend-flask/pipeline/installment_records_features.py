import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Row-Level Feature Creation
def create_row_level_features(df):
    """
    Create basic row-level installment features:
      - PAYMENT_RATIO: actual / scheduled installment
      - MISSED_PAYMENT: 1 if no payment recorded
      - LATE_PAYMENT: 1 if payment recorded after due date
      - PAYMENT_DELAY: days late (entry - due date)
    """
    df = df.copy()

    df["PAYMENT_RATIO"] = df["ACTUAL_PAYMENT_AMOUNT"] / df["SCHEDULED_PAYMENT_AMOUNT"].replace(0, np.nan)
    df["MISSED_PAYMENT"] = (df["ACTUAL_PAYMENT_AMOUNT"] == 0).astype(int)
    df["LATE_PAYMENT"] = (df["DAYS_BEFORE_ACTUAL_PAYMENT"] > df["DAYS_UNTIL_DUE"]).astype(int)
    df["PAYMENT_DELAY"] = df["DAYS_BEFORE_ACTUAL_PAYMENT"] - df["DAYS_UNTIL_DUE"]

    return df

def compute_numeric_aggregates(df):
    """
    Compute standard numeric aggregates (mean, sum, max, min) for each CUSTOMER_ID.
    Excludes PRIOR_LOAN_ID.
    """
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=["PRIOR_LOAN_ID"], errors="ignore")
    agg_funcs = ["mean", "sum", "max", "min"]
    agg_numeric = numeric_df.groupby("CUSTOMER_ID").agg(agg_funcs)

    agg_numeric.columns = [f"installments_agg_{col}_{stat}" for col, stat in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    return agg_numeric

def compute_installments_credit_activity(df):
    """
    Aggregate activity-based installment indicators:
      - Total number of installment records
      - Number and ratio of missed payments
    """
    return df.groupby("CUSTOMER_ID").agg(
        installments_NUM_PAYMENTS=("PRIOR_LOAN_ID", "count"),
        installments_NUM_MISSED_PAYMENTS=("MISSED_PAYMENT", "sum"),
        installments_MISSED_PAYMENT_RATIO=("MISSED_PAYMENT", "mean")
    ).reset_index()

def compute_installments_loan_amount(df):
    """
    Aggregate loan/payment amounts and compliance:
      - Averages and totals for installment/payment amounts
      - Compliance ratio = total paid / total expected
    """
    df_grp = df.groupby("CUSTOMER_ID").agg(
        installments_MEAN_INSTALMENT_AMOUNT=("SCHEDULED_PAYMENT_AMOUNT", "mean"),
        installments_MEAN_ACTUAL_PAYMENT=("ACTUAL_PAYMENT_AMOUNT", "mean"),
        installments_SUM_INSTALMENT_AMOUNT=("SCHEDULED_PAYMENT_AMOUNT", "sum"),
        installments_SUM_ACTUAL_PAYMENT=("ACTUAL_PAYMENT_AMOUNT", "sum"),
    ).reset_index()

    df_grp["installments_PAYMENT_COMPLIANCE_RATIO"] = (
        df_grp["installments_SUM_ACTUAL_PAYMENT"] /
        df_grp["installments_SUM_INSTALMENT_AMOUNT"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_grp

def compute_installments_time_based(df):
    """
    Aggregate time-related indicators:
      - Entry and due date stats (mean, std, min, max)
      - Time since last payment
      - Mean delay (days late, absolute value)
    """
    df_grp = df.groupby("CUSTOMER_ID").agg(
        installments_MEAN_DAYS_ENTRY_PAYMENT=("DAYS_BEFORE_ACTUAL_PAYMENT", "mean"),
        installments_STD_DAYS_ENTRY_PAYMENT=("DAYS_BEFORE_ACTUAL_PAYMENT", "std"),
        installments_MIN_DAYS_ENTRY_PAYMENT=("DAYS_BEFORE_ACTUAL_PAYMENT", "min"),
        installments_MAX_DAYS_ENTRY_PAYMENT=("DAYS_BEFORE_ACTUAL_PAYMENT", "max"),
        installments_MEAN_DAYS_INSTALMENT=("DAYS_UNTIL_DUE", "mean"),
        installments_MIN_DAYS_INSTALMENT=("DAYS_UNTIL_DUE", "min"),
        installments_MAX_DAYS_INSTALMENT=("DAYS_UNTIL_DUE", "max"),
    ).reset_index()

    df_grp["installments_TIME_SINCE_LAST_PAYMENT"] = df_grp["installments_MAX_DAYS_ENTRY_PAYMENT"].abs()
    df_grp["installments_MEAN_PAYMENT_DELAY"] = (
        df_grp["installments_MEAN_DAYS_ENTRY_PAYMENT"]
        - df_grp["installments_MEAN_DAYS_INSTALMENT"]
    ).abs()

    return df_grp

def compute_installments_overdue(df):
    """
    Compute overdue metrics:
      - Mean and std delay across payments
      - Count and ratio of late payments
    """
    return df.groupby("CUSTOMER_ID").agg(
        installments_MEAN_PAYMENT_DELAY=("PAYMENT_DELAY", "mean"),
        installments_STD_PAYMENT_DELAY=("PAYMENT_DELAY", "std"),
        installments_NUM_LATE_PAYMENTS=("LATE_PAYMENT", "sum"),
        installments_LATE_PAYMENT_RATIO=("LATE_PAYMENT", "mean")
    ).reset_index()

def generate_installments_features(df_installment_records):
    """
    Full feature generator for installment_records.
    Steps:
      1. Create row-level delay and payment flags
      2. Aggregate numeric and custom features
      3. Merge all into final CUSTOMER_ID-level output
    """
    logging.info("Starting installment records feature computation...")

    df_temp = create_row_level_features(df_installment_records)
    df_numeric_agg = compute_numeric_aggregates(df_temp)

    df_credit_activity = compute_installments_credit_activity(df_temp)
    df_loan_amount = compute_installments_loan_amount(df_temp)
    df_time_based = compute_installments_time_based(df_temp)
    df_overdue = compute_installments_overdue(df_temp)

    df_final = df_numeric_agg \
        .merge(df_credit_activity, on="CUSTOMER_ID", how="left") \
        .merge(df_loan_amount, on="CUSTOMER_ID", how="left") \
        .merge(df_time_based, on="CUSTOMER_ID", how="left") \
        .merge(df_overdue, on="CUSTOMER_ID", how="left")

    logging.info(f"✅ Installment records features generated. Final shape: {df_final.shape}")
    return df_final
