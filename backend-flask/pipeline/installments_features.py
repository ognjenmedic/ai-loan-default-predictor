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
    Creates row-level features for installments data:
      - PAYMENT_RATIO: Ratio of actual payment to the expected installment.
      - MISSED_PAYMENT: Indicates if a payment was missed (1 if payment is zero).
      - LATE_PAYMENT: Flags late payments (1 if payment was made after due date).
      - PAYMENT_DELAY: Number of days between payment date and due date.
    """
    df = df.copy()
    
    df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)
    
    df["MISSED_PAYMENT"] = (df["AMT_PAYMENT"] == 0).astype(int)
    
    df["LATE_PAYMENT"] = (df["DAYS_ENTRY_PAYMENT"] > df["DAYS_INSTALMENT"]).astype(int)
    
    df["PAYMENT_DELAY"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    
    return df

# Numeric Aggregator => "installments_agg_*"
def compute_numeric_aggregates(df):
    """
    Computes aggregate statistics for numeric columns (excluding SK_ID_PREV).
    Aggregates include mean, sum, max, and min, with column names prefixed as 'installments_agg_'.
    """
    numeric_df = df.select_dtypes(include=['number']).drop(columns=['SK_ID_PREV'], errors='ignore')
    agg_funcs = ['mean', 'sum', 'max', 'min']
    agg_numeric = numeric_df.groupby('SK_ID_CURR').agg(agg_funcs)
    
    # Flatten MultiIndex -> "installments_agg_NUM_INSTALMENT_VERSION_mean", etc.
    agg_numeric.columns = ['installments_agg_' + '_'.join(col) for col in agg_numeric.columns]
    agg_numeric.reset_index(inplace=True)
    return agg_numeric


def compute_installments_credit_activity(df):
    """
    Computes credit activity metrics:
      - Total number of installment payments.
      - Total number of missed payments.
      - Ratio of missed payments to total payments.
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        installments_NUM_PAYMENTS=("SK_ID_PREV", "count"),
        installments_NUM_MISSED_PAYMENTS=("MISSED_PAYMENT", "sum"),
        installments_MISSED_PAYMENT_RATIO=("MISSED_PAYMENT", "mean")
    ).reset_index()
    return df_grp


def compute_installments_loan_amount(df):
    """
    Computes loan amount-related metrics:
      - Average and total installment amounts.
      - Average and total actual payments.
      - Payment compliance ratio (total actual payment / total installment amount).
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
    Computes time-related metrics:
      - Mean, std, min, and max days until payment.
      - Mean and max days until installment due.
      - Time since last payment (absolute max days).
      - Mean payment delay (difference between expected and actual payment date).
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
    Computes overdue payment statistics:
      - Mean and std of payment delays.
      - Number and ratio of late payments.
    """
    df_grp = df.groupby("SK_ID_CURR").agg(
        installments_MEAN_PAYMENT_DELAY=("PAYMENT_DELAY", "mean"),
        installments_STD_PAYMENT_DELAY=("PAYMENT_DELAY", "std"),
        installments_NUM_LATE_PAYMENTS=("LATE_PAYMENT", "sum"),
        installments_LATE_PAYMENT_RATIO=("LATE_PAYMENT", "mean")
    ).reset_index()
    return df_grp



def generate_installments_features(df_installments_payments):
    """
    Computes and merges all engineered features for installments payments.

    Steps:
    1. Create row-level features (PAYMENT_RATIO, MISSED_PAYMENT, etc.).
    2. Compute numeric aggregates for all numerical data.
    3. Compute credit activity, loan amount, time-based, and overdue features.
    4. Merge all datasets into a single DataFrame at `SK_ID_CURR` level.
    """

    logging.info("Starting installments feature computation...")

    # A) Row-level features
    df_temp = create_row_level_features(df_installments_payments)

    # B) Numeric aggregator => "installments_agg_*"
    df_numeric_agg = compute_numeric_aggregates(df_temp)

   # Debug log aggregator columns
    logging.debug("Numeric Aggregates Computed:")
    logging.debug(f"Columns: {df_numeric_agg.columns.tolist()}")
    logging.debug(f"Sample Rows:\n{df_numeric_agg.head(3)}")


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

    logging.info(f"✅ Installments features generated. Final shape: {df_final.shape}")

    return df_final
