import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def safe_int_id(df):
    """
    Ensure CUSTOMER_ID is Int64 (nullable integer) so downstream groupbys
    and merges all use the same dtype.
    """
    if "CUSTOMER_ID" in df.columns:
        df["CUSTOMER_ID"] = (
            pd.to_numeric(df["CUSTOMER_ID"], errors="coerce")
              .astype("Int64")
        )
    return df


def to_int64(series):
    """
    Coerce any column to pandas' nullable Int64.
    Non-numeric values become <NA> instead of raising.
    """
    return (
        pd.to_numeric(series, errors="coerce")   
          .round(0)                             
          .astype("Int64")                       
    )


def force_categorical_dtypes(df, table_name):
    """
    Reads from feature_ranges.json to see if columns
    should be category. Then casts them so that aggregator picks them up.
    """
    import json

    with open(Path(__file__).resolve().parents[1] / "feature_ranges.json", "r") as f:
        feature_ranges = json.load(f)
    
    if table_name not in feature_ranges:
        return df  
    
    table_cols = feature_ranges[table_name]
    for col_name, stats in table_cols.items():
        if stats["dtype"] == "category" and col_name in df.columns:
            df[col_name] = df[col_name].astype("category")
    return df

# -----------------------------------
# 1. Customer Profile Data
# -----------------------------------
def process_customer_profile(payload):
    """
    Convert the 'customer_profile' field in the payload into a DataFrame for merging.
    Assumes 'customer_profile' is a *single row* of data, in dictionary form.
    (Or if front-end sends multiple records, adapt accordingly.)
    """
    if "customer_profile" in payload:
        df_app = pd.DataFrame([payload["customer_profile"]])
        return df_app
    return None

# -----------------------------------
# 2. Credit Summary
# -----------------------------------
from pipeline.credit_summary_features import generate_credit_summary_features
from pipeline.credit_summary_aggregation import aggregate_credit_summary_features

def process_credit_summary(payload):
    if "credit_summary" in payload:
        df_credit_summary = pd.DataFrame(payload["credit_summary"])
        df_credit_summary = safe_int_id(df_credit_summary)
                
        # Force categories
        df_credit_summary = force_categorical_dtypes(df_credit_summary, "credit_summary")
        
        credit_summary_engineered_feats = generate_credit_summary_features(df_credit_summary)
        aggregated_credit_summary_feats = aggregate_credit_summary_features(
            df_credit_summary,
            additional_feature_dfs=[(credit_summary_engineered_feats, "credit_summary_engineered_features")]
        )
        return aggregated_credit_summary_feats
    return None

# -----------------------------------
# 3. Credit Timeline
# -----------------------------------
from pipeline.credit_timeline_features import generate_credit_timeline_features
from pipeline.credit_timeline_aggregation import (
    aggregate_credit_timeline_features,
    final_aggregate_credit_timeline_by_customer_id
)
from pipeline.credit_timeline_mapping import map_credit_timeline_to_customer

def process_credit_timeline(payload):
    """
    Orchestrates the entire credit_timeline pipeline:
      1) Feature engineering
      2) Aggregation at CREDIT_RECORD_ID
      3) Mapping CREDIT_RECORD_ID -> CUSTOMER_ID
      4) Final aggregation at CUSTOMER_ID
    """
    if "credit_timeline" in payload and "credit_summary" in payload:
        df_credit_timeline = pd.DataFrame(payload["credit_timeline"])
        df_credit_timeline = safe_int_id(df_credit_timeline)
        df_credit_summary = pd.DataFrame(payload["credit_summary"])
        df_credit_summary = safe_int_id(df_credit_summary)

        credit_timeline_engineered_feats = generate_credit_timeline_features(df_credit_timeline)
        aggregated_credit_timeline_feats = aggregate_credit_timeline_features(
            df_credit_timeline,
            additional_feature_dfs=[(credit_timeline_engineered_feats, "credit_timeline_engineered_features")]
        )

        credit_timeline_mapping = df_credit_summary[['CREDIT_RECORD_ID', 'CUSTOMER_ID']].drop_duplicates()
        aggregated_credit_timeline_feats_with_customer = map_credit_timeline_to_customer(
            aggregated_credit_timeline_feats,
            credit_timeline_mapping
        )

        df_credit_timeline_final = final_aggregate_credit_timeline_by_customer_id(
            aggregated_credit_timeline_feats_with_customer
        )

        return df_credit_timeline_final

    return None


# -----------------------------------
# 4. Card Activity
# -----------------------------------
from pipeline.card_activity_features import generate_card_activity_features
from pipeline.card_activity_aggregation import aggregate_card_activity_features

def process_card_activity(payload):
    if "card_activity" in payload:
        df_cc = pd.DataFrame(payload["card_activity"])
        df_cc = safe_int_id(df_cc)
        df_cc = force_categorical_dtypes(df_cc, "card_activity")

        cc_engineered_feats = generate_card_activity_features(df_cc)
        aggregated_cc_feats = aggregate_card_activity_features(
            df_cc,
            additional_feature_dfs=[(cc_engineered_feats, "card_activity_engineered_features")]
        )
        return aggregated_cc_feats
    return None



# -----------------------------------
# 5. Cash POS Records
# -----------------------------------
from pipeline.cash_pos_records_features import generate_cash_pos_records_features
from pipeline.cash_pos_records_aggregation import aggregate_cash_pos_records_features

def process_cash_pos_records(payload):
    if "cash_pos_records" in payload:
        df_pos = pd.DataFrame(payload["cash_pos_records"])
        df_pos = safe_int_id(df_pos) 
                        
        # Force categories
        df_pos = force_categorical_dtypes(df_pos, "cash_pos_records")

        # Step A) Generate specialized aggregator features
        pos_engineered_feats = generate_cash_pos_records_features(df_pos)

        # Step B) Use numeric & categorical aggregator on the raw data,
        # then merge the specialized features at the end.
        aggregated_pos_feats = aggregate_cash_pos_records_features(
            df_pos,
            additional_feature_dfs=[(pos_engineered_feats, "cash_pos_engineered_features")]
        )
        return aggregated_pos_feats
    return None

# -----------------------------------
# 6. Prior Loan History
# -----------------------------------
from pipeline.prior_loan_history_features import generate_prior_loan_history_features
from pipeline.prior_loan_history_aggregation import aggregate_prior_loan_history_features

def process_prior_loan_history(payload):
    if "prior_loan_history" in payload:
        df_prior_loan_history = pd.DataFrame(payload["prior_loan_history"])
        df_prior_loan_history = safe_int_id(df_prior_loan_history) 

        # Force categories
        df_prior_loan_history = force_categorical_dtypes(df_prior_loan_history, "prior_loan_history")

        # 1. Compute engineered prior loan history features (including DAYS_SINCE_DECISION_BIN)
        prior_loan_engineered_feats = generate_prior_loan_history_features(df_prior_loan_history)
        
        # 2. Aggregate prior loan history features (numeric/categorical) 
        #    and merge engineered features
        aggregated_prior_loan_feats = aggregate_prior_loan_history_features(
            df_prior_loan_history,
            additional_feature_dfs=[(prior_loan_engineered_feats, "prior_loan_engineered_features")]
        )
        return aggregated_prior_loan_feats
    return None

# -----------------------------------
# 7. Installment Records
# -----------------------------------
from pipeline.installment_records_features import generate_installments_features
from pipeline.installment_records_aggregation import aggregate_installment_records_features

def process_installment_records(payload):
    if "installment_records" in payload:
        df_inst = pd.DataFrame(payload["installment_records"])
        df_inst = safe_int_id(df_inst)


        # 1) Log dtypes before forcing categories
        logging.debug("(Before categoricals) df_inst dtypes:")
        logging.debug(f"{df_inst.dtypes}")
        logging.debug(f"{df_inst.head(3)}")


        # 2) Force categories
        df_inst = force_categorical_dtypes(df_inst, "installment_records")
        
        # 3) Cast certain columns back to numeric 
        #    so the aggregator will produce mean, sum, min, max columns.
        numeric_cols = ["INSTALLMENT_PLAN_VERSION", "INSTALLMENT_SEQUENCE_NUMBER",
                        "SCHEDULED_PAYMENT_AMOUNT", "ACTUAL_PAYMENT_AMOUNT"]
        for col in numeric_cols:
            if col in df_inst.columns:
                df_inst[col] = pd.to_numeric(df_inst[col], errors="coerce")

        # Making sure CUSTOMER_ID is also numeric (int) to match merges
        if "CUSTOMER_ID" in df_inst.columns:
            df_inst["CUSTOMER_ID"] = pd.to_numeric(df_inst["CUSTOMER_ID"], errors="coerce")

        # 4) Pass to aggregator function
        installment_engineered_feats = generate_installments_features(df_inst)
        final_installments_feats = aggregate_installment_records_features(
            df_inst,
            additional_feature_dfs=[(installment_engineered_feats, "installment_engineered_features")]
        )
        return final_installments_feats

    return None



# -----------------------------------
# Merge Helper
# -----------------------------------
def merge_features(main_df, features_list):
    """
    Merge a list of feature DataFrames into the main DataFrame using CUSTOMER_ID.
    Ensures CUSTOMER_ID is cast to int for consistency.
    """
    for feature_df in features_list:
        if feature_df is not None and not feature_df.empty:
            main_df["CUSTOMER_ID"] = to_int64(main_df["CUSTOMER_ID"])
            feature_df["CUSTOMER_ID"] = to_int64(feature_df["CUSTOMER_ID"])

            before = main_df.shape[1]
            main_df = main_df.merge(feature_df, on="CUSTOMER_ID", how="left")
            after = main_df.shape[1]
            logging.info(f"✅ Merged: +{after - before} columns from current block.")
    return main_df


# -----------------------------------
# Orchestrator
# -----------------------------------
def orchestrate_features(payload):
    """
    1) Create an 'customer_profile' DataFrame from payload (the main dataset).
    2) Process each of the other tables to get aggregated features.
    3) Merge them all with the customer_profile DataFrame on CUSTOMER_ID.
    4) Return the final DataFrame.
    """

    # Step 1) Build the main customer_profile DataFrame
    df_app = process_customer_profile(payload)
    if df_app is None or df_app.empty:
        raise ValueError("'customer_profile' data is required in payload to build the main DataFrame.")
    
    # Check if CUSTOMER_ID is in the customer_profile data
    if "CUSTOMER_ID" not in df_app.columns:
        raise ValueError("'CUSTOMER_ID' is missing in the customer_profile data! Cannot merge.")
    
    # Enforce consistent dtype
    df_app["CUSTOMER_ID"] = pd.to_numeric(df_app["CUSTOMER_ID"], downcast="integer", errors="coerce")

    
    logging.info(f"✅ Main customer_profile shape: {df_app.shape}")
    logging.debug(f"{df_app.head(1)}")
    
    # Step 2) Process each of the other data sources
    credit_summary_feats = process_credit_summary(payload)
    credit_timeline_feats = process_credit_timeline(payload)
    cash_pos_feats = process_cash_pos_records(payload)
    card_activity_feats = process_card_activity(payload)
    prior_loan_feats = process_prior_loan_history(payload)
    installment_feats = process_installment_records(payload)

    # Collect them for merging
    features_dict = {
        "Credit Summary Features": credit_summary_feats,
        "Credit Timeline Features": credit_timeline_feats,
        "Cash POS Features": cash_pos_feats,
        "Card Activity Features": card_activity_feats,
        "Prior Loan History Features": prior_loan_feats,
        "Installment Records Features": installment_feats
    }

    for feat_name, df in features_dict.items():
        if df is not None:
            logging.info(f"✅ {feat_name} shape: {df.shape}")
        else:
            logging.warning(f"❌ {feat_name} is None (no data provided).")


    # Step 3) Merge everything onto df_app
    df_merged = merge_features(df_app, [df for df in features_dict.values() if df is not None])
    
    logging.info(f"✅ Final shape after merging all features: {df_merged.shape}")
    logging.debug(f"{df_merged.head(1)}")

    # Drop 'TARGET' if it appears
    if "TARGET" in df_merged.columns:
        df_merged.drop(columns=["TARGET"], inplace=True)

    # Return the final DataFrame
    return df_merged

