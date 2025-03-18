import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def force_categorical_dtypes(df, table_name):
    """
    Reads from feature_ranges.json to see if columns
    should be category. Then casts them so that aggregator picks them up.
    """
    import json

    with open("feature_ranges.json", "r") as f:
        feature_ranges = json.load(f)
    
    if table_name not in feature_ranges:
        return df  
    
    table_cols = feature_ranges[table_name]
    for col_name, stats in table_cols.items():
        if stats["dtype"] == "category" and col_name in df.columns:
            df[col_name] = df[col_name].astype("category")
    return df

# -----------------------------------
# 1. Application Data
# -----------------------------------
def process_application_data(payload):
    """
    Convert the 'application' field in the payload into a DataFrame for merging.
    Assumes 'application' is a *single row* of data, in dictionary form.
    (Or if front-end sends multiple records, adapt accordingly.)
    """
    if "application" in payload:
        df_app = pd.DataFrame([payload["application"]])
        return df_app
    return None

# -----------------------------------
# 2. Bureau Data
# -----------------------------------
from pipeline.bureau_features import generate_bureau_features
from pipeline.bureau_aggregation import aggregate_bureau_features

def process_bureau_data(payload):
    if "bureau_data" in payload:
        df_bureau = pd.DataFrame(payload["bureau_data"])
                
        # Force categories
        df_bureau = force_categorical_dtypes(df_bureau, "bureau_data")
        
        bureau_engineered_feats = generate_bureau_features(df_bureau)
        aggregated_bureau_feats = aggregate_bureau_features(
            df_bureau,
            additional_feature_dfs=[(bureau_engineered_feats, "bureau_engineered_features")]
        )
        return aggregated_bureau_feats
    return None

# -----------------------------------
# 3. Bureau Balance
# -----------------------------------
from pipeline.bureau_balance_features import generate_bureau_balance_features
from pipeline.bureau_balance_aggregation import (
    aggregate_bureau_balance_features,
    final_aggregate_bureau_balance_by_sk_id_curr
)
from pipeline.bureau_balance_mapping import map_bureau_balance_to_curr

def process_bureau_balance_data(payload):
    """
    Orchestrates the entire bureau balance pipeline:
      1) Feature engineering
      2) Aggregation at SK_ID_BUREAU
      3) Mapping SK_ID_BUREAU -> SK_ID_CURR
      4) Final aggregation at SK_ID_CURR
    """
    if "bureau_balance_data" in payload and "bureau_data" in payload:
        df_bureau_balance = pd.DataFrame(payload["bureau_balance_data"])
        df_bureau = pd.DataFrame(payload["bureau_data"])

        # 1) Feature engineering
        bureau_balance_engineered_feats = generate_bureau_balance_features(df_bureau_balance)

        # 2) Initial aggregation at SK_ID_BUREAU
        aggregated_bureau_balance_feats = aggregate_bureau_balance_features(
            df_bureau_balance,
            additional_feature_dfs=[(bureau_balance_engineered_feats, "bureau_balance_engineered_features")]
        )

        # 3) Map SK_ID_BUREAU -> SK_ID_CURR
        bureau_mapping = df_bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].drop_duplicates()
        aggregated_bureau_balance_feats_with_curr = map_bureau_balance_to_curr(
            aggregated_bureau_balance_feats,
            bureau_mapping
        )

        # 4) Final aggregation at SK_ID_CURR
        df_bureau_balance_aggregated_with_curr_final = final_aggregate_bureau_balance_by_sk_id_curr(
            aggregated_bureau_balance_feats_with_curr
        )

        return df_bureau_balance_aggregated_with_curr_final

    # If "bureau_balance_data" or "bureau_data" not in payload, return None
    return None


# -----------------------------------
# 4. Credit Card Balance
# -----------------------------------
from pipeline.credit_card_balance_features import generate_credit_card_balance_features
from pipeline.credit_card_balance_aggregation import aggregate_credit_card_balance_features

def process_credit_card_balance_data(payload):
    if "credit_card_balance_data" in payload:
        df_cc = pd.DataFrame(payload["credit_card_balance_data"])

        # Force categories
        df_cc = force_categorical_dtypes(df_cc, "credit_card_balance_data")

        cc_engineered_feats = generate_credit_card_balance_features(df_cc)
        aggregated_cc_feats = aggregate_credit_card_balance_features(
            df_cc,
            additional_feature_dfs=[(cc_engineered_feats, "credit_card_engineered_features")]
        )
        return aggregated_cc_feats
    return None

# -----------------------------------
# 5. POS Cash Balance
# -----------------------------------
from pipeline.pos_cash_balance_features import generate_pos_cash_balance_features
from pipeline.pos_cash_balance_aggregation import aggregate_pos_cash_balance_features

def process_pos_cash_balance_data(payload):
    if "pos_cash_balance_data" in payload:
        df_pos = pd.DataFrame(payload["pos_cash_balance_data"])
                        
        # Force categories
        df_pos = force_categorical_dtypes(df_pos, "pos_cash_balance_data")
        
        # Step A) Generate specialized aggregator features
        pos_engineered_feats = generate_pos_cash_balance_features(df_pos)
        
        # Step B) Use numeric & categorical aggregator on the raw data,
        # then merge the specialized features at the end.
        aggregated_pos_feats = aggregate_pos_cash_balance_features(
            df_pos,
            additional_feature_dfs=[(pos_engineered_feats, "pos_cash_engineered_features")]
        )
        return aggregated_pos_feats
    return None

# -----------------------------------
# 6. Previous Application
# -----------------------------------
from pipeline.previous_application_features import generate_previous_application_features
from pipeline.previous_application_aggregation import aggregate_previous_application_features

def process_previous_application_data(payload):
    if "previous_application_data" in payload:
        df_prev_app = pd.DataFrame(payload["previous_application_data"])

        # Force categories
        df_prev_app = force_categorical_dtypes(df_prev_app, "previous_application_data")
        
        # 1. Compute engineered previous application features (including DAYS_DECISION_BIN)
        prev_app_engineered_feats = generate_previous_application_features(df_prev_app)
        
        # 2. Aggregate previous application features (numeric/categorical) 
        #    and merge engineered features
        aggregated_prev_app_feats = aggregate_previous_application_features(
            df_prev_app, 
            additional_feature_dfs=[(prev_app_engineered_feats, "previous_app_engineered_features")]
        )
        return aggregated_prev_app_feats
    return None

# -----------------------------------
# 7. Installments Payments
# -----------------------------------
from pipeline.installments_features import generate_installments_features
from pipeline.installments_aggregation import aggregate_installments_payments_features

def process_installments_payments_data(payload):
    if "installments_payments_data" in payload:
        df_inst = pd.DataFrame(payload["installments_payments_data"])

        # 1) Log dtypes before forcing categories
        logging.debug("(Before categoricals) df_inst dtypes:")
        logging.debug(f"{df_inst.dtypes}")
        logging.debug(f"{df_inst.head(3)}")


        # 2) Force categories
        df_inst = force_categorical_dtypes(df_inst, "installments_payments_data")
        
        # 3) Cast certain columns back to numeric 
        #    so the aggregator will produce mean, sum, min, max columns.
        numeric_cols = ["NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
                        "AMT_INSTALMENT", "AMT_PAYMENT"]
        for col in numeric_cols:
            if col in df_inst.columns:
                df_inst[col] = pd.to_numeric(df_inst[col], errors="coerce")

        # Making sure SK_ID_CURR is also numeric (int) to match merges
        if "SK_ID_CURR" in df_inst.columns:
            df_inst["SK_ID_CURR"] = pd.to_numeric(df_inst["SK_ID_CURR"], errors="coerce")

        # 4) Pass to aggregator function
        final_installments_feats = generate_installments_features(df_inst)
        return final_installments_feats

    return None



# -----------------------------------
# Merge Helper
# -----------------------------------
def merge_features(main_df, features_list):
    """
    Merge a list of feature DataFrames into the main DataFrame using SK_ID_CURR.
    """
    for feature_df in features_list:
        if feature_df is not None and not feature_df.empty:
            main_df = main_df.merge(feature_df, on="SK_ID_CURR", how="left")
    return main_df

# -----------------------------------
# Orchestrator
# -----------------------------------
def orchestrate_features(payload):
    """
    1) Create an 'application' DataFrame from payload (the main dataset).
    2) Process each of the other tables to get aggregated features.
    3) Merge them all with the application DataFrame on SK_ID_CURR.
    4) Return the final DataFrame.
    """

    # Step 1) Build the main application DataFrame
    df_app = process_application_data(payload)
    if df_app is None or df_app.empty:
        raise ValueError("'application' data is required in payload to build the main DataFrame.")
    
    # Check if SK_ID_CURR is in the application data
    if "SK_ID_CURR" not in df_app.columns:
        raise ValueError("'SK_ID_CURR' is missing in the application data! Cannot merge.")
    
    logging.info(f"✅ Main application shape: {df_app.shape}")
    logging.debug(f"{df_app.head(1)}")
    
    # Step 2) Process each of the other data sources
    bureau_feats = process_bureau_data(payload)
    bureau_balance_feats = process_bureau_balance_data(payload)
    pos_cash_feats = process_pos_cash_balance_data(payload)
    credit_card_feats = process_credit_card_balance_data(payload)
    previous_app_feats = process_previous_application_data(payload)
    installments_feats = process_installments_payments_data(payload)

    # Collect them for merging
    features_dict = {
        "Bureau Features": bureau_feats,
        "Bureau Balance Features": bureau_balance_feats,
        "POS Cash Features": pos_cash_feats,
        "Credit Card Features": credit_card_feats,
        "Previous Application Features": previous_app_feats,
        "Installments Features": installments_feats
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
