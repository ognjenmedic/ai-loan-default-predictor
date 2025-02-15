from flask import Flask, request, jsonify
import pandas as pd
import pickle
import json
import numpy as np
from pipeline.orchestrator import orchestrate_features

app = Flask(__name__)

# Load metadata and model at startup
with open("df_final_features.json", "r") as f:
    metadata = json.load(f)

TRAIN_FEATURES = metadata["features"]
TRAIN_DTYPES = {col: metadata["dtypes"][col] for col in TRAIN_FEATURES}
CATEGORY_MAPPINGS = metadata.get("category_mappings", {})

with open("models/lightgbm_loan_default.pkl", "rb") as f:
    model = pickle.load(f)

def enforce_dtypes(df):
    """Enforce data types on a DataFrame based on TRAIN_DTYPES and CATEGORY_MAPPINGS."""
    for col in TRAIN_FEATURES:
        if col in df.columns:
            desired_dtype = TRAIN_DTYPES[col]
            if desired_dtype == 'category':
                cats = CATEGORY_MAPPINGS.get(col, None)
                if cats is not None:
                    df[col] = df[col].astype(pd.CategoricalDtype(categories=cats, ordered=False))
                else:
                    df[col] = df[col].astype('category')
            else:
                df[col] = df[col].astype(desired_dtype)
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1) Receive the full JSON payload (nested structure).
        payload = request.get_json(force=True)
        
        # 2) Orchestrate features (merges + engineered features).
        input_df = orchestrate_features(payload)

        # 3) Remove ID columns and target (if present), 
        #    mirroring your training step:
        for col in ["SK_ID_CURR", "SK_ID_BUREAU", "TARGET"]:
            if col in input_df.columns:
                input_df.drop(columns=col, inplace=True)

        # 4) Enforce data types to match training.
        input_df = enforce_dtypes(input_df)

        # 5) Ensure we only pass the exact training features to model.
        input_df = input_df[TRAIN_FEATURES]

        # 6) Generate prediction using loaded model.
        preds = model.predict(input_df)

        # Example threshold > 0.5
        predicted_class = int(preds[0] > 0.5)
        return jsonify({"prediction": predicted_class, "probability": preds[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

@app.route('/generate_dummy', methods=['GET'])
def generate_dummy():
    """
    Return a nested JSON payload with dummy raw features for:
      - application
      - bureau_data
      - bureau_balance_data
      - credit_card_balance_data
      - pos_cash_balance_data
      - installments_payments_data
      - previous_application_data

    Using ranges from 'feature_ranges.json' for realistic predictions.
    """
    try:
        # 1) Load the feature ranges
        with open("feature_ranges.json", "r") as f:
            feature_ranges = json.load(f)

        # 2) Helper function: generate a single row for a given table
        def _generate_table_row(table_name):
            row = {}
            # If the table_name doesn't exist in feature_ranges, skip
            table_info = feature_ranges.get(table_name, {})
            for col_name, stats in table_info.items():
                # Skip 'TARGET'
                if col_name.upper() == "TARGET":
                    continue

                row[col_name] = _sample_from_range(stats)
            return row

        # 3) Function that picks random values from the stats
        import random

        def _sample_from_range(stats):
            # stats: {"dtype": "float64", "min": 25650, "max": 117000000, "mean":..., "std":..., "categories": [...]} 
            col_dtype = stats["dtype"]

            # 3a) Numeric columns (int64 / float64)
            if col_dtype in ["int64", "float64", "Int64"]:
                col_min = stats.get("min", 0)
                col_max = stats.get("max", 1)
                # Simple approach: uniform random between min and max
                val = random.uniform(col_min, col_max)
                # If column is int, convert 
                if "int" in col_dtype.lower():
                    val = int(val)
                return val

            # 3b) Categorical columns
            elif col_dtype == "category":
                categories = stats.get("categories", ["A"])  # fallback if missing
                if categories:
                    return random.choice(categories)
                else:
                    return "A"  # fallback

            # fallback
            return None

        # 4) Build the final nested payload
        dummy_payload = {}

        # 4a) Top-level "SK_ID_CURR"
        # Use the range from the 'application' table if you like, or just pick a random int
        # But let's do it from the feature_ranges
        app_ranges = feature_ranges.get("application", {})
        if "SK_ID_CURR" in app_ranges:
            # Generate SK_ID_CURR from the range
            dummy_payload["SK_ID_CURR"] = _sample_from_range(app_ranges["SK_ID_CURR"])
        else:
            # fallback
            dummy_payload["SK_ID_CURR"] = 123456

        # 4b) "application" (single row)
        application_row = _generate_table_row("application")
        # Ensure SK_ID_CURR matches the top-level
        application_row["SK_ID_CURR"] = dummy_payload["SK_ID_CURR"]
        dummy_payload["application"] = application_row

        # 4c) bureau_data (list of rows)
        bureau_rows = []
        # Suppose we pick 2 random bureau rows 
        for i in range(2):
            bureau_row = _generate_table_row("bureau_data")
            bureau_row["SK_ID_CURR"] = dummy_payload["SK_ID_CURR"]
            bureau_row["SK_ID_BUREAU"] = 9999990 + i  # or sample from range
            bureau_rows.append(bureau_row)
        dummy_payload["bureau_data"] = bureau_rows

        # 4d) bureau_balance_data (list of rows)
        # We can map the same SK_ID_BUREAU
        bb_rows = []
        for i, bureau_row in enumerate(bureau_rows):
            # create 1 or 2 rows for each bureau
            for j in range(2):
                bb_row = _generate_table_row("bureau_balance_data")
                bb_row["SK_ID_BUREAU"] = bureau_row["SK_ID_BUREAU"]
                bb_rows.append(bb_row)
        dummy_payload["bureau_balance_data"] = bb_rows

        # 4e) credit_card_balance_data (1 row for demonstration)
        ccb_row = _generate_table_row("credit_card_balance_data")
        ccb_row["SK_ID_CURR"] = dummy_payload["SK_ID_CURR"]
        ccb_row["SK_ID_PREV"] = 8888881
        dummy_payload["credit_card_balance_data"] = [ccb_row]

        # 4f) pos_cash_balance_data
        pos_row = _generate_table_row("pos_cash_balance_data")
        pos_row["SK_ID_CURR"] = dummy_payload["SK_ID_CURR"]
        pos_row["SK_ID_PREV"] = 8888882
        dummy_payload["pos_cash_balance_data"] = [pos_row]

        # 4g) installments_payments_data
        inst_row = _generate_table_row("installments_payments_data")
        inst_row["SK_ID_CURR"] = dummy_payload["SK_ID_CURR"]
        inst_row["SK_ID_PREV"] = 8888883
        dummy_payload["installments_payments_data"] = [inst_row]

        # 4h) previous_application_data
        prev_app_row = _generate_table_row("previous_application_data")
        prev_app_row["SK_ID_CURR"] = dummy_payload["SK_ID_CURR"]
        prev_app_row["SK_ID_PREV"] = 8888884
        dummy_payload["previous_application_data"] = [prev_app_row]

        # 5) Debug print
        print("\n🔍 [DEBUG] Final dummy_payload keys:", list(dummy_payload.keys()))
        
        return jsonify(dummy_payload), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ✅ Now run Flask only after all routes are registered
if __name__ == '__main__':
    print("🔍 Registered Routes:", app.url_map) 
    app.run(host="0.0.0.0", port=5001, debug=True)
