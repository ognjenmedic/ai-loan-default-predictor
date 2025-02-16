from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import json
import random
import numpy as np
import scipy.stats as stats
from scipy.stats import truncnorm

from pipeline.orchestrator import orchestrate_features

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}, r"/generate_dummy": {"origins": "http://localhost:4200"}})

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

        # Print or log the row
        print("\n[DEBUG] Final input_df shape:", input_df.shape)
        print(input_df.iloc[0].to_dict())  # or head() if multiple rows

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
        return jsonify({
            "prediction": predicted_class, 
            "probability": preds[0], 
            "formatted_probability": f"{round(preds[0] * 100, 2)}%"
        })

    
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
            """
            Generates realistic values for numeric and categorical features.
            Uses:
            - Log-normal for money-related fields (e.g., income, credit amount),
                converting rubles to USD at a rate of 90 RUB per USD.
            - Truncated normal for other numeric fields (within the min/max),
            - Reasonable bounds for loan terms and durations,
            - Realistic frequency sampling for categorical features.
            """
            import random
            import numpy as np
            from scipy.stats import truncnorm

            # Define which columns are money-related (will convert from rubles to USD).
            money_fields = [
                "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT",
                "AMT_BALANCE", "AMT_PAYMENT_CURRENT"
            ]
            # Conversion factor from rubles to USD
            RUB_TO_USD = 90.0

            # Extract metadata from 'stats'
            col_dtype = stats["dtype"]
            feature_name = stats.get("feature_name", "")
            col_min = stats.get("min", 0)
            col_max = stats.get("max", 1)
            col_mean = stats.get("mean", (col_max + col_min) / 2)  # fallback
            col_std = stats.get("std", max((col_max - col_min) / 4, 1e-6))  # avoid zero std

            # 1) Handle money fields in USD with log-normal sampling
            if feature_name in money_fields and col_mean > 0:
                # No bounding/clamping
                sigma = 0.3
                col_mean_usd = col_mean / RUB_TO_USD  # If you still want to display USD
                val_usd = np.random.lognormal(mean=np.log(col_mean_usd), sigma=sigma)

                # Just return it without clamping
                return round(val_usd, 2)

            # 2) Fix Loan Terms (Months) if feature name starts with "CNT_"
            if feature_name.startswith("CNT_"):
                # e.g. ~5 years +- some noise, clipped to [6, 360] months
                return int(min(max(random.gauss(60, 30), 6), 360))

            # 3) Convert "DAYS_*" columns to approximate years (absolute value)
            if "DAYS" in feature_name.upper():
                # e.g. if mean = -2000 => ~5.5 years
                years = abs(col_mean) / 365.0
                # Add a little random variation
                variation = random.uniform(-0.5, 0.5)
                return round(years + variation, 1)

            # 4) For other numeric columns, use truncated normal
            if col_dtype in ["int64", "float64", "Int64"]:
                a, b = (col_min - col_mean) / col_std, (col_max - col_mean) / col_std
                val = truncnorm.rvs(a, b, loc=col_mean, scale=col_std)

                # If it's an external credit source [0, 1], clamp to 0–1
                if "EXT_SOURCE" in feature_name.upper():
                    val = min(max(val, 0.0), 1.0)

                # Return int if original is integer, else float
                if "int" in col_dtype.lower():
                    return int(round(val))
                else:
                    return float(round(val, 2))

            # 5) Categorical columns: sample uniformly from known categories
            if col_dtype == "category":
                categories = stats.get("categories", ["Unknown"])
                return random.choice(categories) if categories else "Unknown"

            # 6) Fallback
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
