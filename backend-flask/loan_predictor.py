from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import json
import random
import numpy as np
import scipy.stats as stats
from scipy.stats import truncnorm
import logging
from pandas.api.types import CategoricalDtype 
from pipeline.orchestrator import orchestrate_features
from tools.feature_config import DOMAIN_OVERRIDES
import shap


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler() 
    ]
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:4200", "http://3.94.247.36", "https://ai.fullstackista.com"]}})

# Load metadata and model at startup
with open("df_final_features.json", "r") as f:
    metadata = json.load(f)

TRAIN_FEATURES = metadata["features"]
TRAIN_DTYPES = {col: metadata["dtypes"][col] for col in TRAIN_FEATURES}
CATEGORY_MAPPINGS = metadata.get("category_mappings", {})

with open("models/lightgbm_loan_default.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model) 


def enforce_dtypes(df):
    """
    Cast every column to the dtype recorded at training time.
    • Numeric   → use pd.to_numeric(..., errors='coerce') first
    • Category  → attach the training category list
    """
    for col in TRAIN_FEATURES:
        if col not in df.columns:
            continue

        desired = TRAIN_DTYPES[col]

        if desired in ("int64", "Int64"):
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")   
                  .round(0)
                  .astype("Int64")                     
            )

        elif desired == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        elif desired == "category":
            cats = CATEGORY_MAPPINGS.get(col)
            if cats is not None:
                df[col] = df[col].astype(
                    pd.CategoricalDtype(categories=cats, ordered=False)
                )
            else:
                df[col] = df[col].astype("category")

        else:                            # fallback
            df[col] = df[col].astype(desired)

    return df


def safe_fill(df):
    """
    Numeric columns  → fillna(0)  
    Categorical cols → add 'Missing' level and fillna('Missing')
    """
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            if "Missing" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(["Missing"])
            df[col] = df[col].fillna("Missing")
        else:
            df[col] = df[col].fillna(0)
    return df


def get_shap_explanations(df_row: pd.DataFrame):
    shap_vals = explainer.shap_values(df_row, check_additivity=False)
    contribs   = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]

    top_idx = np.argsort(np.abs(contribs))[::-1][:3]
    return [
        {
            "feature": feat,
            "value": str(df_row.iloc[0, idx]),
            "impact": float(contribs[idx]),
            "direction": "increased risk" if contribs[idx] > 0 else "reduced risk"
        }
        for idx, feat in zip(top_idx, df_row.columns[top_idx])
    ]
    return explanations




@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # 1) full payload → engineered feature frame
        payload  = request.get_json(force=True)
        input_df = orchestrate_features(payload)

        # 2) match training schema
        input_df = input_df.reindex(columns=TRAIN_FEATURES, fill_value=0)

        # 3) enforce correct dtypes *before* touching NaNs
        input_df = enforce_dtypes(input_df)

        # 4) now fill only categorical NaNs with "Missing"
        for col in input_df.columns:
            if isinstance(input_df[col].dtype, CategoricalDtype):
                if "Missing" not in input_df[col].cat.categories:
                    input_df[col] = input_df[col].cat.add_categories(["Missing"])
                input_df[col] = input_df[col].fillna("Missing")


        # 5) probability + label
        proba  = model.predict_proba(input_df)[0, 1]
        label  = int(proba > 0.5)

        logging.info(f"✅ Prediction: {label}  (p = {proba:.3f})")

        explanations = get_shap_explanations(input_df)

        return jsonify({
            "prediction": label,
            "probability": proba,
            "formatted_probability": f"{proba*100:.2f}%",
            "explanations": explanations
        })


    except Exception as e:
        logging.error("❌ Prediction error", exc_info=True)
        return jsonify({"error": str(e)}), 500

    
@app.route("/api/feature_metadata", methods=["GET"])
def feature_metadata():
    """
    Return category_mappings so the front-end can populate dropdowns.
    """
    return jsonify({"category_mappings": CATEGORY_MAPPINGS}), 200


@app.route('/api/generate_dummy', methods=['GET'])
def generate_dummy():
    """
    Return a nested JSON payload with synthetic raw tables:

    - customer_profile
    - credit_summary
    - credit_timeline
    - card_activity
    - cash_pos_records
    - installment_records
    - prior_loan_history

    Numeric values are drawn from log-normal or truncated-normal ranges
    defined in 'feature_ranges.json'; categorical fields are sampled from
    the synthetic category lists.
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
        def _sample_from_range(stats):
            """
            Generates realistic values for numeric and categorical features.
            Uses:
            - Log-normal for money-related fields (e.g., income, credit amount),
            - Truncated normal for other numeric fields (within the min/max),
            - Reasonable bounds for loan terms and durations,
            - Realistic frequency sampling for categorical features.
            """


            # Define which columns are money-related
            money_fields = [
                "MONTHLY_INCOME", "LOAN_AMOUNT", "LOAN_ANNUITY", "PURCHASED_GOODS_VALUE",
                "TOTAL_CREDIT_AMOUNT", "OUTSTANDING_DEBT", "CREDIT_LIMIT",
                "CURRENT_CARD_BALANCE", "CURRENT_PAYMENT"
            ]

            # Extract metadata from 'stats'
            col_dtype = stats["dtype"]
            feature_name = stats.get("feature_name", "")
            col_min = stats.get("min", 0)
            col_max = stats.get("max", 1)
            col_mean = stats.get("mean", (col_max + col_min) / 2)  # fallback
            col_std = stats.get("std", max((col_max - col_min) / 4, 1e-6))  # avoid zero std

            # Patch values if in DOMAIN_OVERRIDES
            override = DOMAIN_OVERRIDES.get(feature_name)
            if override:
                col_min = override.get("min", col_min)
                col_max = override.get("max", col_max)
                col_mean = override.get("mean", col_mean)
                col_std = override.get("std", col_std)

            # 1) Handle money fields with log-normal sampling
            if feature_name in money_fields and col_mean > 0:
                sigma = 0.3
                val = np.random.lognormal(mean=np.log(col_mean), sigma=sigma)
                return round(val, 2)

            # 2) Fix loan-term-like columns only
            if feature_name in {"PLANNED_NUM_PAYMENTS", "LOAN_TERM_MONTHS"}:
                return int(min(max(random.gauss(60, 30), 6), 360))


            # 3) Convert "DAYS_*" columns to approximate years (absolute value)
            if "DAYS" in feature_name.upper():
                val = abs(random.gauss(col_mean, col_std))  
                return int(round(val))


            # 4) For other numeric columns, use truncated normal
            if col_dtype in ["int64", "float64", "Int64"]:
                a, b = (col_min - col_mean) / col_std, (col_max - col_mean) / col_std
                val = truncnorm.rvs(a, b, loc=col_mean, scale=col_std)

                # If it's an external credit source [0, 1], clamp to 0–1
                if "EXTERNAL_CREDIT_SCORE" in feature_name.upper():
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

        # 4a) Top-level "CUSTOMER_ID"
        app_ranges = feature_ranges.get("customer_profile", {})
        if "CUSTOMER_ID" in app_ranges:
            # Generate CUSTOMER_ID from the range
            dummy_payload["CUSTOMER_ID"] = _sample_from_range(app_ranges["CUSTOMER_ID"])
        else:
            # fallback
            dummy_payload["CUSTOMER_ID"] = 123456

        # 4b) "customer_profile" (single row)
        customer_profile_row = _generate_table_row("customer_profile")
        # Ensure CUSTOMER_ID matches the top-level
        customer_profile_row["CUSTOMER_ID"] = dummy_payload["CUSTOMER_ID"]
        dummy_payload["customer_profile"] = customer_profile_row

        # 4c) credit_summary (list of rows)
        credit_summary_rows = []
        for i in range(2):
            summary_row = _generate_table_row("credit_summary")
            summary_row["CUSTOMER_ID"] = dummy_payload["CUSTOMER_ID"]
            summary_row["CREDIT_RECORD_ID"] = 9999990 + i
            credit_summary_rows.append(summary_row)
        dummy_payload["credit_summary"] = credit_summary_rows

        # 4d) credit_timeline (list of rows) – maps to CREDIT_RECORD_ID
        credit_timeline_rows = []
        for summary_row in credit_summary_rows:
            for j in range(2):
                timeline_row = _generate_table_row("credit_timeline")
                timeline_row["CREDIT_RECORD_ID"] = summary_row["CREDIT_RECORD_ID"]
                credit_timeline_rows.append(timeline_row)
        dummy_payload["credit_timeline"] = credit_timeline_rows

        # 4e) card_activity (1 row)
        ccb_row = _generate_table_row("card_activity")
        ccb_row["CUSTOMER_ID"] = dummy_payload["CUSTOMER_ID"]
        ccb_row["PRIOR_LOAN_ID"] = 8888881
        dummy_payload["card_activity"] = [ccb_row]

        # 4f) cash_pos_records (1 row)
        pos_row = _generate_table_row("cash_pos_records")
        pos_row["CUSTOMER_ID"] = dummy_payload["CUSTOMER_ID"]
        pos_row["PRIOR_LOAN_ID"] = 8888882
        dummy_payload["cash_pos_records"] = [pos_row]

        # 4g) installment_records (1 row)
        inst_row = _generate_table_row("installment_records")
        inst_row["CUSTOMER_ID"] = dummy_payload["CUSTOMER_ID"]
        inst_row["PRIOR_LOAN_ID"] = 8888883
        dummy_payload["installment_records"] = [inst_row]

        # 4h) prior_loan_history (1 row)
        prior_loan_row = _generate_table_row("prior_loan_history")
        prior_loan_row["CUSTOMER_ID"] = dummy_payload["CUSTOMER_ID"]
        prior_loan_row["PRIOR_LOAN_ID"] = 8888884
        dummy_payload["prior_loan_history"] = [prior_loan_row]

        # 5) Debug log
        logging.debug(f"🔍 Final dummy_payload keys: {list(dummy_payload.keys())}")

        
        return jsonify(dummy_payload), 200


    except Exception as e:
        logging.error(f"❌ Error generating dummy data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Run Flask
if __name__ == '__main__':
    logging.info("✅ Starting Flask server...")
    logging.info(f"Registered Routes: {app.url_map}")
    app.run(host="0.0.0.0", port=5001, debug=True)
