"""
A2A.py
Flask Retriever Agent – provides similar past cases for a new application.
Run after data is loaded and indexes are created.

Optional SHAP + risk scores: set CHAMPION_MANIFEST_PATH (see src.inference.champion_explanation).
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import joblib

app = Flask(__name__)
scaler = None
encoder = None
medians = None
feature_cols = None


def _load_preprocessors_if_needed() -> None:
    global scaler, encoder, medians, feature_cols
    if scaler is not None and encoder is not None and medians is not None and feature_cols is not None:
        return
    scaler_path = os.getenv("RETR_SCALER_PATH", "scaler.pkl")
    encoder_path = os.getenv("RETR_ENCODER_PATH", "encoder.pkl")
    medians_path = os.getenv("RETR_MEDIANS_PATH", "medians.pkl")
    feature_cols_path = os.getenv("RETR_FEATURE_COLS_PATH", "feature_cols.pkl")
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    medians = joblib.load(medians_path)
    feature_cols = joblib.load(feature_cols_path)

# Database connection parameters
DB_PARAMS = {
    "host": "localhost",
    "port": 5433,
    "database": "postgres",
    "user": "postgres",
    "password": "mysecretpassword"
}

def get_db():
    return psycopg2.connect(**DB_PARAMS)

# Column definitions (must match training)
numerical_cols = [
    'income', 'name_email_similarity', 'prev_address_months_count',
    'current_address_months_count', 'customer_age', 'days_since_request',
    'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h',
    'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w',
    'credit_risk_score', 'bank_months_count', 'proposed_credit_limit',
    'session_length_in_minutes', 'device_distinct_emails_8w', 'device_fraud_count',
    'month'
]

categorical_cols = [
    'payment_type', 'employment_status', 'email_is_free', 'housing_status',
    'phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 'foreign_request',
    'source', 'device_os', 'keep_alive_session'
]

def preprocess_new_application(raw_dict):
    """
    Convert raw application dict into a scaled feature vector.
    raw_dict must contain all features except fraud_bool.
    """
    _load_preprocessors_if_needed()

    # Create DataFrame with one row
    df_new = pd.DataFrame([raw_dict])
    
    # Handle missing values in numerical columns using saved medians
    for col in numerical_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].replace(-1, np.nan)
            df_new[col] = df_new[col].fillna(medians[col])
        else:
            df_new[col] = medians[col]
    
    # Ensure all categorical columns exist; fill with 'unknown' if missing
    for col in categorical_cols:
        if col not in df_new.columns:
            df_new[col] = 'unknown'
    
    # One-hot encode categoricals
    encoded = encoder.transform(df_new[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)
    df_cat = pd.DataFrame(encoded, columns=cat_feature_names, index=df_new.index)
    
    # Drop original categoricals and concatenate
    df_new = df_new.drop(columns=categorical_cols, errors='ignore')
    df_new = pd.concat([df_new, df_cat], axis=1)
    
    # Ensure columns are in the exact order as training
    df_new = df_new.reindex(columns=feature_cols, fill_value=0)
    
    # Scale - pass .values to avoid feature name validation
    scaled = scaler.transform(df_new.values)
    return scaled[0]  # 1D array


def retrieve_similar_applications(app_data):
    """
    Core retrieval logic: given an application dict, return:
        - similar_cases (list of dicts)
        - local_fraud_rate (float)
        - total_neighbors (int)
    Raises ValueError if preprocessing fails.
    """
    try:
        query_vector = preprocess_new_application(app_data)
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {str(e)}")

    current_month = app_data.get('month', 7)

    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT 
            id,
            fraud_bool,
            month,
            metadata,
            1 - (feature_vector <=> %s::vector) AS similarity
        FROM applications
        WHERE month < %s
        ORDER BY feature_vector <=> %s::vector
        LIMIT 20
    """, (query_vector.tolist(), current_month, query_vector.tolist()))
    results = cur.fetchall()
    cur.close()
    conn.close()

    similar_cases = []
    for r in results:
        similar_cases.append({
            "id": r['id'],
            "fraud_bool": r['fraud_bool'],
            "month": r['month'],
            "similarity": float(r['similarity']),
            "metadata": r['metadata']
        })

    fraud_count = sum(1 for c in similar_cases if c['fraud_bool'] == 1)
    total = len(similar_cases)
    local_fraud_rate = fraud_count / total if total > 0 else 0

    return similar_cases, local_fraud_rate, total


# ---------- A2A‑compatible endpoint ----------
@app.route('/agent/retriever', methods=['POST'])
def agent_retriever():
    task = request.json
    if not task or 'id' not in task or 'input' not in task:
        return jsonify({"error": "Invalid task format"}), 400

    task_id = task['id']
    app_data = task['input'].get('query')
    if not app_data:
        return jsonify({"error": "Missing 'query' in task input"}), 400

    try:
        similar_cases, local_fraud_rate, total = retrieve_similar_applications(app_data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    output = {
        "similar_cases": similar_cases,
        "local_fraud_rate": local_fraud_rate,
        "total_neighbors": total,
    }
    try:
        from src.inference.champion_explanation import attach_champion_explanation

        expl = attach_champion_explanation(app_data)
        if expl is not None:
            output["ml_explanation"] = expl
    except Exception as e:
        output["ml_explanation"] = {"shap_available": False, "shap_error": str(e)}

    return jsonify({"task_id": task_id, "output": output})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
