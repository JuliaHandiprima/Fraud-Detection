"""
app.py
Flask Retriever Agent – finds similar past cases for a given application ID.
Run after data is loaded and indexes are created.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import joblib

app = Flask(__name__)

# Load preprocessors
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
medians = joblib.load('medians.pkl')
feature_cols = joblib.load('feature_cols.pkl')

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
    df_new = pd.DataFrame([raw_dict])
    
    for col in numerical_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].replace(-1, np.nan)
            df_new[col] = df_new[col].fillna(medians[col])
        else:
            df_new[col] = medians[col]
    
    for col in categorical_cols:
        if col not in df_new.columns:
            df_new[col] = 'unknown'
    
    encoded = encoder.transform(df_new[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)
    df_cat = pd.DataFrame(encoded, columns=cat_feature_names, index=df_new.index)
    
    df_new = df_new.drop(columns=categorical_cols, errors='ignore')
    df_new = pd.concat([df_new, df_cat], axis=1)
    df_new = df_new.reindex(columns=feature_cols, fill_value=0)
    scaled = scaler.transform(df_new.values)
    return scaled[0]

def get_application_by_id(app_id):
    """Fetch metadata and month for a given application ID."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT metadata, month FROM applications WHERE id = %s", (app_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise ValueError(f"Application with id {app_id} not found")
    return row['metadata'], row['month']

@app.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.json
    if not data or 'application_id' not in data:
        return jsonify({"error": "Missing 'application_id' in request"}), 400
    
    app_id = data['application_id']
    
    try:
        # Fetch the full feature dictionary and month for this ID
        app_data, current_month = get_application_by_id(app_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
    # Preprocess the fetched features
    try:
        query_vector = preprocess_new_application(app_data)
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {str(e)}"}), 400
    
    # Query similar cases (excluding the query ID itself, only earlier months)
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
        WHERE month < %s AND id != %s
        ORDER BY feature_vector <=> %s::vector
        LIMIT 20
    """, (query_vector.tolist(), current_month, app_id, query_vector.tolist()))
    
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
    local_fraud_rate = fraud_count / len(similar_cases) if similar_cases else 0
    
    response = {
        "similar_cases": similar_cases,
        "local_fraud_rate": local_fraud_rate,
        "total_neighbors": len(similar_cases),
        "query_metadata": app_data,        # <-- NEW: full feature dict of the queried application
        "query_month": current_month       # <-- NEW: month of the queried application
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)