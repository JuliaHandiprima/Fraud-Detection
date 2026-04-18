import requests
import json

task = {
    "id": "test-001",
    "type": "RAG_RETRIEVE",
    "role": "retriever",
    "input": {
        "query": {
            "month": 6,
            "income": 0.8,
            "source": "INTERNET",
            "device_os": "windows",
            "velocity_4w": 3673.902180917269,
            "velocity_6h": 3732.251936473868,
            "customer_age": 40,
            "payment_type": "AC",
            "velocity_24h": 2554.828090921317,
            "zip_count_4w": 2240,
            "email_is_free": 1,
            "housing_status": "BA",
            "foreign_request": 0,
            "has_other_cards": 0,
            "phone_home_valid": 0,
            "bank_months_count": -1,
            "credit_risk_score": 298,
            "employment_status": "CA",
            "days_since_request": 0.004801971797644,
            "device_fraud_count": 0,
            "keep_alive_session": 0,
            "phone_mobile_valid": 1,
            "bank_branch_count_8w": 0,
            "name_email_similarity": 0.2697318169617812,
            "proposed_credit_limit": 2000.0,
            "intended_balcon_amount": -0.7402695885295941,
            "device_distinct_emails_8w": 1,
            "prev_address_months_count": -1,
            "session_length_in_minutes": 0.9319709906425452,
            "current_address_months_count": 256,
            "date_of_birth_distinct_emails_4w": 7
            } # your application dict
    }   
}


resp = requests.post("http://localhost:5001/agent/retriever", json=task)
print(json.dumps(resp.json(), indent=2))