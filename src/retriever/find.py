"""
find_typical_fraud.py
Scans a random sample of fraudulent applications and computes the local fraud rate
among their nearest neighbors (from strictly earlier months). Prints the IDs with the
highest rates, so you can use them as test queries.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import json

# Database connection parameters (adjust if needed)
DB_PARAMS = {
    "host": "localhost",
    "port": 5433,
    "database": "postgres",
    "user": "postgres",
    "password": "mysecretpassword"
}

# Number of fraudulent applications to sample
SAMPLE_SIZE = 100000
# Number of nearest neighbors to retrieve per query
K = 100

def get_random_fraud_sample(conn, sample_size, min_month=6, max_month=7):
    """Fetch random fraudulent application IDs with month between min_month and max_month."""
    query = """
        SELECT id, month
        FROM applications
        WHERE fraud_bool = 1 AND month BETWEEN %s AND %s
        ORDER BY random()
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (min_month, max_month, sample_size))
        return cur.fetchall()

def get_neighbor_fraud_count(conn, target_id, target_month, target_vector, k=K):
    """
    For a given application, retrieve its k nearest neighbors (from months < target_month)
    and return the count of fraudulent neighbors.
    """
    query = """
        SELECT fraud_bool
        FROM applications
        WHERE month < %s AND id != %s
        ORDER BY feature_vector <=> %s::vector
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (target_month, target_id, target_vector, k))
        neighbors = cur.fetchall()
    fraud_count = sum(1 for n in neighbors if n['fraud_bool'] == 1)
    return fraud_count, len(neighbors)

def main():
    print(f"Connecting to database...")
    conn = psycopg2.connect(**DB_PARAMS)

    print(f"Sampling {SAMPLE_SIZE} fraudulent applications from months 6–7...")
    fraud_samples = get_random_fraud_sample(conn, SAMPLE_SIZE, min_month=6, max_month=7)

    if not fraud_samples:
        print("No fraudulent applications found in the specified month range.")
        return

    results = []

    for idx, row in enumerate(fraud_samples, 1):
        target_id = row['id']
        target_month = row['month']
        print(f"Processing {idx}/{SAMPLE_SIZE}: ID {target_id} (month {target_month})")

        # Retrieve the feature_vector of this application
        with conn.cursor() as cur:
            cur.execute("SELECT feature_vector FROM applications WHERE id = %s", (target_id,))
            vec = cur.fetchone()[0]

        # Get neighbor fraud count
        fraud_count, total = get_neighbor_fraud_count(conn, target_id, target_month, vec, k=K)

        if total > 0:
            rate = fraud_count / total
            results.append({
                'id': target_id,
                'month': target_month,
                'fraud_neighbors': fraud_count,
                'total_neighbors': total,
                'local_fraud_rate': rate
            })
        else:
            print(f"   No neighbors found (month too low?)")

    # Sort by local fraud rate descending
    results_df = pd.DataFrame(results).sort_values('local_fraud_rate', ascending=False)

    print("\n=== Top 10 fraudulent applications with highest local fraud rate ===\n")
    print(results_df.head(10).to_string(index=False))

    # Print the full metadata for the top candidate so you can copy it into test.py
    if not results_df.empty:
        top_id = results_df.iloc[0]['id']
        top_id = int(top_id)  # convert numpy float to plain Python int
        print(f"\nTop candidate ID: {top_id}")
        with conn.cursor() as cur:
            cur.execute("SELECT metadata FROM applications WHERE id = %s", (top_id,))
            metadata = cur.fetchone()[0]
        print("\nMetadata for this application (copy into test.py):")
        print(json.dumps(metadata, indent=2))

    conn.close()

if __name__ == "__main__":
    main()