from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.retriever.A2A import retrieve_similar_applications


@dataclass(frozen=True)
class RetrieverFeatureConfig:
    top_k: int = 20


def _safe_similarity_stats(similar_cases: list[dict[str, Any]]) -> tuple[float, float]:
    if not similar_cases:
        return 0.0, 0.0
    scores = np.array([float(case.get("similarity", 0.0)) for case in similar_cases], dtype=float)
    return float(scores.mean()), float(scores.max())


def build_retriever_features_for_records(records: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich account/application rows with retriever evidence.
    Assumes A2A retriever backend is already loaded/configured.
    """
    feature_rows: list[dict[str, float]] = []
    for _, row in records.iterrows():
        payload = row.to_dict()
        try:
            similar_cases, local_fraud_rate, total_neighbors = retrieve_similar_applications(payload)
        except Exception:
            similar_cases, local_fraud_rate, total_neighbors = [], 0.0, 0
        mean_sim, max_sim = _safe_similarity_stats(similar_cases)
        fraud_neighbors = sum(1 for case in similar_cases if int(case.get("fraud_bool", 0)) == 1)
        feature_rows.append(
            {
                "retr_local_fraud_rate": float(local_fraud_rate),
                "retr_total_neighbors": float(total_neighbors),
                "retr_fraud_neighbors": float(fraud_neighbors),
                "retr_similarity_mean": mean_sim,
                "retr_similarity_max": max_sim,
            }
        )
    return pd.DataFrame(feature_rows, index=records.index)


def retrieve_retriever_evidence(payload: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """
    Retrieve similar cases and return both:
    - summary stats used for modeling
    - top-k similar cases for reporting
    """
    try:
        similar_cases, local_fraud_rate, total_neighbors = retrieve_similar_applications(payload)
    except Exception:
        similar_cases, local_fraud_rate, total_neighbors = [], 0.0, 0
    mean_sim, max_sim = _safe_similarity_stats(similar_cases)
    fraud_neighbors = sum(1 for case in similar_cases if int(case.get("fraud_bool", 0)) == 1)
    top_cases = []
    for case in similar_cases[: max(0, int(top_k))]:
        top_cases.append(
            {
                "id": case.get("id"),
                "fraud_bool": int(case.get("fraud_bool", 0)),
                "month": case.get("month"),
                "similarity": float(case.get("similarity", 0.0)),
            }
        )
    return {
        "features": {
            "retr_local_fraud_rate": float(local_fraud_rate),
            "retr_total_neighbors": float(total_neighbors),
            "retr_fraud_neighbors": float(fraud_neighbors),
            "retr_similarity_mean": float(mean_sim),
            "retr_similarity_max": float(max_sim),
        },
        "top_cases": top_cases,
    }
