from __future__ import annotations

import logging

import numpy as np
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def resolve_xgb_compute(prefer_gpu: bool = True) -> dict:
    """
    Resolve XGBoost runtime compute settings.

    For XGBoost >=2, use:
      - tree_method='hist'
      - device='cuda' for GPU
    Falls back to CPU if CUDA is unavailable.
    """
    if not prefer_gpu:
        return {"tree_method": "hist", "device": "cpu"}

    try:
        probe = XGBClassifier(
            n_estimators=2,
            max_depth=2,
            learning_rate=0.3,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            device="cuda",
        )
        X = np.array([[0.0], [1.0], [0.5], [0.2]], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.int32)
        probe.fit(X, y, verbose=False)
        logger.info("XGBoost compute mode: CUDA")
        return {"tree_method": "hist", "device": "cuda"}
    except Exception as exc:
        logger.warning("CUDA unavailable for XGBoost, falling back to CPU. Reason: %s", exc)
        return {"tree_method": "hist", "device": "cpu"}
