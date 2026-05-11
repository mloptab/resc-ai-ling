"""
RESC-AI-LING — Group A Prediction Module (Objective 2)
Extracted from: Load Artefacts + Group A Predictions cells.

Loads 6 regressors × 6 targets and predicts benchmark scores from
structural/architectural features.
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Any, Dict, List
from sklearn.preprocessing import StandardScaler

from config import (
    GROUP_A_TARGETS, GROUP_A_TARGET_LABELS,
    GROUP_A_MODELS, MODELS_DIR,
)


def load_group_a_artefacts(models_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """
    Load all Group A (Objective 2) scalers, feature lists and models.

    Expected files per target:
        {target}_scaler.joblib
        {target}_features.joblib
        {target}_{model_name}.joblib  (one per model in GROUP_A_MODELS)

    Returns
    -------
    dict
        artefacts[target_name] = {
            "scaler":   StandardScaler,
            "features": List[str],
            "models":   { model_name: fitted_estimator }
        }
    """
    artefacts: Dict[str, Any] = {}

    for target in GROUP_A_TARGETS:
        scaler_path   = os.path.join(models_dir, f"{target}_scaler.joblib")
        features_path = os.path.join(models_dir, f"{target}_features.joblib")

        if not os.path.exists(scaler_path):
            print(f"  ⚠️  Scaler not found for '{target}' — skipping.")
            continue
        if not os.path.exists(features_path):
            print(f"  ⚠️  Features not found for '{target}' — skipping.")
            continue

        scaler:   StandardScaler = joblib.load(scaler_path)
        features: List[str]      = joblib.load(features_path)
        models:   Dict[str, Any] = {}

        for mname in GROUP_A_MODELS:
            model_path = os.path.join(models_dir, f"{target}_{mname}.joblib")
            if os.path.exists(model_path):
                models[mname] = joblib.load(model_path)
            else:
                print(f"  ⚠️  Model not found: {target}/{mname}")

        artefacts[target] = {"scaler": scaler, "features": features, "models": models}
        print(f"  ✅  Group A '{target}': {len(models)} models, features={features}")

    return artefacts


def predict_group_a(
    processed_df: pd.DataFrame,
    artefacts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all Group A models on the pre-processed input.

    Each target has its own scaler and feature subset.
    Applies scaler.transform() only — never fit().

    Parameters
    ----------
    processed_df : pd.DataFrame
        Output of preprocess_input().
    artefacts : dict
        Output of load_group_a_artefacts().

    Returns
    -------
    dict
        {
          target_name: {
            "predictions":   { model_name: float },
            "ensemble_mean": float,
            "features_used": List[str],
            "label":         str,
          }
        }
    """
    results: Dict[str, Any] = {}

    for target, art in artefacts.items():
        scaler:   StandardScaler = art["scaler"]
        features: List[str]      = art["features"]
        models:   Dict           = art["models"]

        try:
            X_raw = processed_df[features].values.astype(float)
        except KeyError as exc:
            print(f"  ⚠️  Missing features for target '{target}': {exc}")
            continue

        # transform only — never fit
        X_scaled = scaler.transform(X_raw)

        preds: Dict[str, float] = {}
        for mname, model in models.items():
            try:
                pred = float(model.predict(X_scaled)[0])
                preds[mname] = round(pred, 4)
            except Exception as exc:
                print(f"  ⚠️  Prediction failed {target}/{mname}: {exc}")

        ensemble_mean = round(float(np.mean(list(preds.values()))), 4) if preds else float("nan")

        results[target] = {
            "predictions":   preds,
            "ensemble_mean": ensemble_mean,
            "features_used": features,
            "label":         GROUP_A_TARGET_LABELS.get(target, target),
        }

    return results
