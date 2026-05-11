"""
RESC-AI-LING — Group B Prediction Module (Objective 3)
Extracted from: Load Artefacts + Group B Predictions cells.

Takes 6 benchmark scores as input → predicts structural characteristics
(regression) and architecture/type flags (classification).
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Any, Dict, List, Optional

from config import (
    GROUP_B_NUM_TARGETS, GROUP_B_NUM_LABELS,
    GROUP_B_CAT_TARGETS, GROUP_B_CAT_LABELS,
    GROUP_B_BENCHMARKS, OBJ3_DIR,
)


def load_group_b_artefacts(obj3_dir: str = OBJ3_DIR) -> Dict[str, Any]:
    """
    Load all Group B (Objective 3) artefacts.

    Expected files:
        obj3_regressor_full.joblib      — MultiOutputRegressor wrapper
        obj3_classifier_full.joblib     — MultiOutputClassifier wrapper
        reg_{target}_xgboost.joblib     — individual XGBoost per num target
        clf_{target}_randomforest.joblib— individual RF per binary target
        obj3_features.joblib            — BENCHMARKS list
        obj3_targets_num(_labels).joblib
        obj3_targets_cat(_labels).joblib

    Returns
    -------
    dict with keys: regressor_full, classifier_full, regressors,
                    classifiers, features, targets_num, targets_num_labels,
                    targets_cat, targets_cat_labels
    """
    artefacts: Dict[str, Any] = {"regressors": {}, "classifiers": {}}

    def _load(path: str, label: str) -> Any:
        if os.path.exists(path):
            obj = joblib.load(path)
            print(f"  ✅  {label}")
            return obj
        print(f"  ⚠️  Not found: {path}")
        return None

    # Full wrappers
    artefacts["regressor_full"]  = _load(
        os.path.join(obj3_dir, "obj3_regressor_full.joblib"),
        "Group B — MultiOutputRegressor"
    )
    artefacts["classifier_full"] = _load(
        os.path.join(obj3_dir, "obj3_classifier_full.joblib"),
        "Group B — MultiOutputClassifier"
    )

    # Metadata
    for key, fname in [
        ("features",           "obj3_features.joblib"),
        ("targets_num",        "obj3_targets_num.joblib"),
        ("targets_num_labels", "obj3_targets_num_labels.joblib"),
        ("targets_cat",        "obj3_targets_cat.joblib"),
        ("targets_cat_labels", "obj3_targets_cat_labels.joblib"),
    ]:
        val = _load(os.path.join(obj3_dir, fname), f"Metadata — {fname}")
        if val is not None:
            artefacts[key] = val

    # Fallback to config constants if metadata files are missing
    artefacts.setdefault("features",           GROUP_B_BENCHMARKS)
    artefacts.setdefault("targets_num",        GROUP_B_NUM_TARGETS)
    artefacts.setdefault("targets_num_labels", GROUP_B_NUM_LABELS)
    artefacts.setdefault("targets_cat",        GROUP_B_CAT_TARGETS)
    artefacts.setdefault("targets_cat_labels", GROUP_B_CAT_LABELS)

    # Individual regressors (one XGBoost per numerical target)
    for target in artefacts["targets_num"]:
        safe = target.lower().replace(" ", "_")
        mdl  = _load(
            os.path.join(obj3_dir, f"reg_{safe}_xgboost.joblib"),
            f"Regressor — {target}"
        )
        if mdl is not None:
            artefacts["regressors"][target] = mdl

    # Individual classifiers (one RandomForest per binary target)
    for target in artefacts["targets_cat"]:
        safe = target.lower().replace(" ", "_")
        mdl  = _load(
            os.path.join(obj3_dir, f"clf_{safe}_randomforest.joblib"),
            f"Classifier — {target}"
        )
        if mdl is not None:
            artefacts["classifiers"][target] = mdl

    return artefacts


def build_benchmark_input(
    group_a_results: Dict[str, Any],
    override: bool = False,
    manual: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build the 6-column benchmark-score row for Group B.

    If override=True, uses manual values.
    Otherwise derives from Group A ensemble predictions.
    reasoning_score = mean(bbh, mmlu_pro) → approximate both as equal halves.
    """
    if override and manual:
        row = {k: manual.get(k, 0.0) for k in GROUP_B_BENCHMARKS}
        return pd.DataFrame([row])

    reasoning = group_a_results.get("reasoning_score", {}).get("ensemble_mean", 0.0)
    row: Dict[str, float] = {
        "bbh":      round(reasoning, 4),
        "mmlu_pro": round(reasoning, 4),
        "musr":     group_a_results.get("musr",  {}).get("ensemble_mean", 0.0),
        "math":     group_a_results.get("math",  {}).get("ensemble_mean", 0.0),
        "gpqa":     group_a_results.get("gpqa",  {}).get("ensemble_mean", 0.0),
        "ifeval":   group_a_results.get("ifeval",{}).get("ensemble_mean", 0.0),
    }
    return pd.DataFrame([row])[GROUP_B_BENCHMARKS]


def predict_group_b(
    benchmark_df: pd.DataFrame,
    artefacts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all Group B models on the benchmark-score input.

    Mirrors Group A: calls each individual model separately, then uses
    the full wrapper as cross-check/fallback.

    Returns
    -------
    dict
        {
          "regression":     { label: float },
          "reg_per_model":  { target: float },
          "classification": { label: int },
          "clf_per_model":  { target: int },
          "clf_proba":      { label: float },
          "benchmark_input":{ col: float },
        }
    """
    features           = artefacts.get("features",           GROUP_B_BENCHMARKS)
    targets_num        = artefacts.get("targets_num",        GROUP_B_NUM_TARGETS)
    targets_num_labels = artefacts.get("targets_num_labels", GROUP_B_NUM_LABELS)
    targets_cat        = artefacts.get("targets_cat",        GROUP_B_CAT_TARGETS)
    targets_cat_labels = artefacts.get("targets_cat_labels", GROUP_B_CAT_LABELS)

    X = benchmark_df[features].values.astype(float)

    results: Dict[str, Any] = {
        "regression":     {},
        "reg_per_model":  {},
        "classification": {},
        "clf_per_model":  {},
        "clf_proba":      {},
        "benchmark_input": benchmark_df.iloc[0].to_dict(),
    }

    # ── Regression ─────────────────────────────────────────────────────
    for target, label in zip(targets_num, targets_num_labels):
        mdl = artefacts["regressors"].get(target)
        if mdl is not None:
            val = float(mdl.predict(X)[0])
            results["reg_per_model"][target] = round(val, 4)

    # Full wrapper (cross-check / fallback)
    if artefacts.get("regressor_full") is not None:
        full_pred = artefacts["regressor_full"].predict(X)[0]
        results["regression"] = {
            label: round(float(val), 4)
            for label, val in zip(targets_num_labels, full_pred)
        }
    else:
        results["regression"] = {
            label: results["reg_per_model"].get(target, float("nan"))
            for target, label in zip(targets_num, targets_num_labels)
        }

    # ── Classification ─────────────────────────────────────────────────
    clf_probas: Dict[str, float] = {}
    for target, label in zip(targets_cat, targets_cat_labels):
        mdl = artefacts["classifiers"].get(target)
        if mdl is not None:
            pred  = int(mdl.predict(X)[0])
            results["clf_per_model"][target] = pred
            try:
                proba = float(mdl.predict_proba(X)[0][1])
                clf_probas[label] = round(proba, 4)
            except Exception:
                clf_probas[label] = float(pred)

    results["clf_proba"] = clf_probas

    # Full wrapper (cross-check / fallback)
    if artefacts.get("classifier_full") is not None:
        full_clf_pred = artefacts["classifier_full"].predict(X)[0]
        results["classification"] = {
            label: int(val)
            for label, val in zip(targets_cat_labels, full_clf_pred)
        }
        if not clf_probas:
            try:
                probas = [est.predict_proba(X)[0][1]
                          for est in artefacts["classifier_full"].estimators_]
                results["clf_proba"] = {
                    label: round(float(p), 4)
                    for label, p in zip(targets_cat_labels, probas)
                }
            except Exception:
                pass
    else:
        results["classification"] = {
            label: results["clf_per_model"].get(target, 0)
            for target, label in zip(targets_cat, targets_cat_labels)
        }

    return results
