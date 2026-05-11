"""
RESC-AI-LING — Preprocessing Module
Extracted from: Preprocessing Function cell (notebook section 4).

IMPORTANT: Never calls fit() or fit_transform().
           Only applies deterministic transforms identical to training.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List


# ── Engineered feature columns (must match training order exactly) ────────
ENGINEERED_FEATURES: List[str] = [
    "log_flops", "log_parameters", "log_dataset_size",
    "is_qwen2", "is_llama", "is_mixtral",
    "is_chat", "is_finetune", "is_pretrained", "is_industry",
]


def preprocess_input(user_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a raw user-input dict into a fully feature-engineered DataFrame
    compatible with the trained Group A scalers.

    Steps (identical to training pipeline):
    1. Create single-row DataFrame from input.
    2. Apply log1p to large-scale numerical columns (Scaling Laws).
    3. Derive binary architecture flags from the architecture string.
    4. Derive binary model-type flags from model_type.
    5. Derive binary organization flag from organization_type.

    Parameters
    ----------
    user_input : dict
        Must contain keys: training_flops, parameters, dataset_size,
        architecture, model_type, organization_type.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with ENGINEERED_FEATURES columns.

    Raises
    ------
    ValueError
        If any required key is missing from user_input.
    """
    from config import REQUIRED_INPUT_KEYS

    missing = [k for k in REQUIRED_INPUT_KEYS if k not in user_input]
    if missing:
        raise ValueError(f"Missing required input keys: {missing}")

    # ── Step 1: Base row ─────────────────────────────────────────────────
    df = pd.DataFrame([user_input])

    # ── Step 2: Log-transform large-scale numerics ────────────────────────
    #    log1p = log(1+x) for numerical stability at x=0
    df["log_flops"]        = np.log1p(float(user_input["training_flops"]))
    df["log_parameters"]   = np.log1p(float(user_input["parameters"]))
    df["log_dataset_size"] = np.log1p(float(user_input["dataset_size"]))
    df["co2_log"]          = np.nan   # target, not input

    # ── Step 3: Architecture binary flags ────────────────────────────────
    arch = str(user_input.get("architecture", ""))
    df["is_qwen2"]   = int("Qwen2"   in arch)
    df["is_llama"]   = int("Llama"   in arch)
    df["is_mixtral"] = int("Mixtral" in arch)

    # ── Step 4: Model-type binary flags ──────────────────────────────────
    mtype = str(user_input.get("model_type", ""))
    df["is_chat"]       = int(mtype == "chatmodels")
    df["is_finetune"]   = int("fine-tuned" in mtype)
    df["is_pretrained"] = int(mtype == "pretrained")

    # ── Step 5: Organization binary flag ─────────────────────────────────
    org = str(user_input.get("organization_type", ""))
    df["is_industry"] = int(org == "Industry")

    return df[ENGINEERED_FEATURES]
