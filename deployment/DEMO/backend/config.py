"""
RESC-AI-LING — Global Configuration
All constants mirror the notebook exactly. Do not change without updating
the corresponding notebook cells.
"""

from typing import Dict, List

# ── Model artefact paths ──────────────────────────────────────────────────
MODELS_DIR: str = "models/saved_models"
OBJ3_DIR:   str = "models/saved_models_obj3"

# ── Group A ───────────────────────────────────────────────────────────────
GROUP_A_TARGETS: List[str] = [
    "reasoning_score", "musr", "co2", "math", "gpqa", "ifeval"
]

GROUP_A_TARGET_LABELS: Dict[str, str] = {
    "reasoning_score": "Reasoning Score (BBH+MMLU-PRO)",
    "musr":            "MuSR (Multi-step Reasoning)",
    "co2":             "CO₂ Training Cost (log)",
    "math":            "MATH (Logic & Maths)",
    "gpqa":            "GPQA (Expert Knowledge)",
    "ifeval":          "IFEval (Instruction Following)",
}

GROUP_A_MODELS: List[str] = [
    "elasticnet", "randomforest", "gradientboosting",
    "svr", "huberregressor", "bayesianridge",
]

GROUP_A_FEATURES: Dict[str, List[str]] = {
    "reasoning_score": ["log_flops", "log_parameters", "log_dataset_size", "is_qwen2"],
    "musr":            ["log_dataset_size", "log_flops", "is_qwen2", "log_parameters", "is_chat"],
    "math":            ["log_flops", "log_parameters", "log_dataset_size", "is_qwen2", "is_llama"],
    "gpqa":            ["log_flops", "log_parameters", "log_dataset_size", "is_qwen2", "is_chat"],
    "ifeval":          ["is_chat", "log_flops", "log_dataset_size", "is_qwen2", "is_finetune", "is_pretrained"],
    # co2 features loaded from disk
}

# ── Group B ───────────────────────────────────────────────────────────────
GROUP_B_NUM_TARGETS:   List[str] = ["parametros_b", "flops_entrenamiento_num", "tamano_dataset"]
GROUP_B_NUM_LABELS:    List[str] = ["Parameters (B)", "Training FLOPs", "Dataset Size"]
GROUP_B_CAT_TARGETS:   List[str] = ["is_qwen2", "is_llama", "is_chat", "is_finetune", "is_pretrained"]
GROUP_B_CAT_LABELS:    List[str] = ["Arch: Qwen2", "Arch: Llama", "Type: Chat", "Type: Finetune", "Type: Pretrained"]
GROUP_B_BENCHMARKS:    List[str] = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"]

# ── Gauge scale max per Group A target ────────────────────────────────────
SCALE_MAX: Dict[str, float] = {
    "reasoning_score": 70,
    "musr":            25,
    "co2":             25,
    "math":            80,
    "gpqa":            60,
    "ifeval":          90,
}

# ── Required user input keys ──────────────────────────────────────────────
REQUIRED_INPUT_KEYS: List[str] = [
    "training_flops", "parameters", "dataset_size",
    "architecture", "model_type", "organization_type",
]
