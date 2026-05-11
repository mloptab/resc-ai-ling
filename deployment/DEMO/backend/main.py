"""
RESC-AI-LING — FastAPI Backend
Exposes the full inference pipeline as a REST API.

Run with:
    cd DEMO/backend
    uvicorn main:app --reload --port 8000
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

# Ensure backend dir is on path
sys.path.insert(0, os.path.dirname(__file__))

from config import REQUIRED_INPUT_KEYS, GROUP_A_TARGETS, GROUP_A_TARGET_LABELS, SCALE_MAX
from preprocessing import preprocess_input
from predict_group_a import load_group_a_artefacts, predict_group_a
from predict_group_b import load_group_b_artefacts, build_benchmark_input, predict_group_b

# ── Artefacts (loaded once at startup) ───────────────────────────────────
_group_a_artefacts: Dict[str, Any] = {}
_group_b_artefacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artefacts at startup."""
    global _group_a_artefacts, _group_b_artefacts
    print("\n📦  Loading Group A artefacts …")
    _group_a_artefacts = load_group_a_artefacts()
    print("\n📦  Loading Group B artefacts …")
    _group_b_artefacts = load_group_b_artefacts()
    print(f"\n✅  Group A targets: {list(_group_a_artefacts.keys())}")
    print(f"✅  Group B reg:     {list(_group_b_artefacts.get('regressors', {}).keys())}")
    print(f"✅  Group B clf:     {list(_group_b_artefacts.get('classifiers', {}).keys())}")
    print("\n🚀  RESC-AI-LING API ready.\n")
    yield
    print("👋  Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="RESC-AI-LING Inference API",
    description="Predict LLM benchmark scores and architecture from structural features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend from ../frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Pydantic schemas ──────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    training_flops:    float = Field(..., gt=0, description="Total training FLOPs (raw)")
    parameters:        float = Field(..., gt=0, description="Total parameters (raw, e.g. 7e9)")
    dataset_size:      float = Field(..., gt=0, description="Dataset size in tokens/bytes")
    architecture:      str   = Field(..., description="Architecture class name, e.g. Qwen2ForCausalLM")
    model_type:        str   = Field(..., description="chatmodels | pretrained | fine-tuned")
    organization_type: str   = Field(..., description="Industry | Academic | Community")
    override_benchmarks: bool                    = Field(False, description="Use manual benchmark values")
    manual_benchmarks:   Optional[Dict[str, float]] = Field(None, description="Manual benchmark overrides")

    @validator("model_type")
    def validate_model_type(cls, v: str) -> str:
        allowed = {"chatmodels", "pretrained", "fine-tuned"}
        if v not in allowed:
            raise ValueError(f"model_type must be one of {allowed}")
        return v

    @validator("organization_type")
    def validate_org_type(cls, v: str) -> str:
        allowed = {"Industry", "Academic", "Community"}
        if v not in allowed:
            raise ValueError(f"organization_type must be one of {allowed}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "training_flops":    3e23,
                "parameters":        7e9,
                "dataset_size":      1e12,
                "architecture":      "Qwen2ForCausalLM",
                "model_type":        "chatmodels",
                "organization_type": "Industry",
            }
        }


class HealthResponse(BaseModel):
    status: str
    group_a_targets_loaded: int
    group_b_regressors_loaded: int
    group_b_classifiers_loaded: int


# ── Helpers ───────────────────────────────────────────────────────────────
def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None for JSON serialisation."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    return obj


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/", response_class=FileResponse)
async def root():
    """Serve the frontend."""
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "RESC-AI-LING API is running. POST to /predict"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check API and model loading status."""
    return HealthResponse(
        status="ok",
        group_a_targets_loaded=len(_group_a_artefacts),
        group_b_regressors_loaded=len(_group_b_artefacts.get("regressors", {})),
        group_b_classifiers_loaded=len(_group_b_artefacts.get("classifiers", {})),
    )


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Full inference pipeline.

    Flow:
    1. Validate and convert request → user_input dict
    2. preprocess_input()  → engineered DataFrame
    3. predict_group_a()   → 6 benchmark score predictions
    4. build_benchmark_input() → 6-column benchmark row
    5. predict_group_b()   → structural regression + classification
    6. Return unified JSON response

    Returns
    -------
    JSON with keys:
        preprocessed_features, group_a, group_b
    """
    if not _group_a_artefacts and not _group_b_artefacts:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Check that saved_models/ and saved_models_obj3/ exist."
        )

    user_input: Dict[str, Any] = {
        "training_flops":    req.training_flops,
        "parameters":        req.parameters,
        "dataset_size":      req.dataset_size,
        "architecture":      req.architecture,
        "model_type":        req.model_type,
        "organization_type": req.organization_type,
    }

    try:
        # ── Step 1: Preprocessing ────────────────────────────────────────
        processed_df = preprocess_input(user_input)
        preprocessed = processed_df.iloc[0].to_dict()

        # ── Step 2: Group A ──────────────────────────────────────────────
        group_a_raw = predict_group_a(processed_df, _group_a_artefacts)

        # Enrich Group A results with scale_max for gauge rendering
        group_a_response: Dict[str, Any] = {}
        for target, data in group_a_raw.items():
            group_a_response[target] = {
                **data,
                "scale_max": SCALE_MAX.get(target, 100),
            }

        # ── Step 3: Group B ──────────────────────────────────────────────
        benchmark_df = build_benchmark_input(
            group_a_raw,
            override=req.override_benchmarks,
            manual=req.manual_benchmarks,
        )
        group_b_raw = predict_group_b(benchmark_df, _group_b_artefacts)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return _sanitize({
        "preprocessed_features": preprocessed,
        "group_a":               group_a_response,
        "group_b":               group_b_raw,
    })


@app.get("/models/status")
async def models_status():
    """Return detailed status of loaded artefacts."""
    return _sanitize({
        "group_a": {
            target: {
                "models_loaded": list(art.get("models", {}).keys()),
                "features":      art.get("features_used", art.get("features", [])),
            }
            for target, art in _group_a_artefacts.items()
        },
        "group_b": {
            "regressors":  list(_group_b_artefacts.get("regressors",  {}).keys()),
            "classifiers": list(_group_b_artefacts.get("classifiers", {}).keys()),
            "features":    _group_b_artefacts.get("features", []),
        },
    })
