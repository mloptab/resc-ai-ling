# RESC-AI-LING · LLM Intelligence Analyzer

> **Demo web application for the RESC-AI-LING inference pipeline.**  
> Electric-blue futuristic dashboard that predicts LLM benchmark scores and architecture characteristics from structural features.

---

## Architecture

```
User Input (browser form)
        │
        ▼  HTTP POST /predict
┌───────────────────┐
│   FastAPI Backend │
│   backend/main.py │
└───────────────────┘
        │
        ├─→ preprocessing.py    preprocess_input()
        ├─→ predict_group_a.py  predict 6 benchmark scores (Objective 2)
        └─→ predict_group_b.py  predict structural characteristics (Objective 3)
                │
                ▼
        JSON response → frontend renders gauges, bars, cards, classification
```

---

## Project Structure

```
DEMO/
├── backend/
│   ├── main.py              FastAPI app — /predict, /health, /models/status
│   ├── config.py            All constants (mirrors notebook exactly)
│   ├── preprocessing.py     preprocess_input()
│   ├── predict_group_a.py   load_group_a_artefacts() + predict_group_a()
│   ├── predict_group_b.py   load_group_b_artefacts() + predict_group_b()
│   └── models/
│       ├── saved_models/         ← Group A artefacts (from Objective 2)
│       └── saved_models_obj3/    ← Group B artefacts (from Objective 3)
│
├── frontend/
│   ├── index.html           Single-page dashboard
│   ├── style.css            Electric blue glassmorphism design
│   └── app.js               Fetch + dynamic rendering
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone / copy the DEMO folder
cd DEMO

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Model Setup

Place your trained artefacts in the correct folders before starting the API:

### Group A (from Objective 2 notebook)
```
backend/models/saved_models/
├── reasoning_score_elasticnet.joblib
├── reasoning_score_randomforest.joblib
├── reasoning_score_gradientboosting.joblib
├── reasoning_score_svr.joblib
├── reasoning_score_huberregressor.joblib
├── reasoning_score_bayesianridge.joblib
├── reasoning_score_scaler.joblib
├── reasoning_score_features.joblib
├── musr_*.joblib         (same pattern)
├── co2_*.joblib
├── math_*.joblib
├── gpqa_*.joblib
└── ifeval_*.joblib
```

### Group B (from Objective 3 notebook)
```
backend/models/saved_models_obj3/
├── obj3_regressor_full.joblib
├── obj3_classifier_full.joblib
├── reg_parametros_b_xgboost.joblib
├── reg_flops_entrenamiento_num_xgboost.joblib
├── reg_tamano_dataset_xgboost.joblib
├── clf_is_qwen2_randomforest.joblib
├── clf_is_llama_randomforest.joblib
├── clf_is_chat_randomforest.joblib
├── clf_is_finetune_randomforest.joblib
├── clf_is_pretrained_randomforest.joblib
├── obj3_features.joblib
├── obj3_targets_num.joblib
├── obj3_targets_num_labels.joblib
├── obj3_targets_cat.joblib
└── obj3_targets_cat_labels.joblib
```

---

## Running the Application

### 1. Start the FastAPI backend

```bash
cd DEMO/backend
uvicorn main:app --reload --port 8000
```

You should see:
```
📦  Loading Group A artefacts …
📦  Loading Group B artefacts …
✅  Group A targets: ['reasoning_score', 'musr', 'co2', 'math', 'gpqa', 'ifeval']
✅  Group B reg:     ['parametros_b', 'flops_entrenamiento_num', 'tamano_dataset']
✅  Group B clf:     ['is_qwen2', 'is_llama', 'is_chat', 'is_finetune', 'is_pretrained']
🚀  RESC-AI-LING API ready.
```

### 2. Open the frontend

Open your browser and go to:
```
http://localhost:8000
```

The FastAPI app serves the frontend automatically from `../frontend/`.

> Alternatively, open `frontend/index.html` directly in a browser (the JS fetches from `http://localhost:8000`).

---

## API Endpoints

| Method | Path | Description |
|:-------|:-----|:------------|
| `GET`  | `/`  | Serves `index.html` |
| `GET`  | `/health` | API + model status |
| `GET`  | `/models/status` | Detailed artefact info |
| `POST` | `/predict` | Full inference pipeline |
| `GET`  | `/docs` | Interactive Swagger UI |

### POST /predict — Request body

```json
{
  "training_flops":    3e23,
  "parameters":        7e9,
  "dataset_size":      1e12,
  "architecture":      "Qwen2ForCausalLM",
  "model_type":        "chatmodels",
  "organization_type": "Industry",
  "override_benchmarks": false
}
```

### POST /predict — Response shape

```json
{
  "preprocessed_features": {
    "log_flops": 54.06,
    "log_parameters": 22.67,
    "is_qwen2": 1,
    ...
  },
  "group_a": {
    "reasoning_score": {
      "predictions": { "elasticnet": 20.58, "randomforest": 10.91, ... },
      "ensemble_mean": 16.46,
      "label": "Reasoning Score (BBH+MMLU-PRO)",
      "scale_max": 70
    },
    ...
  },
  "group_b": {
    "regression": { "Parameters (B)": 66.35, "Training FLOPs": 4.42e24, ... },
    "classification": { "Arch: Qwen2": 1, "Arch: Llama": 0, ... },
    "clf_proba": { "Arch: Qwen2": 0.65, "Arch: Llama": 0.32, ... }
  }
}
```

---

## Frontend → Backend Flow

```
1. User fills form → clicks "Analyze Model"
2. app.js builds JSON payload → POST /predict
3. FastAPI:
   a. Validates input with Pydantic
   b. preprocess_input() → log1p + binary flags
   c. predict_group_a()  → 6 targets × 6 models
   d. build_benchmark_input() → 6-score row from Group A
   e. predict_group_b()  → 3 regressors + 5 classifiers
   f. Returns sanitized JSON
4. app.js renders:
   - Feature tags (preprocessed values)
   - Semicircle gauges (Group A ensemble means)
   - Horizontal bar charts per target (all 6 models)
   - Regression cards with formatted large numbers
   - Classification rows with probability bars
```

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Backend | Python 3.10+ · FastAPI · Uvicorn · Pydantic v2 |
| ML | scikit-learn · XGBoost · joblib · numpy · pandas |
| Frontend | HTML5 · CSS3 (glassmorphism) · Vanilla JS |
| Fonts | Orbitron · Rajdhani · JetBrains Mono |

---

*RESC-AI-LING — LLM Intelligence Analyzer · Demo v1.0*
