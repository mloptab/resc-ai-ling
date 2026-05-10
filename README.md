# RESC-AI-LING: Predicting Language Model Performance

This repository contains a data science project focused on analysing and predicting the performance of Large Language Models (LLMs) from their technical characteristics and benchmark scores.

The project combines information from the Epoch AI database and Open LLM Leaderboard-style benchmark data, builds a cleaned modelling dataset, applies preprocessing and feature engineering, and trains predictive models for benchmark performance and model characteristics.

## Project structure

```text
resc-ai-ling/
├── data/
│   ├── raw/                  # Original source files
│   ├── interim/              # Intermediate merged files
│   └── processed/            # Final cleaned datasets and predictions
├── notebooks/                # Main project notebooks, ordered by execution
├── reports/
│   └── figures/              # Optional exported plots for reports/slides
├── README.md
├── requirements.txt
└── .gitignore
```

## Notebooks

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `01_dataset_integration.ipynb` | Integrates Epoch AI data with benchmark data and creates the base merged dataset. |
| 2 | `02_preprocessing_explained.ipynb` | Cleans the dataset, analyses missing values, imputes data, detects outliers and applies log transformations. |
| 3 | `03_variable_importance.ipynb` | Studies the most relevant variables for benchmark performance using correlation and model-based importance methods. |
| 4 | `04_predicting_benchmarks.ipynb` | Builds regression models to predict benchmark scores from LLM characteristics. |
| 5 | `05_predicting_characteristics.ipynb` | Predicts technical model characteristics from benchmark-related information. |

## Data

The repository separates the datasets into three stages:

- `data/raw/`: original input files.
- `data/interim/`: intermediate files generated during the merging process.
- `data/processed/`: cleaned datasets used for modelling.

Main processed files:

- `final_dataset.xlsx`: integrated dataset before the final preprocessing stage.
- `dataset_clean.xlsx`: cleaned and imputed dataset in the original scale.
- `dataset_clean_log.xlsx`: cleaned dataset with selected logarithmic transformations.
- `model_predictions.xlsx`: exported model prediction results.

## Methodology

The workflow follows five main stages:

1. **Dataset integration**: merge technical model metadata with benchmark performance data.
2. **Data preprocessing**: handle missing values, clean inconsistent fields and prepare numeric variables.
3. **Exploratory analysis**: inspect distributions, outliers and benchmark correlations.
4. **Feature importance analysis**: identify which technical variables are most informative.
5. **Predictive modelling**: train and evaluate machine learning models for benchmark and characteristic prediction.

## How to run the project

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebooks in order from the `notebooks/` folder.

## Main dependencies

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- xgboost
- shap
- joblib
- openpyxl
- jupyter

## Authors

Project developed for academic purposes at Universitat Politècnica de València.

Add the full team member names here.

## Repository link for the presentation

Once uploaded to GitHub, use the main repository URL in the presentation:

```text
https://github.com/YOUR-USERNAME/YOUR-REPOSITORY
```
