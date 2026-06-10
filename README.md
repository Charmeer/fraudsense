# fraudsense

A credit card fraud detection project demonstrating end-to-end data analysis, feature engineering, model comparison, explainability, and cost-sensitive threshold selection.

## What this repo contains

- `notebooks/`
  - `01_eda.ipynb` — exploratory data analysis and dataset overview
  - `02_feature_engineering.ipynb` — feature creation and preprocessing
  - `03_model_comparison.ipynb` — training and comparing classifiers
  - `04_shap_explainability.ipynb` — SHAP-based model interpretation
  - `05_cost_threshold.ipynb` — threshold selection based on cost/benefit
- `scripts/generate_powerbi_csvs.py` — script to create CSV reports for Power BI or other dashboards
- `requirements.txt` — Python dependencies for the project
- `.gitignore` — excludes large data/model artifacts and editor artifacts

## Resume-focused highlights

- Focused on fraud detection using transaction data
- Built reproducible analyses and model comparisons
- Included explainability with SHAP analysis
- Incorporated cost-sensitive threshold evaluation to support business decisions

## Notes on repository contents

To keep this repository resume-ready and GitHub-friendly, large raw datasets and model binaries are not included in version control.

The following kinds of files are intentionally excluded:

- large CSV datasets and exported Power BI files
- serialized model artifacts (`*.pkl`, `*.joblib`, etc.)
- generated model/report files
- notebook checkpoint and temporary editor files

## How to use this repo

1. Create a Python environment, for example:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
2. Place the required dataset files in `data/` if you have them locally.
3. Open and run the notebooks in order from `01_eda.ipynb` through `05_cost_threshold.ipynb`.

## If you want a cleaner public repo

This branch is already prepared for resume presentation by keeping only the core notebooks, scripts, and dependency list under version control. Large artifacts are kept locally and excluded from GitHub.
