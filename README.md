# Machine Learning Portfolio: Salary Prediction & Airbnb Superhost Classification

A comprehensive machine learning application demonstrating regression and classification workflows on real-world datasets with rigorous preprocessing, ensemble methods, and SHAP-based explainability analysis.

## Project Overview

This portfolio addresses two distinct ML problems:

1. **Regression**: Salary prediction using Ask A Manager 2021 survey data (23,384 respondents)
   - Target: Annual salary (USD)
   - Challenge: Severe right-skewness (119.1), extreme outliers
   - Solution: Log transformation, outlier filtering, ensemble methods

2. **Classification**: Airbnb superhost prediction on Singapore listings (13,881 properties)
   - Target: Superhost status (binary)
   - Challenge: 6.53:1 class imbalance, 50% missingness in reviews
   - Solution: SMOTE balancing, temporal forward-filling, XGBoost optimization

## Datasets

| Problem | Source | Size | Features |
|---------|--------|------|----------|
| Regression | [Ask A Manager Salary Survey 2021](https://www.askamanager.org/2021/04/how-much-money-do-you-make-4.html) | 23,384 rows | 14 (engineered to 54) |
| Classification | [Inside Airbnb - Singapore](https://insideairbnb.com/get-the-data/) | 13,881 rows | 77 (engineered to 127) |

## Installation

### Option 1: Using `uv` (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Run Jupyter
uv run jupyter notebook
```

### Option 2: Using pip with pyproject.toml

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from pyproject.toml
pip install -e .

# Run Jupyter
jupyter notebook
```

Navigate to `notebooks/regression/` or `notebooks/classification/` and run notebooks sequentially by number.

## Notebooks

**Regression (8 notebooks)**
- 01: Exploratory Data Analysis
- 02: Data Preprocessing
- 03: Feature Engineering
- 04: Traditional ML
- 05: Deep Learning ML
- 06: Model Tuning
- 07: Model Evaluation
- 08: Model Explainability

**Classification (8 notebooks)**
- 01: Exploratory Data Analysis
- 02: Data Preprocessing
- 03: Feature Engineering
- 04: Traditional ML
- 05: Deep Learning ML
- 06: Model Tuning
- 07: Model Evaluation
- 08: Model Explainability

## Dependencies

scikit-learn, xgboost, torch, imbalanced-learn, shap, polars, numpy, scipy, seaborn, matplotlib, jupyter

See `pyproject.toml` for specific versions.
