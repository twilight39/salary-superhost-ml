# Sync regression notebooks with Jupytext (default)
default:
    uv run jupytext --sync notebooks/regression/*.ipynb

# Run smoke tests
@test:
    uv run pytest tests/ -v

# Train regression pipeline
@train-regression:
    uv run python -m scripts.train_regression

# Train classification pipeline
@train-classification:
    uv run python -m scripts.train_classification

# Sync classification notebooks with Jupytext
@sync-classification:
    uv run jupytext --sync notebooks/classification/*.ipynb
