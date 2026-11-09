# Fetal Health Classification

## Problem and intended use

Preventable maternal and child deaths remain a global health concern. Cardiotocography (CTG) is a relatively accessible way to monitor fetal heart rate, fetal movement, uterine contractions, and related signals. Interpreting those measurements can help healthcare professionals identify cases that may need further attention.

This project uses machine-learning classification to demonstrate how CTG measurements can be used to predict one of three fetal-health labels: **Normal**, **Suspect**, or **Pathological**. The trained model is exposed through a FastAPI service: callers provide seven CTG features and receive a predicted class and class probabilities.

This repository is an educational and experimental prototype, not a medical device or a substitute for clinical judgment. Its predictions must not be used by themselves to make clinical decisions.

## Dataset

The dataset contains **2,126 records** of features extracted from CTG exams. According to the dataset description, three expert obstetricians classified the records into Normal, Suspect, and Pathological classes.

Source: [Fetal Health Classification dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data).

Download the dataset and place the CSV at `data/fetal_health.csv`. The training script expects that exact path when run from the repository root.

## Setup with `uv`

The project uses [`uv`](https://docs.astral.sh/uv/) to manage its Python environment and locked dependencies. Install `uv`, then from the repository root run:

```bash
uv sync
```

This creates or updates the project virtual environment using `pyproject.toml` and `uv.lock`. Run project commands with `uv run` so they use that environment.

## Train and save the model

From the repository root, run:

```bash
uv run python scripts/train.py
```

The script loads `data/fetal_health.csv`, evaluates the XGBoost pipeline using stratified five-fold cross-validation (reporting one-vs-rest ROC AUC), then fits the pipeline on the full dataset. It saves the serialized model to:

```text
models/xgboost_pipeline.joblib
```

The saved model is required to run the API or build the Docker image.

## Run the API locally

Start the FastAPI server from the repository root:

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 9696
```

The API provides:

- `GET /health` — reports whether the model loaded.
- `POST /predict` — accepts CTG measurements and returns a predicted class and probabilities.

Interactive API documentation is available at [http://localhost:9696/docs](http://localhost:9696/docs).

## Make a prediction with `predict.py`

With the API running, open a second terminal at the repository root and run:

```bash
uv run python scripts/predict.py
```

This sends the example measurements in `scripts/predict.py` to `http://localhost:9696/predict` and prints the JSON response.

You can also call the endpoint directly:

```bash
curl -X POST http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "prolongued_decelerations": 0.002,
    "accelerations": 0.000,
    "uterine_contractions": 0.007,
    "light_decelerations": 0.006,
    "fetal_movement": 0.001,
    "severe_decelerations": 0.000,
    "baseline_value": 129.0
  }'
```

The endpoint responds with `fetal_health_class` (`1`: Normal, `2`: Suspect, `3`: Pathological) and probabilities for each class.

## Run with Docker

First train the model as described above so `models/xgboost_pipeline.joblib` exists. From the repository root, build and run the image:

```bash
docker build -t fetal-health-classification .
docker run --rm -p 9696:9696 fetal-health-classification
```

The container serves the same API on port `9696`. Once it is running, visit [http://localhost:9696/docs](http://localhost:9696/docs) or use `scripts/predict.py` from the host in another terminal.
