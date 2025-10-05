# Deployment & Operations Guide

This guide captures the steps we recommend for consistently rebuilding the
model, validating it, and deploying the stack for demonstrations or judging.

## 1. Bootstrap data locally

```bash
# activate your virtualenv first
pip install -r backend/requirements.txt
python research/scripts/bootstrap_training_environment.py --out research/work/bootstrap --count 75
```

The script caches light curves under `research/work/bootstrap/lightcurves/`
and produces a feature table at `research/work/bootstrap/features.csv`. Use
this cache to avoid repeated calls to the NASA Exoplanet Archive when you
iterate on feature engineering.

## 2. Train and evaluate

```bash
# Train on the incremental workflow (resumes if CSV exists)
python research/train_from_nasa.py

# Run the reproducible evaluation suite on the cached features
python research/evaluation_suite.py --features research/work/bootstrap/lightcurves/features.csv --out research/work/eval
```

The evaluation suite exports the comparison models under
`research/work/eval/*.joblib` and a JSON report summarising AUC, precision,
recall, F1 and cross-validation variance. This artifact is referenced in
judging materials to substantiate the model’s performance.

## 3. Deploy Locally

# Frontend

```bash
 cd frontend
 npm run dev
```

# Backend

```bash
 cd backend
 uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend mounts `backend/app/models/current`, so any new `model.joblib`
and `config.json` written there become instantly available to the running API.
The frontend connects to the backend over the internal Docker network via
`http://backend:8000`.
