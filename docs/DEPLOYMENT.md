# Deployment & Operations Guide

This guide captures the steps we recommend for consistently rebuilding the
model, validating it, and deploying the stack for demonstrations or judging.

## 1. Bootstrap data locally

```bash
# activate your virtualenv first
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
python research/evaluation_suite.py --features research/work/bootstrap/features.csv --out research/work/eval
```

The evaluation suite exports the comparison models under
`research/work/eval/*.joblib` and a JSON report summarising AUC, precision,
recall, F1 and cross-validation variance. This artifact is referenced in
judging materials to substantiate the model’s performance.

## 3. Deploy the stack with Docker Compose

Build and run the backend, frontend, and (optionally) the training worker in a
single command. The `trainer` profile lets you rebuild the model inside the
same network when required.

```bash
cd deploy
# Launch backend and frontend
docker compose up --build backend frontend

# (Optional) trigger a model rebuild inside the compose network
docker compose --profile training run trainer
```

The backend mounts `backend/app/models/current`, so any new `model.joblib`
and `config.json` written there become instantly available to the running API.
The frontend connects to the backend over the internal Docker network via
`http://backend:8000`.

## 4. Regenerate artifacts for judging

After a successful evaluation run, copy the updated model artifacts into
`backend/app/models/current/` and commit the `research/work/eval/evaluation_report.json`
file so judges can trace the reported metrics back to the generated evidence.

## 5. Troubleshooting

- **NASA archive throttling**: use the bootstrap script to refresh caches
  overnight; the training loop will reuse cached negatives and only fetch new
  IDs that are missing.
- **GPU/CPU constraints**: the HistGradientBoosting pipeline runs efficiently on
  CPU; no GPU acceleration is required.
- **Data drift**: rerun the evaluation suite on each new cache drop and compare
  the JSON reports across commits to document improvements or regressions.
