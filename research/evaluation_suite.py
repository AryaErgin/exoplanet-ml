"""Reproducible evaluation suite for exoplanet classifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from train_from_nasa import MODEL_FEATURES, _transform_frame


def _build_pipelines(random_state: int = 7) -> Dict[str, Pipeline]:
    common = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "hist_gradient_boosting": Pipeline(
            steps=common
            + [
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=8,
                        learning_rate=0.07,
                        max_iter=600,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "random_forest": Pipeline(
            steps=common
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=18,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "logistic_regression": Pipeline(
            steps=common
            + [
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }


def evaluate(features_csv: Path, out_dir: Path) -> Dict:
    df = pd.read_csv(features_csv)
    df = _transform_frame(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("feature table is empty")

    X = df[MODEL_FEATURES]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    pipelines = _build_pipelines()
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() == 2 else None
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        acc = float(accuracy_score(y_test, preds))
        cm = confusion_matrix(y_test, preds).tolist()
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        cv_scores = []
        if y.nunique() == 2:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, test_idx in skf.split(X, y):
                pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
                cv_proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
                cv_scores.append(float(roc_auc_score(y.iloc[test_idx], cv_proba)))

        results[name] = {
            "auc": auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "cv_auc_mean": float(np.mean(cv_scores)) if cv_scores else None,
            "cv_auc_std": float(np.std(cv_scores)) if cv_scores else None,
        }

        model_path = out_dir / f"{name}.joblib"
        pipe.fit(X, y)
        joblib.dump({"model": pipe, "features": MODEL_FEATURES}, model_path)

    summary_path = out_dir / "evaluation_report.json"
    summary_path.write_text(json.dumps({"models": results, "features": MODEL_FEATURES}, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate exoplanet ML pipelines")
    parser.add_argument("--features", type=Path, default=Path("research/work/features_incremental.csv"))
    parser.add_argument("--out", type=Path, default=Path("research/work/eval"))
    args = parser.parse_args()

    results = evaluate(args.features, args.out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
