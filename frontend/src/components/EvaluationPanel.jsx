import React, { useEffect, useState } from "react";
import { getEvaluationReport } from "../lib/api";
import { toMono } from "../lib/utils";

const METRIC_FIELDS = [
  { key: "auc", label: "AUC" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
  { key: "accuracy", label: "Accuracy" },
  { key: "cv_auc_std", label: "CV σ" },
];

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "–";
  }
  const num = Number(value);
  if (!Number.isFinite(num)) return "–";
  if (num >= 10) return num.toFixed(2);
  return num.toFixed(3);
}

export default function EvaluationPanel() {
  const [state, setState] = useState({ status: "loading", payload: null });

  useEffect(() => {
    const ctrl = new AbortController();
    getEvaluationReport(ctrl.signal)
      .then((payload) => {
        if (!payload || !payload.report) {
          setState({ status: "empty", payload: null });
          return;
        }
        setState({ status: "ready", payload });
      })
      .catch(() => setState({ status: "error", payload: null }));
    return () => ctrl.abort();
  }, []);

  if (state.status === "loading") {
    return (
      <section className="py-10" aria-labelledby="evaluation-heading">
        <div className="mx-auto max-w-7xl px-4 text-white/70 text-sm">
          Loading evaluation metrics…
        </div>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="py-10" aria-labelledby="evaluation-heading">
        <div className="mx-auto max-w-7xl px-4 text-white/70 text-sm">
          Unable to load evaluation metrics from the API.
        </div>
      </section>
    );
  }

  if (state.status === "empty") {
    return (
      <section className="py-10" aria-labelledby="evaluation-heading">
        <div className="mx-auto max-w-7xl px-4 text-white/70 text-sm">
          No evaluation report published yet. Run the evaluation suite to generate
          metrics for judges.
        </div>
      </section>
    );
  }

  const { report, source } = state.payload;
  const models = Object.entries(report.models || {});

  return (
    <section className="py-10" aria-labelledby="evaluation-heading">
      <div className="mx-auto max-w-7xl px-4">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <h2
              id="evaluation-heading"
              className="text-white text-2xl font-bold"
            >
              Evaluation Metrics
            </h2>
            <p className="text-white/70 text-sm mt-1 max-w-2xl">
              Latest benchmark results from the evaluation suite. Metrics are
              computed on an 80/20 split with five-fold cross-validation to show
              both point performance and variance.
            </p>
          </div>
          {source ? (
            <span className="text-white/50 text-xs font-mono">
              {toMono(source)}
            </span>
          ) : null}
        </div>
        <div className="mt-6 overflow-x-auto">
          <table className="min-w-full divide-y divide-white/10 text-sm">
            <thead className="text-white/60 uppercase tracking-wider text-xs">
              <tr>
                <th className="py-3 pr-4 text-left">Model</th>
                {METRIC_FIELDS.map((field) => (
                  <th key={field.key} className="py-3 px-4 text-right">
                    {field.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/10 text-white/80">
              {models.map(([name, metrics]) => (
                <tr key={name}>
                  <td className="py-3 pr-4 font-semibold capitalize">
                    {name.replace(/_/g, " ")}
                  </td>
                  {METRIC_FIELDS.map((field) => (
                    <td key={field.key} className="py-3 px-4 text-right font-mono">
                      {toMono(formatMetric(metrics[field.key]))}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
