import React from "react";
import Plot from "react-plotly.js";
import { movingAverage } from "../lib/utils";
import PhaseFoldedView from "./PhaseFoldedView.jsx";
import { phaseFold } from "../lib/astro";

export default function VisualizationPanel({ data, result }) {
  if (!data?.time?.length || !data?.flux?.length) return null;

  const sm = movingAverage(data.flux, 11);

  let phasePack = { phase: [], f: [] };
  if (result?.periodDays && result?.t0 != null) {
    phasePack = phaseFold(data.time, data.flux, result.periodDays, result.t0);
  }

  return (
    <section id="visualization" className="py-10">
      <div className="mx-auto max-w-7xl px-4 grid lg:grid-cols-2 gap-6">
        <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
          <div className="text-white/80 text-sm mb-2">Light Curve</div>
          <Plot
            data={[
              {
                x: data.time,
                y: data.flux,
                type: "scattergl",
                mode: "markers",
                marker: { size: 3, opacity: 0.6 },
                hovertemplate: "t=%{x:.4f}<br>f=%{y:.6f}<extra></extra>",
              },
              {
                x: data.time,
                y: sm,
                type: "scatter",
                mode: "lines",
                line: { width: 1 },
                hovertemplate: "t=%{x:.4f}<br>sm=%{y:.6f}<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              height: 360,
              margin: { l: 50, r: 10, t: 10, b: 40 },
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              xaxis: {
                title: "Time (days)",
                gridcolor: "rgba(255,255,255,0.08)",
              },
              yaxis: {
                title: "Flux (norm.)",
                gridcolor: "rgba(255,255,255,0.08)",
              },
              font: { color: "white" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        </div>

        <PhaseFoldedView phase={phasePack.phase} flux={phasePack.f} />
      </div>
    </section>
  );
}
