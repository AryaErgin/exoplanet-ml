import React from "react";
import Plot from "react-plotly.js";

export default function PhaseFoldedView({ phase, flux }) {
  if (!phase?.length || !flux?.length) return null;
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
      <div className="text-white/80 text-sm mb-2">Phase‑folded Light Curve</div>
      <Plot
        data={[
          {
            x: phase,
            y: flux,
            type: "scattergl",
            mode: "markers",
            marker: { size: 3, opacity: 0.7 },
            hovertemplate: "phase=%{x:.3f}<br>flux=%{y:.6f}<extra></extra>",
          },
        ]}
        layout={{
          autosize: true,
          margin: { l: 50, r: 10, t: 10, b: 40 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          xaxis: {
            title: "Phase",
            gridcolor: "rgba(255,255,255,0.08)",
            zeroline: false,
          },
          yaxis: {
            title: "Flux (norm.)",
            gridcolor: "rgba(255,255,255,0.08)",
            zeroline: false,
          },
          font: { color: "white" },
          height: 360,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
