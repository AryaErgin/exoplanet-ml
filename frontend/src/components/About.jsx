import React from "react";

export default function About() {
  return (
    <section id="about" className="py-10">
      <div className="mx-auto max-w-7xl px-4">
        <h2 className="text-white text-2xl font-bold mb-4">About</h2>
        <div className="rounded-3xl border border-white/10 bg-white/5 p-6 text-white/80 leading-relaxed">
          <p>
            ExoVision AI is a research-grade interface for rapid exoplanet
            candidate screening. It ingests time-series light curves, applies
            denoising and dip detection, and presents explainable confidence
            metrics with interactive visualizations.
          </p>
          <p className="mt-3">
            Built for NASA Space Apps 2025. Data remains local in this demo. For
            production, connect a FastAPI/Flask backend at
            <code className="bg-black/40 px-1 rounded">/predict</code>.
          </p>
        </div>
      </div>
    </section>
  );
}
