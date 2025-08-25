import React, { useRef, useState } from "react";
import {
  normalizeFlux,
  detectDips,
  isFiniteNum,
  toMono,
  classNames,
} from "../lib/utils";
import { postPredict } from "../lib/api";
import { PlayCircle, Gauge as GaugeIcon, Info } from "lucide-react";
import RadialGauge from "./RadialGauge";

export default function PredictionCard({ data, onAddToCatalog, onResult }) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const ctrlRef = useRef();
  const canRun = Boolean(data?.time?.length && data?.flux?.length);

  const runInference = async () => {
    if (!canRun || running) return;
    ctrlRef.current?.abort();
    ctrlRef.current = new AbortController();
    setRunning(true);
    let res;
    try {
      const body = await postPredict(
        { time: data.time, flux: data.flux, meta: data.meta || {} },
        ctrlRef.current.signal
      );
      res = {
        probability: Number(body.probability) || 0,
        dipsAt: Array.isArray(body.dipsAt) ? body.dipsAt : [],
        periodDays: Number(body.periodDays) || null,
        t0: Number(body.t0) || null,
        depthPpm: Number(body.depthPpm) || null,
        durationHr: Number(body.durationHr) || null,
        snr: Number(body.snr) || null,
        topPeriods: Array.isArray(body.topPeriods) ? body.topPeriods : [],
        vetting: body.vetting || {},
        id: body.id,
        starId: `KIC-${Math.floor(Math.random() * 9_000_000)}`,
        filename: data?.meta?.filename || "uploaded.csv",
        rows: data?.meta?.rows || data.time.length,
        createdAt: new Date().toISOString(),
      };
    } catch {
      const nf = normalizeFlux(data.flux);
      const { indices, score } = detectDips(data.time, nf);
      const dipsAt = indices
        .slice(0, 5)
        .map((i) => data.time[i])
        .filter(isFiniteNum);
      res = {
        probability: score,
        dipsAt,
        periodDays: null,
        t0: null,
        depthPpm: null,
        durationHr: null,
        snr: null,
        topPeriods: [],
        vetting: {},
        id: `EV-${Date.now()}`,
        starId: `KIC-${Math.floor(Math.random() * 9_000_000)}`,
        filename: data?.meta?.filename || "uploaded.csv",
        rows: data?.meta?.rows || data.time.length,
        createdAt: new Date().toISOString(),
      };
    }
    setResult(res);
    onAddToCatalog?.(res);
    onResult?.(res);
    setRunning(false);
  };

  const extraCols = Array.isArray(data?.meta?.columns) ? data.meta.columns : [];

  return (
    <section id="results" className="py-10">
      <div className="mx-auto max-w-7xl px-4">
        <h2 className="text-white text-2xl font-bold mb-4">Results</h2>
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 rounded-3xl border border-white/10 bg-gradient-to-b from-white/5 to-white/0 p-6">
            <div className="flex items-center gap-2 text-white/80 text-sm">
              <GaugeIcon className="h-4 w-4" /> Exoplanet Candidate Probability
            </div>
            <div className="mt-4">
              <RadialGauge value={result ? result.probability : 0} />
            </div>
            <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
              <Stat
                label="Period (days)"
                value={result?.periodDays}
                fmt={(v) => (Number.isFinite(v) ? v.toFixed(4) : "–")}
              />
              <Stat
                label="t₀ (days)"
                value={result?.t0}
                fmt={(v) => (Number.isFinite(v) ? v.toFixed(4) : "–")}
              />
              <Stat
                label="Depth (ppm)"
                value={result?.depthPpm}
                fmt={(v) => (Number.isFinite(v) ? Math.round(v) : "–")}
              />
              <Stat
                label="Duration (hr)"
                value={result?.durationHr}
                fmt={(v) => (Number.isFinite(v) ? v.toFixed(2) : "–")}
              />
              <Stat
                label="SNR"
                value={result?.snr}
                fmt={(v) => (Number.isFinite(v) ? v.toFixed(2) : "–")}
              />
              <Stat
                label="Top periods"
                value={
                  result?.topPeriods?.length
                    ? result.topPeriods.map((p) => p.toFixed(3)).join(", ")
                    : "–"
                }
              />
            </div>
            <div className="mt-3 text-white/70 text-sm">
              Top dips (days):{" "}
              <span className="text-white">
                {Array.isArray(result?.dipsAt) && result.dipsAt.length
                  ? result.dipsAt.map((d) => Number(d).toFixed(3)).join(", ")
                  : "–"}
              </span>
            </div>
            <div className="mt-6">
              <button
                disabled={!canRun || running}
                onClick={runInference}
                className={classNames(
                  "inline-flex items-center gap-2 rounded-2xl px-5 py-3 text-sm font-semibold text-white shadow-lg",
                  !canRun || running
                    ? "bg-white/20 cursor-not-allowed"
                    : "bg-purple-600 hover:bg-purple-500"
                )}
              >
                <PlayCircle className="h-5 w-5" />{" "}
                {running ? "Analyzing…" : "Predict"}
              </button>
            </div>
            <p className="mt-2 text-xs text-white/60 flex items-center gap-1">
              <Info className="h-3.5 w-3.5" /> Uses backend if available; falls
              back to local heuristic.
            </p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-white/5 p-6">
            <h3 className="text-white font-semibold">Metadata</h3>
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm text-white/80">
              <div>
                <div className="text-white/60">File</div>
                <div className="font-mono">
                  {toMono(data?.meta?.filename || "–")}
                </div>
              </div>
              <div>
                <div className="text-white/60">Rows</div>
                <div className="font-mono">
                  {toMono(data?.meta?.rows || data?.time?.length || 0)}
                </div>
              </div>
              <div>
                <div className="text-white/60">Star (demo)</div>
                <div className="font-mono">{toMono(result?.starId || "–")}</div>
              </div>
              <div>
                <div className="text-white/60">Run ID</div>
                <div className="font-mono">{toMono(result?.id || "–")}</div>
              </div>
              <div className="col-span-2">
                <div className="text-white/60">Extra columns (CSV)</div>
                <div className="font-mono break-words">
                  {extraCols.length ? extraCols.join(", ") : "None"}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value, fmt }) {
  const v = value == null ? "–" : fmt ? fmt(value) : String(value);
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2">
      <div className="text-white/60 text-xs">{label}</div>
      <div className="text-white font-medium">{v}</div>
    </div>
  );
}
