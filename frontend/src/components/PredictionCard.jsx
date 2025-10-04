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
        periodogram: Array.isArray(body.periodogram) ? body.periodogram : [],
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
        periodogram: [],
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
            <div className="mt-6 grid md:grid-cols-2 gap-4 text-sm">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <h4 className="text-white/80 font-semibold text-xs uppercase tracking-wider">
                  Transit Power (Top 3)
                </h4>
                <ul className="mt-2 space-y-1 text-white/80">
                  {result?.vetting?.power?.length
                    ? result.vetting.power.slice(0, 3).map((pow, idx) => (
                        <li key={idx} className="flex justify-between font-mono">
                          <span>P{idx + 1}</span>
                          <span>{pow?.toFixed ? pow.toFixed(2) : Number(pow).toFixed(2)}</span>
                        </li>
                      ))
                    : (
                        <li className="text-white/50 font-mono">No peaks</li>
                      )}
                </ul>
                <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-white/70">
                  <StatMini label="CDPP (ppm)" value={result?.vetting?.cdpp_ppm} />
                  <StatMini label="Flicker (ppm)" value={result?.vetting?.flicker_ppm} />
                  <StatMini label="Noise ρ" value={result?.vetting?.noise_autocorr} fmt={(v) => v.toFixed(3)} />
                </div>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <h4 className="text-white/80 font-semibold text-xs uppercase tracking-wider">
                  BLS Periodogram
                </h4>
                <div className="mt-2 h-32">
                  <PeriodogramPlot data={result?.periodogram} />
                </div>
              </div>
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

function StatMini({ label, value, fmt }) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return (
      <div className="rounded-lg border border-white/5 bg-white/10 px-2 py-1 text-center">
        <div className="text-white/40 text-[10px] uppercase">{label}</div>
        <div className="text-white/60 text-xs">–</div>
      </div>
    );
  }
  const formatted = fmt ? fmt(num) : num.toFixed(1);
  return (
    <div className="rounded-lg border border-white/5 bg-white/10 px-2 py-1 text-center">
      <div className="text-white/40 text-[10px] uppercase">{label}</div>
      <div className="text-white text-xs font-semibold">{formatted}</div>
    </div>
  );
}

function PeriodogramPlot({ data }) {
  const points = Array.isArray(data) ? data : [];
  if (!points.length) {
    return (
      <div className="h-full flex items-center justify-center text-white/40 text-xs">
        No periodogram
      </div>
    );
  }
  const periods = points.map((d) => Number(d[0]));
  const powers = points.map((d) => Number(d[1]));
  const minP = Math.min(...periods);
  const maxP = Math.max(...periods);
  const minPow = Math.min(...powers);
  const maxPow = Math.max(...powers);
  const scaleX = (p) => ((p - minP) / (maxP - minP || 1)) * 100;
  const scaleY = (pow) => 100 - ((pow - minPow) / (maxPow - minPow || 1)) * 100;
  const coords = points.map(
    (pt) => `${scaleX(Number(pt[0])).toFixed(2)},${scaleY(Number(pt[1])).toFixed(2)}`
  );
  const path = coords.map((coord, idx) => `${idx ? "L" : "M"}${coord}`).join(" ");
  return (
    <svg viewBox="0 0 100 100" className="h-full w-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="blsGradient" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="rgba(168,85,247,0.7)" />
          <stop offset="100%" stopColor="rgba(59,130,246,0.2)" />
        </linearGradient>
      </defs>
      <path d={path} fill="none" stroke="url(#blsGradient)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      <polyline
        points={coords.join(" ")}
        fill="none"
        stroke="rgba(168,85,247,0.2)"
        strokeWidth="5"
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity="0.25"
      />
      <g stroke="rgba(255,255,255,0.1)" strokeWidth="0.2">
        {[0, 25, 50, 75, 100].map((y) => (
          <line key={y} x1="0" x2="100" y1={y} y2={y} />
        ))}
      </g>
    </svg>
  );
}
