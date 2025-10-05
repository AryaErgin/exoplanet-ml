import React, { useRef, useState } from "react";
import Papa from "papaparse";
import { FileUp, Sparkles } from "lucide-react";
import {
  classNames,
  isFiniteNum,
  toMono,
  generateSyntheticLightCurve,
} from "../lib/utils";

export default function UploadPanel({ onDataLoaded }) {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState("Idle");

  const handleFiles = (files) => {
    const file = files?.[0];
    if (!file) return;

    const name = file.name.toLowerCase();

    if (name.endsWith(".csv")) {
      setStatus("Parsing CSV…");

      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        complete: (res) => {
          try {
            const rows = (res?.data || []).filter(
              (r) =>
                r &&
                typeof r.time !== "undefined" &&
                typeof r.flux !== "undefined"
            );

            const time = rows.map((r) => Number(r.time)).filter(isFiniteNum);
            const flux = rows.map((r) => Number(r.flux)).filter(isFiniteNum);

            if (time.length && flux.length && time.length === flux.length) {
              onDataLoaded({
                time,
                flux,
                meta: { filename: file.name, rows: time.length },
              });
              setStatus(`Loaded ${time.length} rows`);
            } else {
              setStatus("CSV missing required columns: time, flux");
            }
          } catch {
            setStatus("Failed to parse CSV");
          }
        },
        error: () => setStatus("Failed to parse CSV"),
      });
    } else {
      setStatus("Only CSV supported in demo. FITS support coming.");
    }
  };

  return (
    <section id="upload" className="py-10">
      <div className="mx-auto max-w-7xl px-4">
        <h2 className="text-white text-2xl font-bold mb-4">
          Upload Light Curve
        </h2>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={classNames(
            "rounded-3xl border-2 border-dashed p-10 text-center transition-colors",
            dragOver
              ? "border-blue-400 bg-blue-400/10"
              : "border-white/20 bg-white/5"
          )}
        >
          <div className="flex flex-col items-center justify-center gap-3 text-white/80">
            <FileUp className="h-8 w-8 text-blue-400" />
            <p>
              Drop your <span className="text-white">CSV</span> here or
            </p>

            <button
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-2 rounded-2xl bg-blue-600 hover:bg-blue-500 px-4 py-2 text-sm font-semibold text-white"
            >
              Browse Files
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />

            <p className="text-xs text-white/60">
              Required columns: <code>time</code>, <code>flux</code>
            </p>

            <button
              onClick={() => {
                const demo = generateSyntheticLightCurve(1200, true, 42);
                onDataLoaded({
                  ...demo,
                  meta: {
                    filename: "synthetic_demo.csv",
                    rows: demo.time.length,
                  },
                });
                setStatus(`Loaded synthetic sample (${demo.time.length} rows)`);
              }}
              className="mt-4 inline-flex items-center gap-2 rounded-2xl bg-white/10 hover:bg-white/20 px-4 py-2 text-sm font-semibold text-white"
            >
              <Sparkles className="h-4 w-4" /> Load Synthetic Sample
            </button>
          </div>
        </div>

        <p className="mt-3 text-white/70 text-sm">Status: {toMono(status)}</p>
      </div>
    </section>
  );
}
