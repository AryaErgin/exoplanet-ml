import React from "react";
import { toMono, formatPct } from "../lib/utils";

export default function CatalogTable({ items }) {
  const safeItems = Array.isArray(items) ? items : [];

  return (
    <section id="catalog" className="py-10">
      <div className="mx-auto max-w-7xl px-4">
        <div className="flex items-center justify-between">
          <h2 className="text-white text-2xl font-bold mb-4">Catalog</h2>
          <div className="text-white/70 text-sm">
            {safeItems.length} entries
          </div>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-white/10">
          <table className="min-w-full text-sm">
            <thead className="bg-white/10 text-white/80">
              <tr>
                <th className="px-3 py-2 text-left">Run ID</th>
                <th className="px-3 py-2 text-left">File</th>
                <th className="px-3 py-2 text-left">Star</th>
                <th className="px-3 py-2 text-left">Probability</th>
                <th className="px-3 py-2 text-left">Top Dips (days)</th>
                <th className="px-3 py-2 text-left">Timestamp</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-white/10 text-white/80">
              {safeItems.map((it) => {
                const ts =
                  it.createdAt && !isNaN(new Date(it.createdAt))
                    ? new Date(it.createdAt).toLocaleString()
                    : "";
                return (
                  <tr key={toMono(it.id)} className="hover:bg-white/5">
                    <td className="px-3 py-2 font-mono">{toMono(it.id)}</td>
                    <td className="px-3 py-2 font-mono">
                      {toMono(it.filename)}
                    </td>
                    <td className="px-3 py-2 font-mono">{toMono(it.starId)}</td>
                    <td className="px-3 py-2 font-mono">
                      {formatPct(it.probability)}
                    </td>
                    <td className="px-3 py-2 font-mono">
                      {Array.isArray(it.dipsAt)
                        ? it.dipsAt.map((d) => Number(d).toFixed(2)).join(", ")
                        : ""}
                    </td>
                    <td className="px-3 py-2 font-mono">{toMono(ts)}</td>
                  </tr>
                );
              })}

              {safeItems.length === 0 && (
                <tr>
                  <td
                    colSpan={6}
                    className="px-3 py-8 text-center text-white/60"
                  >
                    No runs yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
