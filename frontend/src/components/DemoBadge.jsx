import React, { useState } from "react";
import RadialGauge from "./RadialGauge";
import { Gauge as GaugeIcon } from "lucide-react";

export default function DemoBadge() {
  const [pct] = useState(0.84);

  return (
    <div className="relative rounded-3xl border border-white/10 bg-white/5 p-6">
      <div className="absolute inset-0 rounded-3xl bg-gradient-to-tr from-blue-600/10 to-purple-600/10" />
      <div className="relative">
        <div className="flex items-center gap-2 text-white/80 text-sm">
          <GaugeIcon className="h-4 w-4" /> Candidate Confidence
        </div>

        <div className="mt-2">
          <RadialGauge value={pct} />
        </div>

        <div className="mt-3 text-xs text-white/60">
          Synthetic demo. Real confidence appears after model inference.
        </div>
      </div>
    </div>
  );
}
