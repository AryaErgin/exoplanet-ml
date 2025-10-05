import React, { useEffect, useState } from "react";
import { generateSyntheticLightCurve, normalizeFlux, movingAverage, detectDips } from "../lib/utils";

export default function HarnessTestSuite() {
	const [log, setLog] = useState([]);

	useEffect(() => {
		const L = [];

		try {
			const demo = generateSyntheticLightCurve(100, true);
			if (!Array.isArray(demo.time) || !Array.isArray(demo.flux))
				throw new Error("Synthetic LC invalid");
			L.push("Synthetic LC ok");

			const nf = normalizeFlux(demo.flux);
			if (nf.length !== demo.flux.length)
				throw new Error("normalizeFlux length mismatch");
			L.push("normalizeFlux ok");

			const sm = movingAverage(nf, 9);
			if (sm.length !== nf.length)
				throw new Error("movingAverage length mismatch");
			L.push("movingAverage ok");

			const res1 = detectDips(demo.time, nf);
			if (!res1 || typeof res1.score !== "number" || !Array.isArray(res1.indices))
				throw new Error("detectDips invalid");
			L.push("detectDips ok");

			const flat = { time: demo.time, flux: demo.time.map(() => 1) };
			const res2 = detectDips(flat.time, normalizeFlux(flat.flux));
			if (res2.score > 0.4)
				throw new Error("flat light curve scored too high");
			L.push("flat baseline ok");
		} catch (e) {
			L.push("Test failure: " + String(e.message || e));
			console.error(e);
		}

		setLog(L);
	}, []);

	return (
		<div className="mx-auto max-w-7xl px-4 pt-2">
			<div className="text-xs text-white/50">Tests: {log.join(" • ")}</div>
		</div>
	);
}

