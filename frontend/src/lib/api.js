const BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function postPredict(payload, signal) {
  const r = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

export async function getRuns(signal) {
  const r = await fetch(`${BASE}/runs`, { signal });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

export async function getHealth(signal) {
  const r = await fetch(`${BASE}/health`, { signal });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

export async function getEvaluationReport(signal) {
  const r = await fetch(`${BASE}/evaluation/report`, { signal });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}
