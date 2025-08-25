export function classNames(...arr) {
  return arr.filter(Boolean).join(" ");
}

export function isFiniteNum(x) {
  return typeof x === "number" && Number.isFinite(x);
}

export function toMono(v) {
  return typeof v === "string" || typeof v === "number" ? String(v) : "";
}

export function formatPct(p) {
  const x = Number(p);
  if (!Number.isFinite(x)) return "--%";
  const c = Math.max(0, Math.min(0.999, x));
  return `${Math.round(c * 100)}%`;
}

export function movingAverage(arr, window = 9) {
  if (!Array.isArray(arr) || arr.length === 0) return [];
  const half = Math.floor(window / 2);

  return arr.map((_, i) => {
    const start = Math.max(0, i - half);
    const end = Math.min(arr.length - 1, i + half);
    let sum = 0,
      n = 0;

    for (let j = start; j <= end; j++) {
      const v = arr[j];
      if (isFiniteNum(v)) {
        sum += v;
        n++;
      }
    }

    return n ? sum / n : 0;
  });
}

export function normalizeFlux(flux) {
  if (!Array.isArray(flux) || flux.length === 0) return [];
  const nums = flux.filter(isFiniteNum);
  if (nums.length === 0) return [];

  const sorted = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const eps = median === 0 ? 1e-9 : median;

  return flux.map((f) => (isFiniteNum(f) ? f / eps : 0));
}

export function detectDips(time, flux) {
  if (
    !Array.isArray(time) ||
    !Array.isArray(flux) ||
    time.length !== flux.length ||
    time.length < 20
  )
    return { indices: [], score: 0 };

  const sm = movingAverage(flux, 11);
  const resid = flux.map((f, i) => f - sm[i]);
  const mean = resid.reduce((a, b) => a + b, 0) / Math.max(1, resid.length);
  const sd =
    Math.sqrt(
      resid.reduce((a, b) => a + (b - mean) ** 2, 0) /
        Math.max(1, resid.length - 1)
    ) || 1e-9;
  const z = resid.map((r) => (r - mean) / (sd || 1e-9));

  const minimaIdx = [];
  for (let i = 1; i < z.length - 1; i++) {
    if (z[i] < z[i - 1] && z[i] < z[i + 1] && z[i] < -2.0) minimaIdx.push(i);
  }

  let score = 0;
  if (minimaIdx.length > 0) {
    const depths = minimaIdx.map((i) => -z[i]);
    const avgDepth = depths.reduce((a, b) => a + b, 0) / depths.length;
    score = Math.min(
      0.99,
      Math.max(0, 0.2 + 0.1 * minimaIdx.length + 0.05 * avgDepth)
    );
  }

  return { indices: minimaIdx, score };
}

export function generateSyntheticLightCurve(
  n = 1200,
  seeded = true,
  seed = 42
) {
  const rng = seeded ? mulberry32(seed) : Math.random;
  const time = Array.from({ length: n }, (_, i) => i * 0.02);
  let flux = time.map(() => 1 + (rng() - 0.5) * 0.01);
  // simple periodic dips
  const p = 3.2,
    depth = 0.04; // 4%
  for (let i = 0; i < n; i++) {
    const phase = (time[i] % p) / p;
    if (phase > 0.49 && phase < 0.51)
      flux[i] -= depth * (1 - Math.abs(phase - 0.5) * 100);
  }
  return { time, flux };
}

function mulberry32(a) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function polar(cx, cy, r, deg) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}
