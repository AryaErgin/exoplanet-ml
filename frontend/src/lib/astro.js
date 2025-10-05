export function phaseFold(time, flux, periodDays, t0) {
  if (
    !Array.isArray(time) ||
    !Array.isArray(flux) ||
    time.length !== flux.length ||
    !isFinite(periodDays) ||
    periodDays <= 0
  ) {
    return { phase: [], f: [] };
  }
  const phase = time.map((t) => ((((t - t0) / periodDays) % 1) + 1) % 1);
  const zipped = phase.map((p, i) => [p, flux[i]]);
  zipped.sort((a, b) => a[0] - b[0]);
  return { phase: zipped.map((z) => z[0]), f: zipped.map((z) => z[1]) };
}
