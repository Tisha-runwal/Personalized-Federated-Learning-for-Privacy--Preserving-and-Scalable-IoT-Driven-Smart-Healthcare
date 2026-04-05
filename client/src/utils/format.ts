/**
 * Format a number as a percentage string.
 * @param value - Value between 0 and 1 (or 0-100 if already percent)
 * @param decimals - Number of decimal places
 * @param asDecimal - If true, multiply by 100 (value is 0-1 range)
 */
export function formatPercent(value: number | undefined, decimals = 1, asDecimal = true): string {
  if (value === undefined || value === null || isNaN(value)) return '—';
  const pct = asDecimal ? value * 100 : value;
  return `${pct.toFixed(decimals)}%`;
}

/**
 * Format bytes into human-readable size string.
 */
export function formatBytes(bytes: number | undefined): string {
  if (bytes === undefined || bytes === null || isNaN(bytes)) return '—';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(Math.abs(bytes)) / Math.log(1024));
  const clamped = Math.min(i, units.length - 1);
  return `${(bytes / Math.pow(1024, clamped)).toFixed(2)} ${units[clamped]}`;
}

/**
 * Format privacy epsilon value with appropriate precision.
 */
export function formatEpsilon(epsilon: number | undefined): string {
  if (epsilon === undefined || epsilon === null || isNaN(epsilon)) return '—';
  if (epsilon < 0.01) return epsilon.toExponential(2);
  if (epsilon < 10) return epsilon.toFixed(3);
  return epsilon.toFixed(1);
}

/**
 * Returns a Tailwind color class based on accuracy value.
 * Green for high accuracy, yellow for medium, red for low.
 */
export function accuracyColor(accuracy: number | undefined): string {
  if (accuracy === undefined || accuracy === null || isNaN(accuracy)) return 'text-slate-400';
  const pct = accuracy <= 1 ? accuracy * 100 : accuracy;
  if (pct >= 80) return 'text-green-400';
  if (pct >= 60) return 'text-yellow-400';
  return 'text-red-400';
}

/**
 * Format milliseconds into a readable duration string.
 */
export function formatDuration(ms: number | undefined): string {
  if (ms === undefined || ms === null || isNaN(ms)) return '—';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}
