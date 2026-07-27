/**
 * Client-side cron -> human description. The daemon stores the raw `cron_expr`
 * and computes `next_run`, but ships no human-readable rendering, so the "every
 * 15 min" helper line under a cadence is computed here. Deliberately partial:
 * anything outside the recognized shapes returns `null` so the caller shows the
 * raw expression rather than a guessed (and possibly wrong) phrase.
 */

const DAY_NAMES = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];

const MACROS: Record<string, string> = {
  '@yearly': 'yearly',
  '@annually': 'yearly',
  '@monthly': 'monthly',
  '@weekly': 'weekly',
  '@daily': 'daily 00:00',
  '@midnight': 'daily 00:00',
  '@hourly': 'hourly',
  '@reboot': 'on daemon start',
};

function pad2(n: number): string {
  return String(n).padStart(2, '0');
}

function asInt(field: string): number | null {
  return /^\d+$/.test(field) ? Number(field) : null;
}

function stepOf(field: string): number | null {
  const m = /^\*\/(\d+)$/.exec(field);
  return m ? Number(m[1]) : null;
}

function ordinal(n: number): string {
  const rem100 = n % 100;
  if (rem100 >= 11 && rem100 <= 13) return `${n}th`;
  switch (n % 10) {
    case 1:
      return `${n}st`;
    case 2:
      return `${n}nd`;
    case 3:
      return `${n}rd`;
    default:
      return `${n}th`;
  }
}

/** "hh:mm" when both fields are single integers, else null. */
function timeOf(min: string, hour: string): string | null {
  const m = asInt(min);
  const h = asInt(hour);
  if (m === null || h === null || h > 23 || m > 59) return null;
  return `${pad2(h)}:${pad2(m)}`;
}

function describeDow(dow: string, time: string): string | null {
  if (dow === '1-5') return `weekdays ${time}`;
  if (dow === '6,0' || dow === '0,6' || dow === '6,7') return `weekends ${time}`;
  const d = asInt(dow);
  if (d !== null && d >= 0 && d <= 7) return `${DAY_NAMES[d === 7 ? 0 : d]}s ${time}`;
  return null;
}

/**
 * Human-readable phrase for a 5-field cron expression or a `@macro`, or null
 * when the shape isn't one we describe (caller falls back to the raw expr).
 */
export function describeCron(expr: string | null | undefined): string | null {
  if (!expr) return null;
  const trimmed = expr.trim();
  if (trimmed.startsWith('@')) return MACROS[trimmed.toLowerCase()] ?? null;

  const parts = trimmed.split(/\s+/);
  if (parts.length !== 5) return null;
  const [min, hour, dom, mon, dow] = parts as [string, string, string, string, string];
  const everyMonth = mon === '*';

  // every minute / every N minutes
  if (min === '*' && hour === '*' && dom === '*' && everyMonth && dow === '*')
    return 'every minute';
  const minStep = stepOf(min);
  if (minStep !== null && hour === '*' && dom === '*' && everyMonth && dow === '*') {
    return minStep === 1 ? 'every minute' : `every ${minStep} min`;
  }

  // every N hours (on the minute)
  const hourStep = stepOf(hour);
  if (asInt(min) !== null && hourStep !== null && dom === '*' && everyMonth && dow === '*') {
    return hourStep === 1 ? 'hourly' : `every ${hourStep} hours`;
  }

  // hourly (at :mm)
  if (asInt(min) !== null && hour === '*' && dom === '*' && everyMonth && dow === '*') {
    const m = asInt(min)!;
    return m === 0 ? 'hourly' : `hourly at :${pad2(m)}`;
  }

  const time = timeOf(min, hour);
  if (time === null) return null;

  // daily at hh:mm
  if (dom === '*' && everyMonth && dow === '*') return `daily ${time}`;

  // weekly by day-of-week
  if (dom === '*' && everyMonth && dow !== '*') return describeDow(dow, time);

  // monthly on the Nth
  const domInt = asInt(dom);
  if (domInt !== null && domInt >= 1 && domInt <= 31 && everyMonth && dow === '*') {
    return `${ordinal(domInt)} of the month ${time}`;
  }

  return null;
}
