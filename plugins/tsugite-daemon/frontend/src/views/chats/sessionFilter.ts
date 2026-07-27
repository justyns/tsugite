/**
 * Sessions-rail filter grammar. Whitespace-separated tokens, modelled on the
 * jobs filter (stores/jobsFilter.ts) so the two search boxes read the same:
 *   agent:smoke     -> restrict to that agent   (repeatable, OR within the axis)
 *   status:failed   -> restrict to that status  (repeatable, OR within the axis)
 *   is:pinned       -> boolean facet: pinned | unread | primary | needs-you
 *   #sess-42        -> free-text term (the '#' is stripped)
 *   anything else   -> free-text term
 * Axes AND together; every free-text term must substring-match the row's
 * title/topic/label/id haystack.
 *
 * The free-text half is also handed to the server (?q=) so a query reaches the
 * full store, not just the ~100 rows the rail holds; the facet half stays a
 * client filter over whatever rows are loaded. Pure + node-testable.
 */

export interface SessionFilter {
  agents: string[];
  statuses: string[];
  flags: string[];
  terms: string[];
}

/** The subset of a session row this grammar reads. */
export interface SessionFilterRow {
  id: string;
  title: string | null;
  topic?: string;
  label?: string;
  agent: string;
  status: string;
  pinned: boolean;
  unread: boolean;
  isPrimary: boolean;
  needsYou: boolean;
}

const FLAG_NAMES = new Set(['pinned', 'unread', 'primary', 'needs-you']);

/** A facet token is `key:value` with a non-empty value; otherwise it's free text. */
function facet(token: string, key: string): string | null {
  if (!token.startsWith(`${key}:`)) return null;
  const value = token.slice(key.length + 1);
  return value.length > 0 ? value.toLowerCase() : null;
}

export function parseSessionFilter(text: string): SessionFilter {
  const agents: string[] = [];
  const statuses: string[] = [];
  const flags: string[] = [];
  const terms: string[] = [];
  for (const token of text.trim().split(/\s+/)) {
    if (!token) continue;
    const agent = facet(token, 'agent');
    const status = facet(token, 'status');
    const flag = facet(token, 'is');
    if (agent !== null) agents.push(agent);
    else if (status !== null) statuses.push(status);
    else if (flag !== null && FLAG_NAMES.has(flag)) flags.push(flag);
    else if (token.startsWith('#') && token.length > 1) terms.push(token.slice(1).toLowerCase());
    else terms.push(token.toLowerCase());
  }
  return { agents, statuses, flags, terms };
}

function haystack(row: SessionFilterRow): string {
  return [row.title, row.topic, row.label, row.id].filter(Boolean).join(' ').toLowerCase();
}

function flagHeld(row: SessionFilterRow, flag: string): boolean {
  switch (flag) {
    case 'pinned':
      return row.pinned;
    case 'unread':
      return row.unread;
    case 'primary':
      return row.isPrimary;
    case 'needs-you':
      return row.needsYou;
    default:
      return false;
  }
}

export function sessionMatchesFilter(row: SessionFilterRow, filter: SessionFilter): boolean {
  if (filter.agents.length && !filter.agents.includes(row.agent.toLowerCase())) return false;
  if (filter.statuses.length && !filter.statuses.includes(row.status.toLowerCase())) return false;
  if (filter.flags.length && !filter.flags.every((f) => flagHeld(row, f))) return false;
  if (filter.terms.length) {
    const hay = haystack(row);
    if (!filter.terms.every((t) => hay.includes(t))) return false;
  }
  return true;
}

/** Whether any facet or term is set (an empty box shows the full grouped list). */
export function isActiveFilter(filter: SessionFilter): boolean {
  return (
    filter.agents.length > 0 ||
    filter.statuses.length > 0 ||
    filter.flags.length > 0 ||
    filter.terms.length > 0
  );
}

/** The free-text portion, rejoined for the server-side ?q= full-store merge. */
export function filterFreeText(filter: SessionFilter): string {
  return filter.terms.join(' ');
}
