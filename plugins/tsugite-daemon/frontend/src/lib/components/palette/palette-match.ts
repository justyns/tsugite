// Fuzzy match + grouping engine for the command palette. Pure, framework-free,
// and unit-tested (palette-match.test.ts) so the Svelte component stays a thin
// view over this model.

export interface PaletteItem {
  /** Grouping bucket, shown as an uppercase header when the query is empty. */
  group: string;
  /** Icon name (see Palette.svelte's ICON_PATHS). */
  icon: string;
  /** Primary text that is matched against and rendered. */
  label: string;
  /** Trailing meta (status, age, kind) shown right-aligned. */
  meta: string;
  /** Navigation target for jump items. */
  href?: string;
  /** Marks a quick-action row rather than a jump target. */
  action?: boolean;
  /** Extra matchable text that is never highlighted or shown - e.g. a session's topic. */
  keywords?: string;
}

export interface MatchResult {
  score: number;
  /** `[start, end)` of the highlighted run for contiguous matches, else null. */
  highlight: [number, number] | null;
}

export type PaletteRow =
  | { kind: 'group'; label: string }
  | { kind: 'item'; item: PaletteItem; index: number; highlight: [number, number] | null };

/** Empty query lists a scannable slice of everything, grouped. */
export const MAX_RESULTS_EMPTY = 14;
/** A query flattens and ranks; keep the top slice tight. */
export const MAX_RESULTS_QUERY = 12;
/** Sessions fold into a query as their own trailing group; cap to the most recent. */
export const MAX_SESSION_RESULTS = 8;

/**
 * Score `text` against `query`. A contiguous substring wins (score by earliness,
 * highlight the run); otherwise an in-order subsequence is a weak match with no
 * highlight; no subsequence returns null. Case-insensitive.
 */
export function matchItem(query: string, text: string): MatchResult | null {
  if (!query) return { score: 0, highlight: null };
  const q = query.toLowerCase();
  const s = text.toLowerCase();

  const i = s.indexOf(q);
  if (i > -1) return { score: 100 - i, highlight: [i, i + q.length] };

  let from = 0;
  let hits = 0;
  for (const ch of q) {
    const f = s.indexOf(ch, from);
    if (f < 0) return null;
    from = f + 1;
    hits++;
  }
  return { score: hits * 2, highlight: null };
}

/**
 * Score one item against a non-empty, lowercased query. A label (title) hit
 * carries its highlight run; failing that, a plain substring hit on meta / group
 * / keywords filters the item in with no highlight (fuzzy subsequences there
 * would false-positive across unrelated items). No hit → null.
 */
function scoreItem(q: string, item: PaletteItem): MatchResult | null {
  const onLabel = matchItem(q, item.label);
  if (onLabel) return onLabel;
  const meta = `${item.meta ?? ''}`.toLowerCase();
  const group = `${item.group ?? ''}`.toLowerCase();
  const keywords = `${item.keywords ?? ''}`.toLowerCase();
  return meta.includes(q) || group.includes(q) || keywords.includes(q)
    ? { score: 0, highlight: null }
    : null;
}

/** Rank a pool by score (desc), stable on source order, capped to `limit`. */
function rankMatches(
  items: PaletteItem[],
  q: string,
  limit: number,
): { item: PaletteItem; highlight: [number, number] | null }[] {
  return items
    .map((item, order) => ({ item, order, match: scoreItem(q, item) }))
    .filter((m): m is typeof m & { match: MatchResult } => m.match !== null)
    .sort((a, b) => b.match.score - a.match.score || a.order - b.order)
    .slice(0, limit)
    .map((m) => ({ item: m.item, highlight: m.match.highlight }));
}

/**
 * Build the render model. Empty/whitespace query → `items` in source order with a
 * header before each group run, capped at MAX_RESULTS_EMPTY (sessions excluded -
 * they never crowd the default list). Non-empty query → `items` matches ranked by
 * score (stable on ties), no headers, capped at MAX_RESULTS_QUERY, then any
 * matching `sessions` under their own trailing header, capped at
 * MAX_SESSION_RESULTS in the order given (live-first upstream). Item rows carry a
 * contiguous selectable `index` across both groups.
 */
export function buildRows(
  items: PaletteItem[],
  query: string,
  sessions: PaletteItem[] = [],
): PaletteRow[] {
  const q = query.trim().toLowerCase();

  if (!q) {
    const rows: PaletteRow[] = [];
    let group = '';
    let index = 0;
    for (const item of items.slice(0, MAX_RESULTS_EMPTY)) {
      if (item.group !== group) {
        group = item.group;
        rows.push({ kind: 'group', label: group });
      }
      rows.push({ kind: 'item', item, index: index++, highlight: null });
    }
    return rows;
  }

  const rows: PaletteRow[] = [];
  let index = 0;
  for (const m of rankMatches(items, q, MAX_RESULTS_QUERY)) {
    rows.push({ kind: 'item', item: m.item, index: index++, highlight: m.highlight });
  }

  const sessionMatches = rankMatches(sessions, q, MAX_SESSION_RESULTS);
  if (sessionMatches.length) {
    rows.push({ kind: 'group', label: 'sessions' });
    for (const m of sessionMatches) {
      rows.push({ kind: 'item', item: m.item, index: index++, highlight: m.highlight });
    }
  }

  return rows;
}
