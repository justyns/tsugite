// Pure helpers for the forge pages, kept free of the DOM so they run under node too.

/** The PRs that pass every switched-on filter. `viewers` maps a forge name to the login it reads as. */
export function filterPrs(prs, filters, viewers) {
  return prs.filter(
    (pr) =>
      (!filters.mine || pr.author === viewers[pr.forge]) &&
      (!filters.reviewRequested || pr.requested_reviewers.includes(viewers[pr.forge])) &&
      (!filters.ciFailing || pr.ci_state === 'failure') &&
      (!filters.unresolved || pr.unresolved_threads > 0) &&
      (!filters.label || pr.labels.includes(filters.label)),
  );
}

/** [[`forge:owner/repo`, prs], ...] sorted by key, keeping the PRs' order inside each group. */
export function groupByRepo(prs) {
  const groups = new Map();
  for (const pr of prs) {
    const key = `${pr.forge}:${pr.repo}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(pr);
  }
  return [...groups].sort(([a], [b]) => a.localeCompare(b));
}

export function labelsOf(prs) {
  return [...new Set(prs.flatMap((pr) => pr.labels))].sort();
}

export function duration(seconds) {
  for (const [unit, size] of [['d', 86400], ['h', 3600], ['m', 60]]) {
    if (seconds >= size) return `${Math.floor(seconds / size)}${unit}`;
  }
  return '0m';
}

export function age(iso, now) {
  return duration((now - Date.parse(iso)) / 1000);
}

/** A session's refs split into the visible pinned and touched rows, visible mentions, and hidden rows. */
export function partitionRefs(refs) {
  return {
    main: refs.filter((ref) => !ref.hidden && ref.source !== 'mentioned'),
    mentioned: refs.filter((ref) => !ref.hidden && ref.source === 'mentioned'),
    hidden: refs.filter((ref) => ref.hidden),
  };
}
