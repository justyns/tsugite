/**
 * Pure catalog-building logic for the Skills view, kept out of the .svelte
 * file so the join/filter/format rules are directly unit-testable.
 *
 * Two backend endpoints disagree on what "a skill" is:
 *  - GET /api/skill-files lists every *.md file under each skill root
 *    (references/assets docs included), scoped per-agent workspace.
 *  - GET /api/skills/issues (tsugite.tools.skills.get_failed_skills_list)
 *    only considers immediate SKILL.md manifests, via a process-lifetime
 *    singleton SkillManager that scans CWD-relative roots (+ ~/.agents/skills)
 *    once and never rescans - independent of any agent's configured
 *    workspace_dir, so its root set can diverge from skill-files' whenever the
 *    daemon isn't launched from a workspace_dir.
 * buildSkillCatalog() reconciles both: it narrows skill-files down to real
 * manifests, joins issues onto them by path (falling back to name), and never
 * drops an issue that fails to join - an unmatched issue still surfaces as a
 * synthetic row rather than disappearing.
 */
import type { MdFile, SkillIssue } from '$lib/stores/agentsMeta.svelte';

const SKILL_MANIFEST_NAME = 'SKILL.md';

export type SkillIssueSeverity = 'error' | 'warning';

export interface SkillIssueRow {
  name: string | null;
  source: string;
  path: string;
  severity: SkillIssueSeverity;
  message: string;
}

export type SkillStatus = 'ok' | 'warning' | 'error';

export interface SkillCatalogRow {
  path: string;
  name: string;
  description: string;
  /** 'builtin' | 'project' | 'global' from the backend, or '' for a synthetic row. */
  source: string;
  readonly: boolean;
  /** true when this row has no backing /api/skill-files entry (issue-only). */
  synthetic: boolean;
  status: SkillStatus;
  issues: SkillIssueRow[];
}

/** A skill's identity file is SKILL.md exactly (agentskills.io spec) - not any
 * other markdown that happens to live under the same directory tree. */
export function isSkillManifestPath(path: string): boolean {
  const normalized = path.replace(/\\/g, '/');
  const basename = normalized.slice(normalized.lastIndexOf('/') + 1);
  return basename === SKILL_MANIFEST_NAME;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

export function normalizeSkillIssue(raw: SkillIssue): SkillIssueRow {
  const record = raw as Record<string, unknown>;
  const name = asString(record.name);
  return {
    name: name && name !== '?' ? name : null,
    source: asString(record.source, 'scan'),
    path: asString(record.path),
    severity: record.severity === 'error' ? 'error' : 'warning',
    message: asString(record.message, 'unknown issue'),
  };
}

function worstSeverity(issues: SkillIssueRow[]): SkillStatus {
  if (issues.some((issue) => issue.severity === 'error')) return 'error';
  if (issues.length > 0) return 'warning';
  return 'ok';
}

/** Groups orphan issues (no matching file row) by path so one broken skill
 * yields one synthetic row, not one row per warning. Issues with neither a
 * path nor a name collapse onto a single shared key rather than fanning out
 * into an unbounded number of near-duplicate rows. */
function orphanGroupKey(issue: SkillIssueRow): string {
  return issue.path || (issue.name != null ? `name:${issue.name}` : 'unparseable');
}

/** Best-effort display name for an issue with no frontmatter `name` - the
 * containing directory (.../pdf-extract/SKILL.md -> "pdf-extract") reads far
 * better in the name column than a raw absolute path. */
function skillDirName(path: string): string | null {
  const segments = path
    .replace(/\\/g, '/')
    .split('/')
    .filter((segment) => segment.length > 0);
  if (segments.length === 0) return null;
  const last = segments[segments.length - 1]!;
  if (last === SKILL_MANIFEST_NAME && segments.length >= 2) return segments[segments.length - 2]!;
  return last;
}

export function buildSkillCatalog(files: MdFile[], rawIssues: SkillIssue[]): SkillCatalogRow[] {
  const issues = rawIssues.map(normalizeSkillIssue);
  const manifests = files.filter((f) => isSkillManifestPath(f.path));
  const claimed = new Set<number>();

  const rows: SkillCatalogRow[] = manifests.map((file) => {
    const matched = issues.filter((issue, idx) => {
      if (claimed.has(idx)) return false;
      const hit = issue.path === file.path || (issue.name != null && issue.name === file.name);
      if (hit) claimed.add(idx);
      return hit;
    });
    return {
      path: file.path,
      name: file.name,
      description: file.description,
      source: file.source,
      readonly: file.readonly,
      synthetic: false,
      status: worstSeverity(matched),
      issues: matched,
    };
  });

  const orphansByKey = new Map<string, SkillIssueRow[]>();
  issues.forEach((issue, idx) => {
    if (claimed.has(idx)) return;
    const key = orphanGroupKey(issue);
    const group = orphansByKey.get(key);
    if (group) group.push(issue);
    else orphansByKey.set(key, [issue]);
  });

  for (const [key, group] of orphansByKey) {
    const first = group[0]!;
    rows.push({
      path: first.path || key,
      name: first.name ?? skillDirName(first.path) ?? key,
      description: '',
      source: '',
      readonly: true,
      synthetic: true,
      status: worstSeverity(group),
      issues: group,
    });
  }

  return rows.sort((a, b) => a.name.localeCompare(b.name));
}

export function filterSkillCatalog(rows: SkillCatalogRow[], query: string): SkillCatalogRow[] {
  const q = query.trim().toLowerCase();
  if (!q) return rows;
  return rows.filter((row) =>
    [row.name, row.description, row.source].some((field) => field.toLowerCase().includes(q)),
  );
}

export interface CatalogSummary {
  total: number;
  withIssues: number;
}

export function catalogSummary(rows: SkillCatalogRow[]): CatalogSummary {
  return {
    total: rows.length,
    withIssues: rows.filter((row) => row.status !== 'ok').length,
  };
}

/** "ok" | "N warning(s)" | "N error(s)" - errors are counted on their own
 * (dominant) severity even when warnings co-occur on the same row. */
export function skillStatusLabel(row: SkillCatalogRow): string {
  if (row.status === 'ok') return 'ok';
  const count =
    row.issues.filter((issue) => issue.severity === row.status).length || row.issues.length;
  const word = row.status === 'error' ? 'error' : 'warning';
  return `${count} ${word}${count === 1 ? '' : 's'}`;
}

/** "N skill(s)" or "N skill(s) · M with issues" - a single pre-joined string
 * so the template interpolates one expression instead of mixing text nodes
 * with an {#if}, which a reformat can whitespace-collapse (a real bug this
 * once caused: "issues· 3" with the space silently eaten - see git history). */
export function catalogHeading(summary: CatalogSummary): string {
  const base = `${summary.total} skill${summary.total === 1 ? '' : 's'}`;
  return summary.withIssues > 0 ? `${base} · ${summary.withIssues} with issues` : base;
}

/** "issues" or "issues · N" for the drawer's issues section title - see
 * catalogHeading() for why this is a single pre-joined string. */
export function issuesHeading(count: number): string {
  return count > 0 ? `issues · ${count}` : 'issues';
}
