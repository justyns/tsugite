import { describe, expect, test } from 'vitest';
import type { MdFile, SkillIssue } from '$lib/stores/agentsMeta.svelte';
import {
  buildSkillCatalog,
  catalogHeading,
  catalogSummary,
  filterSkillCatalog,
  isSkillManifestPath,
  issuesHeading,
  skillStatusLabel,
} from './skillCatalog';

function file(overrides: Partial<MdFile> = {}): MdFile {
  return {
    path: '/skills/pdf-extract/SKILL.md',
    name: 'pdf-extract',
    source: 'project',
    readonly: false,
    description: 'tables + ocr from pdf attachments',
    ...overrides,
  };
}

function issue(overrides: Partial<SkillIssue> = {}): SkillIssue {
  return {
    name: 'pdf-extract',
    source: 'scan',
    path: '/skills/pdf-extract/SKILL.md',
    severity: 'warning',
    message: "missing 'description' field (recommended)",
    ...overrides,
  };
}

describe('isSkillManifestPath', () => {
  test('accepts a path whose basename is exactly SKILL.md', () => {
    expect(isSkillManifestPath('/a/b/pdf-extract/SKILL.md')).toBe(true);
  });

  test('rejects other markdown files bundled under a skill dir', () => {
    expect(isSkillManifestPath('/a/b/pdf-extract/references/notes.md')).toBe(false);
    expect(isSkillManifestPath('/a/b/README.md')).toBe(false);
  });

  test('is case-sensitive (skill.md is not a manifest)', () => {
    expect(isSkillManifestPath('/a/b/skill.md')).toBe(false);
  });
});

describe('buildSkillCatalog', () => {
  test('a clean skill with no matching issues gets status ok', () => {
    const rows = buildSkillCatalog([file()], []);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ name: 'pdf-extract', status: 'ok', synthetic: false });
  });

  test('filters out non-manifest markdown files from the catalog', () => {
    const rows = buildSkillCatalog(
      [file(), file({ path: '/skills/pdf-extract/references/notes.md', name: 'notes' })],
      [],
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]!.name).toBe('pdf-extract');
  });

  test('joins an issue to its row by exact path match', () => {
    const rows = buildSkillCatalog([file()], [issue()]);
    expect(rows).toHaveLength(1);
    expect(rows[0]!.status).toBe('warning');
    expect(rows[0]!.issues).toHaveLength(1);
  });

  test('joins an issue to its row by name when the path differs', () => {
    // e.g. the file listing resolves symlinks differently than the scanner.
    const rows = buildSkillCatalog(
      [file({ path: '/resolved/pdf-extract/SKILL.md' })],
      [issue({ path: '/unresolved/pdf-extract/SKILL.md' })],
    );
    expect(rows[0]!.issues).toHaveLength(1);
  });

  test('a row with only warnings is status warning; any error wins status error', () => {
    const rows = buildSkillCatalog(
      [file()],
      [issue({ severity: 'warning' }), issue({ severity: 'error', message: 'boom' })],
    );
    expect(rows[0]!.status).toBe('error');
    expect(rows[0]!.issues).toHaveLength(2);
  });

  test('an issue with no matching file row still surfaces as a synthetic row (never silently dropped)', () => {
    const rows = buildSkillCatalog(
      [],
      [issue({ name: 'ghost', path: '/elsewhere/ghost/SKILL.md', severity: 'error' })],
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ name: 'ghost', status: 'error', synthetic: true });
  });

  test('groups multiple orphan issues at the same path into one synthetic row', () => {
    const rows = buildSkillCatalog(
      [],
      [
        issue({ name: null, path: '/elsewhere/broken/SKILL.md', severity: 'error', message: 'a' }),
        issue({ name: null, path: '/elsewhere/broken/SKILL.md', severity: 'error', message: 'b' }),
      ],
    );
    expect(rows).toHaveLength(1);
    expect(rows[0]!.issues).toHaveLength(2);
  });

  test('a nameless orphan falls back to its containing directory, not the raw file path', () => {
    const rows = buildSkillCatalog(
      [],
      [issue({ name: null, path: '/elsewhere/pdf-extract/SKILL.md', severity: 'error' })],
    );
    expect(rows[0]!.name).toBe('pdf-extract');
  });

  test('an issue with no name and no path (unparseable) still produces exactly one row, not one per issue', () => {
    const rows = buildSkillCatalog(
      [],
      [
        issue({ name: null, path: '', severity: 'error' }),
        issue({ name: null, path: '', severity: 'error' }),
      ],
    );
    expect(rows).toHaveLength(1);
  });

  test('sorts rows alphabetically by name', () => {
    const rows = buildSkillCatalog(
      [file({ path: '/z/SKILL.md', name: 'zeta' }), file({ path: '/a/SKILL.md', name: 'alpha' })],
      [],
    );
    expect(rows.map((r) => r.name)).toEqual(['alpha', 'zeta']);
  });

  test('tolerates a malformed issue record defensively (unknown severity, missing message)', () => {
    const rows = buildSkillCatalog(
      [file()],
      [{ path: file().path, severity: 'nonsense' } as unknown as SkillIssue],
    );
    // Unknown severity degrades to warning, not a crash and not silently dropped.
    expect(rows[0]!.status).toBe('warning');
    expect(rows[0]!.issues[0]!.message).toBeTruthy();
  });
});

describe('filterSkillCatalog', () => {
  const rows = buildSkillCatalog(
    [
      file({ path: '/a/SKILL.md', name: 'pdf-extract', description: 'tables + ocr' }),
      file({
        path: '/b/SKILL.md',
        name: 'web-search',
        source: 'builtin',
        description: 'searx + fetch pipeline',
      }),
    ],
    [],
  );

  test('empty query returns every row', () => {
    expect(filterSkillCatalog(rows, '')).toHaveLength(2);
  });

  test('matches case-insensitively on name', () => {
    expect(filterSkillCatalog(rows, 'PDF').map((r) => r.name)).toEqual(['pdf-extract']);
  });

  test('matches on description', () => {
    expect(filterSkillCatalog(rows, 'fetch pipeline').map((r) => r.name)).toEqual(['web-search']);
  });

  test('matches on source', () => {
    expect(filterSkillCatalog(rows, 'builtin').map((r) => r.name)).toEqual(['web-search']);
  });

  test('no match returns an empty list', () => {
    expect(filterSkillCatalog(rows, 'nonexistent')).toHaveLength(0);
  });
});

describe('catalogSummary', () => {
  test('counts total rows and rows with any issue', () => {
    const rows = buildSkillCatalog(
      [file({ path: '/a/SKILL.md', name: 'a' }), file({ path: '/b/SKILL.md', name: 'b' })],
      [issue({ name: 'a', path: '/a/SKILL.md' })],
    );
    expect(catalogSummary(rows)).toEqual({ total: 2, withIssues: 1 });
  });
});

describe('catalogHeading', () => {
  test('singular skill, no issues - no trailing clause', () => {
    expect(catalogHeading({ total: 1, withIssues: 0 })).toBe('1 skill');
  });

  test('plural skills, some with issues - exact spacing around the middot', () => {
    expect(catalogHeading({ total: 13, withIssues: 3 })).toBe('13 skills · 3 with issues');
  });

  test('zero withIssues omits the clause entirely, even with plural skills', () => {
    expect(catalogHeading({ total: 2, withIssues: 0 })).toBe('2 skills');
  });
});

describe('issuesHeading', () => {
  test('zero issues has no count suffix', () => {
    expect(issuesHeading(0)).toBe('issues');
  });

  test('non-zero issues - exact spacing around the middot', () => {
    expect(issuesHeading(3)).toBe('issues · 3');
  });
});

describe('skillStatusLabel', () => {
  test('ok has no count', () => {
    const [row] = buildSkillCatalog([file()], []);
    expect(skillStatusLabel(row!)).toBe('ok');
  });

  test('singular vs plural warning count', () => {
    const [oneWarn] = buildSkillCatalog([file()], [issue({ severity: 'warning' })]);
    expect(skillStatusLabel(oneWarn!)).toBe('1 warning');

    const [twoWarn] = buildSkillCatalog(
      [file()],
      [issue({ severity: 'warning', message: 'a' }), issue({ severity: 'warning', message: 'b' })],
    );
    expect(skillStatusLabel(twoWarn!)).toBe('2 warnings');
  });

  test('error count ignores co-occurring warnings', () => {
    const [row] = buildSkillCatalog(
      [file()],
      [issue({ severity: 'warning' }), issue({ severity: 'error', message: 'e' })],
    );
    expect(skillStatusLabel(row!)).toBe('1 error');
  });
});
