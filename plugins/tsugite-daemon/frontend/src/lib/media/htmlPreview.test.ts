import { describe, expect, test } from 'vitest';
import {
  HTML_CSP,
  HTML_SANDBOX,
  buildSrcdoc,
  collectAssetRefs,
  inlineAssets,
  isHtml,
  loadInlineAssets,
  resolveWorkspaceAsset,
  type InlinedAsset,
} from './htmlPreview';

describe('isHtml', () => {
  test('matches .html and .htm, case-insensitively', () => {
    expect(isHtml('report.html')).toBe(true);
    expect(isHtml('report.htm')).toBe(true);
    expect(isHtml('REPORT.HTML')).toBe(true);
    expect(isHtml('a/b/c/index.html')).toBe(true);
  });

  test('rejects look-alikes and other extensions', () => {
    expect(isHtml('notes.md')).toBe(false);
    expect(isHtml('template.html.j2')).toBe(false);
    expect(isHtml('html')).toBe(false);
    expect(isHtml('x.xhtml')).toBe(false);
  });
});

describe('isolation policy constants', () => {
  test('the sandbox grants nothing at all', () => {
    // An empty token list denies every sandbox capability, including
    // allow-scripts and allow-same-origin. Loosening this needs a very good
    // reason: allow-same-origin alone would hand the frame the bearer token.
    expect(HTML_SANDBOX).toBe('');
  });

  test('the CSP blocks every network source', () => {
    expect(HTML_CSP).toContain("default-src 'none'");
    expect(HTML_CSP).not.toMatch(/https?:/);
    expect(HTML_CSP).toContain("base-uri 'none'");
    // No script source is carved out of default-src 'none'.
    expect(HTML_CSP).not.toContain('script-src');
  });
});

describe('buildSrcdoc', () => {
  const meta = `<meta http-equiv="Content-Security-Policy" content="${HTML_CSP}">`;

  test('wraps the document in a head carrying the policy', () => {
    expect(buildSrcdoc('<h1>report</h1>')).toBe(
      `<!doctype html><html><head>${meta}</head><body><h1>report</h1></body></html>`,
    );
  });

  test('carries the document through untouched, its own head included', () => {
    const doc =
      '<html><head><link rel="stylesheet" href="a.css"></head><body><p>hi</p></body></html>';
    expect(buildSrcdoc(doc)).toBe(
      `<!doctype html><html><head>${meta}</head><body>${doc}</body></html>`,
    );
  });
});

describe('collectAssetRefs', () => {
  test('finds stylesheet links and images, in order and de-duplicated', () => {
    const html = `
      <link rel="stylesheet" href="style.css">
      <link rel="icon" href="favicon.ico">
      <img src="chart.png" alt="chart">
      <img src='chart.png'>
      <link rel="STYLESHEET" href="print.css" />
    `;
    expect(collectAssetRefs(html)).toEqual([
      { href: 'style.css', kind: 'style' },
      { href: 'chart.png', kind: 'image' },
      { href: 'print.css', kind: 'style' },
    ]);
  });

  test('ignores non-stylesheet links and reference-less tags', () => {
    expect(collectAssetRefs('<link rel="preload" href="x.css"><img><link href="y.css">')).toEqual(
      [],
    );
  });
});

describe('resolveWorkspaceAsset', () => {
  test('resolves against the document directory', () => {
    expect(resolveWorkspaceAsset('reports/cov/index.html', 'style.css')).toBe(
      'reports/cov/style.css',
    );
    expect(resolveWorkspaceAsset('reports/cov/index.html', './a/b.png')).toBe(
      'reports/cov/a/b.png',
    );
    expect(resolveWorkspaceAsset('reports/cov/index.html', '../shared.css')).toBe(
      'reports/shared.css',
    );
    expect(resolveWorkspaceAsset('index.html', 'style.css')).toBe('style.css');
  });

  test('reads a leading slash as workspace-root-relative', () => {
    expect(resolveWorkspaceAsset('reports/cov/index.html', '/assets/app.css')).toBe(
      'assets/app.css',
    );
  });

  test('refuses anything that is not a workspace file', () => {
    const doc = 'reports/index.html';
    expect(resolveWorkspaceAsset(doc, 'https://cdn.example.com/a.css')).toBeNull();
    expect(resolveWorkspaceAsset(doc, '//cdn.example.com/a.css')).toBeNull();
    expect(resolveWorkspaceAsset(doc, 'data:text/css,body{}')).toBeNull();
    expect(resolveWorkspaceAsset(doc, 'javascript:alert(1)')).toBeNull();
    expect(resolveWorkspaceAsset(doc, '#anchor')).toBeNull();
    expect(resolveWorkspaceAsset(doc, '')).toBeNull();
  });

  test('refuses a path that climbs out of the workspace', () => {
    expect(resolveWorkspaceAsset('reports/index.html', '../../etc/passwd')).toBeNull();
    expect(resolveWorkspaceAsset('index.html', '../secret.css')).toBeNull();
    expect(resolveWorkspaceAsset('a/b/index.html', '../../../../x.css')).toBeNull();
  });

  test('drops the query and fragment before resolving', () => {
    expect(resolveWorkspaceAsset('index.html', 'style.css?v=3#top')).toBe('style.css');
  });
});

describe('inlineAssets', () => {
  const resolved = new Map<string, InlinedAsset>([
    ['a.css', { kind: 'style', css: 'body{color:red}' }],
    ['chart.png', { kind: 'image', dataUri: 'data:image/png;base64,AAA' }],
  ]);

  test('swaps a stylesheet link for an inline style block', () => {
    const out = inlineAssets('<head><link rel="stylesheet" href="a.css"></head>', resolved);
    expect(out).toBe('<head><style>body{color:red}</style></head>');
  });

  test('rewrites an image src to the data URI, leaving its other attributes alone', () => {
    const out = inlineAssets('<img src="chart.png" alt="a &amp; b" width="20">', resolved);
    expect(out).toBe('<img src="data:image/png;base64,AAA" alt="a &amp; b" width="20">');
  });

  test('leaves an unresolved reference untouched', () => {
    const html = '<img src="https://cdn.example.com/x.png"><link rel="stylesheet" href="b.css">';
    expect(inlineAssets(html, resolved)).toBe(html);
  });

  test('neutralizes a </style> sequence hiding inside inlined CSS', () => {
    const nasty = new Map<string, InlinedAsset>([
      ['a.css', { kind: 'style', css: 'body{}</style><script>steal()</script>' }],
    ]);
    const out = inlineAssets('<link rel="stylesheet" href="a.css">', nasty);
    expect(out).not.toContain('</style><script>');
    expect(out).toContain('<\\/style>');
  });
});

describe('loadInlineAssets', () => {
  function spyReaders(fail?: string) {
    const asked: string[] = [];
    return {
      asked,
      readers: {
        readText: async (path: string) => {
          asked.push(path);
          if (path === fail) throw new Error('nope');
          return `/* ${path} */`;
        },
        readDataUri: async (path: string) => {
          asked.push(path);
          if (path === fail) throw new Error('nope');
          return `data:image/png;base64,${path}`;
        },
      },
    };
  }

  const html = [
    '<link rel="stylesheet" href="style.css">',
    '<link rel="stylesheet" href="https://cdn.example.com/x.css">',
    '<img src="img/a.png">',
    '<img src="../../escape.png">',
  ].join('');

  test('fetches only the references that resolve inside the workspace', async () => {
    const { asked, readers } = spyReaders();
    const out = await loadInlineAssets(html, 'reports/index.html', readers);

    expect(asked.sort()).toEqual(['reports/img/a.png', 'reports/style.css']);
    expect(out.get('style.css')).toEqual({ kind: 'style', css: '/* reports/style.css */' });
    expect(out.get('img/a.png')).toEqual({
      kind: 'image',
      dataUri: 'data:image/png;base64,reports/img/a.png',
    });
    expect(out.has('https://cdn.example.com/x.css')).toBe(false);
    expect(out.has('../../escape.png')).toBe(false);
  });

  test('omits an asset that fails to read instead of failing the preview', async () => {
    const { readers } = spyReaders('reports/style.css');
    const out = await loadInlineAssets(html, 'reports/index.html', readers);

    expect(out.has('style.css')).toBe(false);
    expect(out.has('img/a.png')).toBe(true);
  });

  test('stops at the asset limit', async () => {
    const many = Array.from({ length: 8 }, (_, i) => `<img src="a${i}.png">`).join('');
    const { asked, readers } = spyReaders();
    const out = await loadInlineAssets(many, 'index.html', readers, 3);

    expect(asked).toHaveLength(3);
    expect(out.size).toBe(3);
  });

  test('an inlined preview round-trips through inlineAssets and buildSrcdoc', async () => {
    const { readers } = spyReaders();
    const doc = '<html><head><link rel="stylesheet" href="style.css"></head></html>';
    const resolved = await loadInlineAssets(doc, 'reports/index.html', readers);
    const out = buildSrcdoc(inlineAssets(doc, resolved));

    expect(out).toContain('<style>/* reports/style.css */</style>');
    expect(out).not.toContain('<link');
    expect(out.indexOf('Content-Security-Policy')).toBeLessThan(out.indexOf('<style>'));
  });

  test('an href with a query and fragment still inlines, keyed by the original href', async () => {
    const { asked, readers } = spyReaders();
    const doc = '<link rel="stylesheet" href="style.css?v=3#top">';
    const resolved = await loadInlineAssets(doc, 'reports/index.html', readers);

    expect(asked).toEqual(['reports/style.css']);
    const out = inlineAssets(doc, resolved);
    expect(out).toBe('<style>/* reports/style.css */</style>');
  });
});
