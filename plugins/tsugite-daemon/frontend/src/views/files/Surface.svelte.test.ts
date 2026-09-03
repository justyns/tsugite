/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi, beforeEach } from 'vitest';
import { WORKSPACE } from './__fixtures__/workspace';
import { routeHistory } from '$lib/router.svelte';
import { TESTID } from '$lib/testids';
// The column measurements below depend on the app's global border-box reset.
import '../../styles/tokens.css';

vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}), api: WORKSPACE.api }));

beforeEach(async () => {
  await page.viewport(1440, 900);
  WORKSPACE.reset();
  const { agentsMeta } = await import('$lib/stores/agentsMeta.svelte');
  agentsMeta.runtime = null;
  const { filesWorkspace } = await import('./workspace.svelte');
  filesWorkspace.ws = null;
  filesWorkspace.loading = false;
  filesWorkspace.error = null;
  filesWorkspace.indexState = 'none';
  const { files } = await import('$lib/stores/files.svelte');
  files.lastWrite = null;
});

async function mountSurface(path: string) {
  const { default: Surface } = await import('./Surface.svelte');
  render(Surface, { props: { params: { path } } });
}

/** Replay the file_write frame through the real shell router. */
async function broadcastFileWrite(path: string) {
  const { routeShellEvent } = await import('$lib/api/events');
  const { files } = await import('$lib/stores/files.svelte');
  const sink = { onSessionEvent: (data: Record<string, unknown>) => files.applySessionEvent(data) };
  const data = { session_id: 's1', event_type: 'file_write', path, line_count: 1 };
  routeShellEvent({ type: 'session_event', seq: 1, data }, sink);
}

test('opens the pointed-at note and renders its markdown', async () => {
  await mountSurface('index.md');
  await expect.element(page.getByRole('heading', { name: 'Home', level: 1 })).toBeInTheDocument();
});

test('wikilinks resolve, missing pages are flagged, and navigation follows them', async () => {
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  const beta = page.getByRole('link', { name: /\[\[beta\]\]/ });
  await expect.element(beta).toHaveAttribute('data-wk-nav', 'ops/beta.md');
  await expect
    .element(page.getByRole('link', { name: /ghost.*missing page/i }))
    .toBeInTheDocument();

  await beta.click();
  await expect.element(page.getByRole('heading', { name: 'Beta', level: 1 })).toBeInTheDocument();
});

test('a focused wikilink activates on Enter and on Space, like a click', async () => {
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  const beta = page.getByRole('link', { name: /\[\[beta\]\]/ });
  (beta.element() as HTMLElement).focus();
  await userEvent.keyboard('{Enter}');
  await expect.element(page.getByRole('heading', { name: 'Beta', level: 1 })).toBeInTheDocument();

  const alpha = page.getByRole('link', { name: /\[\[alpha\]\]/ });
  (alpha.element() as HTMLElement).focus();
  await userEvent.keyboard(' ');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();
});

test('a note carrying raw HTML and a script URL renders inert', async () => {
  WORKSPACE.setContent(
    'ops/alpha.md',
    '# Alpha\n\n<img src=x onerror="alert(1)">\n\n[click](javascript:alert(1))\n',
  );
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  const docEl = page.getByTestId(TESTID.filesDoc).element();
  expect(docEl.querySelector('img')).toBeNull();
  expect(docEl.querySelector('a[href]')).toBeNull();
  expect(docEl.textContent).toContain('<img src=x onerror="alert(1)">');
});

test('selecting text in the rendered document raises the annotation popover', async () => {
  await mountSurface('ops/alpha.md');
  await expect
    .element(page.getByRole('heading', { name: 'Section', level: 2 }))
    .toBeInTheDocument();

  const popover = page.getByRole('menu', { name: 'Selection actions', includeHidden: true });
  await expect.element(popover).not.toBeVisible();

  const docEl = page.getByTestId(TESTID.filesDoc).element() as HTMLElement;
  const para = Array.from(docEl.querySelectorAll('.doc-md p')).find(
    (p) => p.textContent?.trim() === 'selectable paragraph',
  ) as HTMLParagraphElement;
  const textNode = para.firstChild as Text;
  const range = document.createRange();
  range.setStart(textNode, 0);
  range.setEnd(textNode, textNode.length);
  const sel = window.getSelection();
  sel?.removeAllRanges();
  sel?.addRange(range);
  docEl.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));

  await expect.element(popover).toBeVisible();

  await page.getByRole('menuitem', { name: /Copy ref/ }).click();
  await expect.element(popover).not.toBeVisible();
});

test('backlinks and related notes appear after the explicit on-demand scan, never eagerly', async () => {
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  // No scan yet: the meta pane offers it instead of silently bulk-reading.
  await expect.element(page.getByRole('button', { name: 'Scan workspace' })).toBeInTheDocument();

  await page.getByRole('button', { name: 'Scan workspace' }).click();
  const backlinks = page.getByTestId(TESTID.filesBacklinks);
  await expect.element(backlinks.getByText('ops/beta.md')).toBeInTheDocument();
  await expect.element(page.getByText(/1 note shares/)).toBeInTheDocument();
});

test('the raw toggle shows the source, tags line and all', async () => {
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  await page.getByRole('button', { name: 'raw', exact: true }).click();
  await expect.element(page.getByText('tags: #ops #x')).toBeInTheDocument();
});

test('an agent tool edit to the open note refreshes the tab in place', async () => {
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  WORKSPACE.setContent('ops/alpha.md', '# Alpha\n\ntags: #ops\n\nrewritten by the agent\n');
  await broadcastFileWrite('ops/alpha.md');

  await expect.element(page.getByText('rewritten by the agent')).toBeInTheDocument();
});

test('a write to another file leaves the open note alone', async () => {
  await mountSurface('ops/alpha.md');
  const section = page.getByRole('heading', { name: 'Section', level: 2 });
  await expect.element(section).toBeInTheDocument();

  WORKSPACE.setContent('ops/alpha.md', '# Alpha\n\nrewritten by the agent\n');
  await broadcastFileWrite('ops/beta.md');

  await expect.element(page.getByText('rewritten by the agent')).not.toBeInTheDocument();
  await expect.element(section).toBeInTheDocument();
});

test('unsaved edits survive an agent write and raise a stale-content warning', async () => {
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  await page.getByRole('button', { name: 'edit', exact: true }).click();
  const area = page.getByRole('textbox', { name: 'Edit ops/alpha.md' });
  await area.fill('my local draft');

  const fresh = '# Alpha\n\nrewritten by the agent\n';
  WORKSPACE.setContent('ops/alpha.md', fresh);
  await broadcastFileWrite('ops/alpha.md');

  await expect.element(page.getByTestId(TESTID.filesStale)).toBeVisible();
  await expect.element(area).toHaveValue('my local draft');

  await page.getByRole('button', { name: /reload from disk/i }).click();
  await expect.element(area).toHaveValue(fresh);
  await expect.element(page.getByTestId(TESTID.filesStale)).not.toBeInTheDocument();
});

test('at phone width the toolbar shows a back affordance that clears the path to the list', async () => {
  // Phone drilldown: an open document is a screen reached from the file tree; its
  // toolbar leads with back, which clears ?path back to the #files list.
  await page.viewport(390, 780);
  routeHistory.prev = null;
  location.hash = '#files?agent=smoke&path=ops/alpha.md';
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.phoneBack)).toBeVisible();
  await page.getByTestId(TESTID.phoneBack).click();
  expect(location.hash).toBe('#files');
});

test('at desktop width the toolbar back affordance is hidden', async () => {
  await page.viewport(1440, 900);
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.phoneBack)).not.toBeVisible();
});

// The column keys off pane width, not window width. Mount at a pane width and
// assert the document gets the whole pane whenever the column is hidden.
async function paneSurface(width: number) {
  await page.viewport(1440, 900);
  const { default: Surface } = await import('./Surface.svelte');
  const { container } = await render(Surface, {
    props: { params: { path: 'ops/alpha.md' } },
  });
  container.style.width = `${width}px`;
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();
  const shell = container.querySelector('.wk-shell') as HTMLElement;
  const doc = container.querySelector('section[aria-label="Document"]') as HTMLElement;
  const meta = container.querySelector(`[data-testid="${TESTID.filesMeta}"]`) as HTMLElement;
  return {
    shell: shell.getBoundingClientRect().width,
    doc: doc.getBoundingClientRect().width,
    metaHidden: getComputedStyle(meta).display === 'none',
    meta: meta.getBoundingClientRect().width,
  };
}

test('a wide pane shows the metadata column beside the document', async () => {
  const { shell, doc, metaHidden, meta } = await paneSurface(1000);
  expect(metaHidden).toBe(false);
  expect(meta).toBeGreaterThan(0);
  expect(doc + meta).toBeCloseTo(shell, 0);
});

test('an intermediate pane drops the metadata column without reserving its track', async () => {
  const { shell, doc, metaHidden } = await paneSurface(600);
  expect(metaHidden).toBe(true);
  expect(doc).toBeCloseTo(shell, 0);
});

test('a narrow pane drops the metadata column without reserving its track', async () => {
  const { shell, doc, metaHidden } = await paneSurface(380);
  expect(metaHidden).toBe(true);
  expect(doc).toBeCloseTo(shell, 0);
});

async function previewFrame(): Promise<HTMLIFrameElement> {
  const frame = page.getByTestId(TESTID.filesHtmlFrame);
  await expect.element(frame).toBeInTheDocument();
  return frame.element() as HTMLIFrameElement;
}

test('an html file opens in a sandboxed rendered preview by default', async () => {
  await mountSurface('reports/report.html');
  const frame = await previewFrame();

  // Empty sandbox: without allow-same-origin the frame cannot reach app state.
  expect(frame.getAttribute('sandbox')).toBe('');
  expect(frame.srcdoc).toContain('Content-Security-Policy');
  expect(frame.srcdoc).toContain("default-src 'none'");
  expect(frame.srcdoc).toContain('<h1>Coverage</h1>');
});

test('the preview inlines a same-workspace stylesheet and leaves the external one blocked', async () => {
  await mountSurface('reports/report.html');
  const frame = await previewFrame();

  await vi.waitFor(() => expect(frame.srcdoc).toContain('rebeccapurple'));
  expect(frame.srcdoc).toContain('<style>h1 { color: rebeccapurple }</style>');
  expect(frame.srcdoc).not.toContain('href="report.css"');
  // The CDN reference stays as written; the CSP is what stops it loading.
  expect(frame.srcdoc).toContain('https://cdn.example.com/evil.css');
});

test('the raw toggle falls back to the source, script tag and all, and rendered comes back', async () => {
  await mountSurface('reports/report.html');
  await expect.element(page.getByTestId(TESTID.filesHtmlFrame)).toBeInTheDocument();

  await page.getByRole('button', { name: 'raw', exact: true }).click();
  await expect.element(page.getByTestId(TESTID.filesHtmlFrame)).not.toBeInTheDocument();
  await expect.element(page.getByText(/parent\.steal\(\)/)).toBeInTheDocument();

  await page.getByRole('button', { name: 'rendered', exact: true }).click();
  await expect.element(page.getByTestId(TESTID.filesHtmlFrame)).toBeInTheDocument();
});

test('a plain text file still opens raw, with no rendered segment offered', async () => {
  WORKSPACE.setContent('reports/report.css', 'h1 { color: rebeccapurple }');
  await mountSurface('reports/report.css');
  await expect.element(page.getByText('rebeccapurple')).toBeInTheDocument();
  await expect
    .element(page.getByRole('button', { name: 'rendered', exact: true }))
    .not.toBeInTheDocument();
});
