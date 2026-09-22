/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi, beforeEach } from 'vitest';
import { TESTID } from '$lib/testids';
import { theme } from '$lib/stores/theme.svelte';
import { WORKSPACE } from '../files/__fixtures__/workspace';
// The pane ships resolved token values into the frame, so the test page needs
// the real sheet.
import '../../styles/tokens.css';

vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}), api: WORKSPACE.api }));

beforeEach(async () => {
  await page.viewport(1200, 800);
  theme.set('mocha');
  WORKSPACE.reset();
  const { artifacts } = await import('$lib/stores/artifacts.svelte');
  artifacts.items = {};
  const { spaces, defaultSpace } = await import('$lib/stores/spaces.svelte');
  const fresh = defaultSpace();
  spaces.spaces = [fresh];
  spaces.activeSpaceId = fresh.id;
});

/** Feed the daemon's frame through the real store, as the shell sink does. */
async function openArtifact(extra: Record<string, unknown> = {}) {
  const { artifacts, ARTIFACT_EVENT } = await import('$lib/stores/artifacts.svelte');
  artifacts.applySessionEvent({
    session_id: 'sess-1',
    event_type: ARTIFACT_EVENT,
    artifact_id: 'agent',
    path: 'ops/alpha.md',
    content: null,
    content_type: 'markdown',
    mode: 'rendered',
    title: 'alpha.md',
    placement: 'right',
    opened_by: 'agent',
    ...extra,
  });
  return artifacts;
}

function frameEl(): HTMLIFrameElement {
  return page.getByTestId(TESTID.artifactHtmlFrame).element() as HTMLIFrameElement;
}

/** The document the pane handed the sandboxed frame. */
async function frameDoc(): Promise<string> {
  await expect.element(page.getByTestId(TESTID.artifactHtmlFrame)).toBeInTheDocument();
  return frameEl().srcdoc;
}

async function mount(id = 'agent', sessionId: string | null = 'sess-1', tabId?: string) {
  const { default: Surface } = await import('./Surface.svelte');
  return render(Surface, {
    props: { params: sessionId ? { id, sessionId } : { id }, tabId },
  });
}

async function dockArtifactTab(id = 'agent', sessionId: string | null = 'sess-1') {
  const { spaces } = await import('$lib/stores/spaces.svelte');
  const { collectLeaves } = await import('$lib/shell/mux/layout');
  const rootPane = collectLeaves(spaces.active.layout.root)[0]!.id;
  spaces.dock(rootPane, {
    kind: 'artifact',
    params: sessionId ? { id, sessionId } : { id },
  });
  return collectLeaves(spaces.active.layout.root)
    .flatMap((l) => l.tabs)
    .find((t) => t.kind === 'artifact')!.id;
}

test('a markdown artifact renders, badged as opened by the agent', async () => {
  await openArtifact();
  await mount();

  await expect.element(page.getByTestId(TESTID.artifactPane)).toBeVisible();
  await expect.element(page.getByTestId(TESTID.artifactAgentBadge)).toBeVisible();
  expect(await frameDoc()).toContain('<h1>Alpha</h1>');
});

test('an html artifact renders in the same sandboxed frame the file browser uses', async () => {
  await openArtifact({ path: 'reports/report.html', content_type: 'html', title: 'report.html' });
  await mount();

  const frame = page.getByTestId(TESTID.artifactHtmlFrame);
  await expect.element(frame).toBeInTheDocument();
  const el = frame.element() as HTMLIFrameElement;
  expect(el.getAttribute('sandbox')).toBe('');
  expect(el.srcdoc).toContain("default-src 'none'");
  expect(el.srcdoc).toContain('<h1>Coverage</h1>');
});

test('markdown content cannot inject an element into the app document', async () => {
  const payload = '<img src=x onerror="window.stolen = true">';
  await openArtifact({ path: null, content: `# Doc\n\n${payload}\n`, title: 'Doc' });
  await mount();
  await expect.element(page.getByTestId(TESTID.artifactPane)).toBeVisible();

  expect(document.querySelector('img[onerror]')).toBeNull();

  expect(await frameDoc()).toContain('onerror');
});

test('the source toggle drops out of the rendered view and back', async () => {
  await openArtifact({ path: 'reports/report.html', content_type: 'html' });
  await mount();
  await expect.element(page.getByTestId(TESTID.artifactHtmlFrame)).toBeInTheDocument();

  await page.getByRole('button', { name: 'source', exact: true }).click();
  await expect.element(page.getByTestId(TESTID.artifactHtmlFrame)).not.toBeInTheDocument();
  await expect.element(page.getByText(/parent\.steal\(\)/)).toBeInTheDocument();

  await page.getByRole('button', { name: 'rendered', exact: true }).click();
  await expect.element(page.getByTestId(TESTID.artifactHtmlFrame)).toBeInTheDocument();
});

test('a second open replaces what the pane shows without remounting it', async () => {
  await openArtifact();
  await mount();
  expect(await frameDoc()).toContain('<h1>Alpha</h1>');

  await openArtifact({ path: 'ops/beta.md', title: 'beta.md' });

  await expect.poll(frameDoc).toContain('<h1>Beta</h1>');
  expect(await frameDoc()).not.toContain('<h1>Alpha</h1>');
  await expect.element(page.getByText('alpha.md')).not.toBeInTheDocument();
});

test('ephemeral generated content shows without any file read', async () => {
  await openArtifact({ path: null, content: '# Summary\n\nall good\n', title: 'Summary' });
  await mount();

  // The pane shows the inline text, so no fixture file was read for it.
  const doc = await frameDoc();
  expect(doc).toContain('<h1>Summary</h1>');
  expect(doc).toContain('all good');
  // Nothing to toggle away from for plain text, but markdown keeps its toggle.
  await expect.element(page.getByTestId(TESTID.artifactModeSeg)).toBeVisible();
});

test('plain text opens as source with no rendered segment offered', async () => {
  await openArtifact({ path: null, content: 'line one', content_type: 'text', title: 'log' });
  await mount();

  await expect.element(page.getByText('line one')).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.artifactModeSeg)).not.toBeInTheDocument();
});

test('the close control dismisses the artifact', async () => {
  const store = await openArtifact();
  const tabId = await dockArtifactTab();
  await mount('agent', 'sess-1', tabId);
  await expect.element(page.getByTestId(TESTID.artifactPane)).toBeVisible();

  await page.getByTestId(TESTID.artifactClose).click();

  expect(Object.keys(store.items)).toEqual([]);
  await expect.element(page.getByText(/no longer open/i)).toBeInTheDocument();
  const { spaces } = await import('$lib/stores/spaces.svelte');
  const { collectLeaves } = await import('$lib/shell/mux/layout');
  const kinds = collectLeaves(spaces.active.layout.root).flatMap((l) => l.tabs.map((t) => t.kind));
  expect(kinds).not.toContain('artifact');
});

test('closing one of two tabs sharing a surface key leaves the other tab and their shared artifact', async () => {
  const store = await openArtifact();
  const { spaces } = await import('$lib/stores/spaces.svelte');
  const { collectLeaves } = await import('$lib/shell/mux/layout');
  const keepTabId = await dockArtifactTab();
  const rootPane = collectLeaves(spaces.active.layout.root)[0]!.id;
  spaces.split(rootPane, 'row', { kind: 'chat', title: 'Chat' });
  const chatPane = collectLeaves(spaces.active.layout.root).find((l) => l.id !== rootPane)!.id;
  spaces.dock(chatPane, { kind: 'artifact', params: { id: 'agent', sessionId: 'sess-1' } });
  const dupTabId = collectLeaves(spaces.active.layout.root)
    .find((l) => l.id === chatPane)!
    .tabs.find((t) => t.kind === 'artifact')!.id;

  await mount('agent', 'sess-1', dupTabId);
  await page.getByTestId(TESTID.artifactClose).click();

  const remaining = collectLeaves(spaces.active.layout.root).flatMap((l) =>
    l.tabs.filter((t) => t.kind === 'artifact'),
  );
  expect(remaining.map((t) => t.id)).toEqual([keepTabId]);
  expect(store.get('agent', 'sess-1')?.title).toBe('alpha.md');
});

test('the close control still undocks a pane whose artifact record already expired', async () => {
  const store = await openArtifact();
  const tabId = await dockArtifactTab();
  store.close('agent', 'sess-1');
  await mount('agent', 'sess-1', tabId);
  await expect.element(page.getByText(/no longer open/i)).toBeInTheDocument();

  await page.getByTestId(TESTID.artifactClose).click();

  const { spaces } = await import('$lib/stores/spaces.svelte');
  const { collectLeaves } = await import('$lib/shell/mux/layout');
  const kinds = collectLeaves(spaces.active.layout.root).flatMap((l) => l.tabs.map((t) => t.kind));
  expect(kinds).not.toContain('artifact');
});

test('a slot with no record explains itself instead of rendering blank', async () => {
  await mount('gone');
  await expect.element(page.getByText(/no longer open/i)).toBeInTheDocument();
});

/** `--bg1` as tokens.css resolves it, so the theme assertions are about values
 *  and not merely the presence of a <style> block. */
const BG1 = { mocha: '#181825', latte: '#e6e9ef' } as const;

test('a generated markdown artifact renders in the active theme, and follows a switch', async () => {
  await openArtifact({ path: null, content: '# Summary\n\nall good\n', title: 'Summary' });
  await mount();
  expect(await frameDoc()).toContain(BG1.mocha);
  expect(getComputedStyle(frameEl()).colorScheme).toBe('dark');

  theme.set('latte');

  await expect.poll(frameDoc).toContain(BG1.latte);
  await expect.poll(() => getComputedStyle(frameEl()).colorScheme).toBe('light');
});

test("a generated html artifact's own styles win over the injected defaults", async () => {
  await openArtifact({
    path: null,
    content: '<style>body{background:#ffffff}</style><h1>Report</h1>',
    content_type: 'html',
    title: 'Report',
  });
  await mount();

  const doc = await frameDoc();
  expect(doc).toContain(BG1.mocha);
  expect(doc.indexOf(BG1.mocha)).toBeLessThan(doc.indexOf('body{background:#ffffff}'));
});

test('a path-backed html file is left as its author wrote it', async () => {
  await openArtifact({ path: 'reports/report.html', content_type: 'html', title: 'report.html' });
  await mount();

  expect(await frameDoc()).not.toContain('--bg1');
  expect(getComputedStyle(frameEl()).colorScheme).toBe('light');
});

test('a slow read never overwrites a newer artifact', async () => {
  const { files } = await import('$lib/stores/files.svelte');
  let release: (file: { content: string }) => void = () => {};
  const read = vi
    .spyOn(files, 'read')
    .mockImplementation(() => new Promise((resolve) => (release = resolve as typeof release)));

  await openArtifact({ path: 'ops/alpha.md', title: 'alpha.md' });
  await mount();

  // The agent replaces the slot while that read is still in flight.
  await openArtifact({ path: null, content: '# Newer\n', title: 'Newer' });
  await expect.poll(frameDoc).toContain('<h1>Newer</h1>');

  release({ content: '# Stale\n' });
  await new Promise((resolve) => setTimeout(resolve, 60));

  const doc = await frameDoc();
  expect(doc).toContain('<h1>Newer</h1>');
  expect(doc).not.toContain('Stale');
  read.mockRestore();
});
