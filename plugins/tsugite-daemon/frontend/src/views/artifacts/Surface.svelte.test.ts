/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi, beforeEach } from 'vitest';
import { TESTID } from '$lib/testids';
import { WORKSPACE } from '../files/__fixtures__/workspace';

vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}), api: WORKSPACE.api }));

beforeEach(async () => {
  await page.viewport(1200, 800);
  WORKSPACE.reset();
  const { artifacts } = await import('$lib/stores/artifacts.svelte');
  artifacts.items = {};
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

/** The document the pane handed the sandboxed frame. */
async function frameDoc(): Promise<string> {
  const frame = page.getByTestId(TESTID.artifactHtmlFrame);
  await expect.element(frame).toBeInTheDocument();
  return (frame.element() as HTMLIFrameElement).srcdoc;
}

async function mount(id = 'agent') {
  const { default: Surface } = await import('./Surface.svelte');
  return render(Surface, { props: { params: { id } } });
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
  await mount();
  await expect.element(page.getByTestId(TESTID.artifactPane)).toBeVisible();

  await page.getByTestId(TESTID.artifactClose).click();

  expect(Object.keys(store.items)).toEqual([]);
  await expect.element(page.getByText(/no longer open/i)).toBeInTheDocument();
});

test('a slot with no record explains itself instead of rendering blank', async () => {
  await mount('gone');
  await expect.element(page.getByText(/no longer open/i)).toBeInTheDocument();
});
