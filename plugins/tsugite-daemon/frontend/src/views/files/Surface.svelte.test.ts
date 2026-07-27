/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi, beforeEach } from 'vitest';
import { WORKSPACE } from './__fixtures__/workspace';
import { routeHistory } from '$lib/router.svelte';

vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}), api: WORKSPACE.api }));

beforeEach(async () => {
  await page.viewport(1440, 900);
  const { agentsMeta } = await import('$lib/stores/agentsMeta.svelte');
  agentsMeta.agents = [];
  const { filesWorkspace } = await import('./workspace.svelte');
  filesWorkspace.agent = '';
  filesWorkspace.ws = null;
  filesWorkspace.loading = false;
  filesWorkspace.error = null;
  filesWorkspace.indexState = 'none';
});

async function mountSurface(path: string) {
  const { default: Surface } = await import('./Surface.svelte');
  render(Surface, { props: { params: { agent: 'smoke', path } } });
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

test('backlinks and related notes appear after the explicit on-demand scan, never eagerly', async () => {
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  // No scan yet: the meta pane offers it instead of silently bulk-reading.
  await expect.element(page.getByRole('button', { name: 'Scan workspace' })).toBeInTheDocument();

  await page.getByRole('button', { name: 'Scan workspace' }).click();
  const backlinks = page.getByTestId('files-backlinks');
  await expect.element(backlinks.getByText('ops/beta.md')).toBeInTheDocument();
  await expect.element(page.getByText(/1 note shares/)).toBeInTheDocument();
});

test('the raw toggle shows the source, tags line and all', async () => {
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();

  await page.getByRole('button', { name: 'raw', exact: true }).click();
  await expect.element(page.getByText('tags: #ops #x')).toBeInTheDocument();
});

test('at phone width the toolbar shows a back affordance that clears the path to the list', async () => {
  // Phone drilldown: an open document is a screen reached from the file tree; its
  // toolbar leads with back, which clears ?path back to the #files list.
  await page.viewport(390, 780);
  routeHistory.prev = null;
  location.hash = '#files?agent=smoke&path=ops/alpha.md';
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();
  await expect.element(page.getByTestId('phone-back')).toBeVisible();
  await page.getByTestId('phone-back').click();
  expect(location.hash).toBe('#files');
});

test('at desktop width the toolbar back affordance is hidden', async () => {
  await page.viewport(1440, 900);
  await mountSurface('ops/alpha.md');
  await expect.element(page.getByRole('heading', { name: 'Alpha', level: 1 })).toBeInTheDocument();
  await expect.element(page.getByTestId('phone-back')).not.toBeVisible();
});
