/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), post: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { terminals } from '$lib/stores/terminals.svelte';
import TerminalsRail from './TerminalsRail.svelte';

const RAIL_PROPS = { focusedTerminalId: null, onOpenTerminal: () => {} };

afterEach(cleanup);

beforeEach(() => {
  vi.mocked(api.get).mockReset();
  terminals.list = [];
  terminals.loading = false;
  terminals.error = null;
  terminals.states = {};
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

test('shows a loading skeleton in the rail while the initial fetch is in flight', async () => {
  const gate = deferred<{ terminals: never[] }>();
  vi.mocked(api.get).mockReturnValue(gate.promise);
  const { container } = await render(TerminalsRail, { props: RAIL_PROPS });
  expect(container.querySelector('.t-skel')).not.toBeNull();
  gate.resolve({ terminals: [] });
});

test('renders a truthful empty state when there are no terminals', async () => {
  vi.mocked(api.get).mockResolvedValue({ terminals: [] });
  render(TerminalsRail, { props: RAIL_PROPS });
  await expect.element(page.getByText('No terminals yet')).toBeInTheDocument();
});

test('renders an error pane with retry on a fetch failure, and retry re-fetches', async () => {
  vi.mocked(api.get).mockRejectedValue(new Error('backend on fire'));
  await render(TerminalsRail, { props: RAIL_PROPS });
  await expect.element(page.getByText("Couldn't load terminals")).toBeInTheDocument();
  await expect.element(page.getByText('backend on fire')).toBeInTheDocument();

  vi.mocked(api.get).mockResolvedValue({ terminals: [] });
  await page.getByRole('button', { name: /retry/i }).click();
  await expect.element(page.getByText('No terminals yet')).toBeInTheDocument();
});
