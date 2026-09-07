/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), post: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { terminals, type Terminal, type TerminalState } from '$lib/stores/terminals.svelte';
import TerminalsRail from './TerminalsRail.svelte';

const RAIL_PROPS = { focusedTerminalId: null, onOpenTerminal: () => {} };

afterEach(cleanup);
afterEach(() => vi.restoreAllMocks());

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

function term(id: string, cmd: string, state: TerminalState): Terminal {
  return {
    id,
    cmd,
    cwd: null,
    state,
    pid: 4242,
    exit_code: state === 'running' ? null : 0,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    resolved_at: state === 'running' ? null : '2026-01-01T00:00:09Z',
    bytes_out: 12,
    lines_out: 2,
    last_line: 'ok',
    parent_session_id: null,
    truncated: false,
  };
}

/** Returns false when the row handler called preventDefault on the contextmenu. */
async function rightClickRow(name: RegExp): Promise<boolean> {
  const row = page.getByRole('option', { name });
  await expect.element(row).toBeInTheDocument();
  return row
    .element()
    .dispatchEvent(
      new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 20, clientY: 20 }),
    );
}

test('right-clicking a terminal row opens the actions menu and suppresses the browser default', async () => {
  vi.mocked(api.get).mockResolvedValue({ terminals: [term('t1', 'npm test', 'running')] });
  render(TerminalsRail, { props: RAIL_PROPS });

  expect(await rightClickRow(/npm test/)).toBe(false);
  const menu = page.getByRole('menu', { name: 'Terminal actions' });
  await expect.element(menu).toBeInTheDocument();

  await userEvent.keyboard('{Escape}');
  await expect.element(menu).not.toBeInTheDocument();
});

test('a live terminal offers kill and the copy actions, never restart', async () => {
  vi.mocked(api.get).mockResolvedValue({ terminals: [term('t1', 'npm test', 'running')] });
  render(TerminalsRail, { props: RAIL_PROPS });

  await rightClickRow(/npm test/);
  await expect.element(page.getByRole('menuitem', { name: 'Kill' })).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Copy command' })).toBeInTheDocument();
  await expect
    .element(page.getByRole('menuitem', { name: 'Copy terminal id' }))
    .toBeInTheDocument();
  expect(page.getByRole('menuitem', { name: 'Restart' }).query()).toBeNull();
});

test('an exited terminal offers restart, never kill', async () => {
  vi.mocked(api.get).mockResolvedValue({ terminals: [term('t2', 'ruff check', 'succeeded')] });
  render(TerminalsRail, { props: RAIL_PROPS });

  await rightClickRow(/ruff check/);
  await expect.element(page.getByRole('menuitem', { name: 'Restart' })).toBeInTheDocument();
  expect(page.getByRole('menuitem', { name: 'Kill' }).query()).toBeNull();
});

test('picking kill from the row menu kills that terminal without opening it', async () => {
  vi.mocked(api.get).mockResolvedValue({ terminals: [term('t1', 'npm test', 'running')] });
  const kill = vi.spyOn(terminals, 'kill').mockResolvedValue();
  const onOpenTerminal = vi.fn();
  render(TerminalsRail, { props: { ...RAIL_PROPS, onOpenTerminal } });

  await rightClickRow(/npm test/);
  await page.getByRole('menuitem', { name: 'Kill' }).click();

  expect(kill).toHaveBeenCalledWith('t1');
  expect(onOpenTerminal).not.toHaveBeenCalled();
});
