/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { cleanup, render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import SessionsRail from './SessionsRail.svelte';
import { spaces } from '$lib/stores/spaces.svelte';
import { collectLeaves } from '$lib/shell/mux/layout';
import { TESTID } from '$lib/testids';
import { sessions } from '$lib/stores/sessions.svelte';
import { agent, sessionRow as row } from './__fixtures__/sessionRow';
import '../../styles/tokens.css';

// The ended section and agent groups persist their open/closed choice.
beforeEach(() => {
  for (const key of Object.keys(localStorage)) {
    if (key.startsWith('tsugite_rail_')) localStorage.removeItem(key);
  }
});
afterEach(() => vi.restoreAllMocks());

const base = {
  selectedId: null,
  attn: new Set<string>(),
  onSelect: vi.fn(),
  onNew: vi.fn(),
  onServerSearch: vi.fn(),
  onOpenSchedule: vi.fn(),
};

test('right-clicking a session row opens the actions menu', async () => {
  render(SessionsRail, { ...base, rows: [row('s1', { title: 'sse backoff' })] });
  await page.getByRole('button', { name: /sse backoff/ }).click({ button: 'right' });
  const menu = page.getByRole('menu', { name: 'Session actions' });
  await expect.element(menu).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Open in new tab' })).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Pin' })).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Copy session id' })).toBeInTheDocument();
  await expect.element(page.getByRole('menuitem', { name: 'Mark complete' })).toBeInTheDocument();

  // Ended sessions offer no lifecycle action; pinned rows offer Unpin.
  await userEvent.keyboard('{Escape}');
  await expect.element(menu).not.toBeInTheDocument();
});

test('"Open in new tab" docks the session as a fresh chat tab', async () => {
  render(SessionsRail, { ...base, rows: [row('s2', { title: 'backup prune' })] });
  const before = collectLeaves(spaces.active.layout.root).flatMap((l) => l.tabs).length;

  await page.getByRole('button', { name: /backup prune/ }).click({ button: 'right' });
  await page.getByRole('menuitem', { name: 'Open in new tab' }).click();

  const tabs = collectLeaves(spaces.active.layout.root).flatMap((l) => l.tabs);
  expect(tabs.length).toBe(before + 1);
  const docked = tabs[tabs.length - 1]!;
  expect(docked.kind).toBe('chat');
  expect(docked.params.sessionId).toBe('s2');
});

test('finished sessions tuck into a collapsed "ended" section with a count', async () => {
  render(SessionsRail, {
    ...base,
    rows: [
      row('l', { title: 'live draft' }),
      row('c', { title: 'wrapped up sync', status: 'completed' }),
      row('f', { title: 'blew up mid run', status: 'failed' }),
    ],
  });
  // the live row stays in the main recency flow
  await expect.element(page.getByRole('button', { name: /live draft/ })).toBeInTheDocument();
  // finished rows leave that flow for a collapsed section that counts them
  const toggle = page.getByRole('button', { name: 'ended 2' });
  await expect.element(toggle).toBeInTheDocument();
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect.element(page.getByText('wrapped up sync')).not.toBeInTheDocument();
  await expect.element(page.getByText('blew up mid run')).not.toBeInTheDocument();
});

test('expanding "ended" reveals the finished rows, visibly muted', async () => {
  render(SessionsRail, {
    ...base,
    rows: [row('c', { title: 'wrapped up sync', status: 'completed' })],
  });
  await page.getByRole('button', { name: 'ended 1' }).click();
  const rowBtn = page.getByRole('button', { name: /wrapped up sync/ });
  await expect.element(rowBtn).toBeInTheDocument();
  await expect.element(rowBtn).toHaveClass(/is-ended/);
});

test('a finished session is never counted as needs-you', async () => {
  render(SessionsRail, {
    ...base,
    attn: new Set(['c']),
    rows: [row('c', { title: 'wrapped up sync', status: 'completed' })],
  });
  await expect.element(page.getByTestId(TESTID.chatNeedsYou)).toHaveTextContent('needs you 0');
});

test('a status: filter surfaces finished sessions even though ended is collapsed by default', async () => {
  render(SessionsRail, {
    ...base,
    rows: [
      row('l', { title: 'live draft' }),
      row('c', { title: 'wrapped up sync', status: 'completed' }),
    ],
  });
  await page.getByLabelText('Filter sessions').fill('status:completed');
  await expect.element(page.getByText('wrapped up sync')).toBeInTheDocument();
  await expect.element(page.getByText('live draft')).not.toBeInTheDocument();
});

test('double-clicking a session row docks it as a fresh chat tab', async () => {
  render(SessionsRail, { ...base, rows: [row('s3', { title: 'reconnect backoff' })] });
  const before = collectLeaves(spaces.active.layout.root).flatMap((l) => l.tabs).length;

  await userEvent.dblClick(page.getByRole('button', { name: /reconnect backoff/ }));

  const tabs = collectLeaves(spaces.active.layout.root).flatMap((l) => l.tabs);
  expect(tabs.length).toBe(before + 1);
  const docked = tabs[tabs.length - 1]!;
  expect(docked.kind).toBe('chat');
  expect(docked.params).toEqual({ sessionId: 's3' });
});

function dragRowOnto(from: HTMLElement, to: HTMLElement, half: 'top' | 'bottom') {
  const dataTransfer = new DataTransfer();
  from.dispatchEvent(new DragEvent('dragstart', { dataTransfer, bubbles: true }));
  const r = to.getBoundingClientRect();
  const clientY = half === 'top' ? r.top + r.height * 0.25 : r.top + r.height * 0.75;
  for (const type of ['dragover', 'drop']) {
    to.dispatchEvent(new DragEvent(type, { dataTransfer, bubbles: true, clientY }));
  }
}

test('dragging a pinned row above another sends the new pinned order', async () => {
  const reorder = vi.spyOn(sessions, 'reorderPins').mockResolvedValue(undefined);
  render(SessionsRail, {
    ...base,
    rows: [
      row('p1', { title: 'alpha pin', pinned: true, pin_position: 0 }),
      row('p2', { title: 'beta pin', pinned: true, pin_position: 1 }),
    ],
  });

  const first = (await page.getByText('alpha pin').element()).closest('.srow-drag') as HTMLElement;
  const second = (await page.getByText('beta pin').element()).closest('.srow-drag') as HTMLElement;
  dragRowOnto(second, first, 'top');

  expect(reorder).toHaveBeenCalledWith(['p2', 'p1']);
});

test('a filtered rail refuses to reorder pins', async () => {
  // The rail only shows part of the pin set under a filter, and the daemon
  // rewrites positions 0..N-1 for exactly the ids it is sent - so reordering
  // what is visible would collide the hidden pins onto those positions.
  const reorder = vi.spyOn(sessions, 'reorderPins').mockResolvedValue(undefined);
  render(SessionsRail, {
    ...base,
    rows: [
      row('p1', { title: 'alpha pin', pinned: true, pin_position: 0 }),
      row('p2', { title: 'hidden pin', pinned: true, pin_position: 1 }),
      row('p3', { title: 'alpha other', pinned: true, pin_position: 2 }),
    ],
  });

  await userEvent.fill(page.getByRole('searchbox', { name: 'Filter sessions' }), 'alpha');
  await expect.element(page.getByText('hidden pin')).not.toBeInTheDocument();

  const first = (await page.getByText('alpha pin').element()).closest('.srow-drag') as HTMLElement;
  const third = (await page.getByText('alpha other').element()).closest(
    '.srow-drag',
  ) as HTMLElement;
  dragRowOnto(third, first, 'top');

  expect(reorder).not.toHaveBeenCalled();
});

test('an unpinned row is not a reorder drop target', async () => {
  const reorder = vi.spyOn(sessions, 'reorderPins').mockResolvedValue(undefined);
  render(SessionsRail, {
    ...base,
    rows: [
      row('p1', { title: 'alpha pin', pinned: true, pin_position: 0 }),
      row('u1', { title: 'loose one' }),
    ],
  });

  const pinned = (await page.getByText('alpha pin').element()).closest('.srow-drag') as HTMLElement;
  const loose = (await page.getByText('loose one').element()).closest('.srow-drag') as HTMLElement;
  dragRowOnto(pinned, loose, 'top');

  expect(reorder).not.toHaveBeenCalled();
});

const twoAgents = [agent('ops', 'homelab'), agent('main', 'atlas', { is_default: true })];

test('a single agent renders the flat rail with no agent level', async () => {
  render(SessionsRail, {
    ...base,
    agents: [agent('main', 'atlas', { is_default: true })],
    rows: [row('s1', { title: 'sse backoff' })],
  });
  await expect.element(page.getByRole('button', { name: /sse backoff/ })).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: /^atlas/ })).not.toBeInTheDocument();
});

test('agents list default first with only the default expanded', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [
      row('m1', { title: 'deploy pipeline', agent_id: 'main' }),
      row('o1', { title: 'nas scrub', agent_id: 'ops' }),
    ],
  });
  const names = [...document.querySelectorAll('.agent-hd .agent-name')].map((n) => n.textContent);
  expect(names).toEqual(['atlas', 'homelab']);
  await expect.element(page.getByRole('button', { name: /deploy pipeline/ })).toBeInTheDocument();
  await expect.element(page.getByText('nas scrub')).not.toBeInTheDocument();
});

test('expanding an agent shows its sessions and the choice persists', async () => {
  const props = {
    ...base,
    agents: twoAgents,
    rows: [row('o1', { title: 'nas scrub', agent_id: 'ops' })],
  };
  render(SessionsRail, props);
  const toggle = page.getByRole('button', { name: /^homelab/ });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
  await toggle.click();
  await expect.element(page.getByRole('button', { name: /nas scrub/ })).toBeInTheDocument();
  cleanup();

  render(SessionsRail, props);
  await expect.element(page.getByRole('button', { name: /nas scrub/ })).toBeInTheDocument();
});

test('a collapsed agent row shows busy and needs-you from its sessions', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    attn: new Set(['o2']),
    rows: [
      row('o1', { title: 'nas scrub', agent_id: 'ops', busy: true }),
      row('o2', { title: 'disk alert', agent_id: 'ops' }),
    ],
  });
  const toggle = page.getByRole('button', { name: /^homelab/ });
  await expect.element(toggle).toHaveAccessibleName('homelab, working, 1 needs you');
});

test('a filter hides agents with no matching sessions and opens the rest', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [
      row('m1', { title: 'main chat', agent_id: 'main' }),
      row('o1', { title: 'nas scrub', agent_id: 'ops' }),
    ],
  });
  await page.getByLabelText('Filter sessions').fill('scrub');
  await expect.element(page.getByRole('button', { name: /^homelab/ })).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: /nas scrub/ })).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: /^atlas/ })).not.toBeInTheDocument();
});

test('the agent tree does not scroll sideways at phone width', async () => {
  await page.viewport(380, 700);
  render(SessionsRail, {
    ...base,
    agents: [
      agent('main', 'atlas', { is_default: true }),
      agent('ops', 'an-agent-with-a-very-long-name-that-should-truncate-cleanly'),
    ],
    attn: new Set(['o1']),
    rows: [row('o1', { title: 'nas scrub', agent_id: 'ops', busy: true })],
  });
  const scroll = (await page.getByTestId(TESTID.chatRail).element()).querySelector(
    '.rail-scroll',
  ) as HTMLElement;
  expect(scroll.scrollWidth).toBeLessThanOrEqual(scroll.clientWidth);
});

test('collapsing the default agent persists across a re-render', async () => {
  const props = {
    ...base,
    agents: twoAgents,
    rows: [row('m1', { title: 'main chat', agent_id: 'main' })],
  };
  render(SessionsRail, props);
  await page.getByRole('button', { name: /^atlas/ }).click();
  expect(localStorage.getItem('tsugite_rail_agent_open:main')).toBe('0');
  cleanup();

  render(SessionsRail, props);
  await expect
    .element(page.getByRole('button', { name: /^atlas/ }))
    .toHaveAttribute('aria-expanded', 'false');
  await expect.element(page.getByText('main chat')).not.toBeInTheDocument();
});

test('sessions of an unknown agent render in a trailing group', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [
      row('m1', { title: 'main chat', agent_id: 'main' }),
      row('o1', { title: 'nas scrub', agent_id: 'ops' }),
      row('x1', { title: 'stray run', agent_id: 'gone' }),
    ],
  });
  const names = [...document.querySelectorAll('.agent-hd .agent-name')].map((n) => n.textContent);
  expect(names).toEqual(['atlas', 'homelab', 'unknown agent']);
  await page.getByRole('button', { name: /^unknown agent/ }).click();
  await expect.element(page.getByRole('button', { name: /stray run/ })).toBeInTheDocument();
});

function dotColor(agentName: string): string {
  const hd = [...document.querySelectorAll('.agent-hd')].find(
    (h) => h.querySelector('.agent-name')?.textContent === agentName,
  )!;
  return getComputedStyle(hd.querySelector('.agent-dot')!).backgroundColor;
}

function tokenColor(token: string): string {
  const probe = document.createElement('span');
  probe.style.color = `var(${token})`;
  document.body.append(probe);
  const color = getComputedStyle(probe).color;
  probe.remove();
  return color;
}

test('an agent color token resolves through the theme', async () => {
  render(SessionsRail, {
    ...base,
    agents: [
      agent('main', 'atlas', { is_default: true, color: 'green' }),
      agent('ops', 'homelab', { color: 'pink' }),
      agent('dev', 'coding'),
    ],
    rows: [row('m1', { agent_id: 'main' })],
  });
  await expect.element(page.getByRole('button', { name: /^coding/ })).toBeInTheDocument();
  expect(dotColor('atlas')).toBe(tokenColor('--st-ok'));
  expect(dotColor('coding')).toBe(tokenColor('--tx3'));
  const mochaPink = dotColor('homelab');

  document.documentElement.dataset.theme = 'latte';
  try {
    expect(dotColor('atlas')).toBe(tokenColor('--st-ok'));
    expect(dotColor('homelab')).not.toBe(mochaPink);
  } finally {
    delete document.documentElement.dataset.theme;
  }
});

test('the agent owning the open session expands without storing a choice', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    selectedId: 'o1',
    rows: [row('o1', { title: 'nas scrub', agent_id: 'ops' })],
  });
  await expect.element(page.getByRole('button', { name: /nas scrub/ })).toBeInTheDocument();
  expect(localStorage.getItem('tsugite_rail_agent_open:ops')).toBeNull();

  await page.getByRole('button', { name: /^homelab/ }).click();
  await expect.element(page.getByText('nas scrub')).not.toBeInTheDocument();
});

test('clicking an agent name opens its main chat while the chevron only toggles', async () => {
  const onOpenAgent = vi.fn();
  render(SessionsRail, {
    ...base,
    onOpenAgent,
    agents: twoAgents,
    rows: [row('o1', { title: 'nas scrub', agent_id: 'ops' })],
  });
  const toggle = page.getByRole('button', { name: /^homelab/ });

  await page.getByRole('button', { name: 'Open homelab main chat' }).click();
  expect(onOpenAgent).toHaveBeenCalledWith('ops');
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');

  await toggle.click();
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
  expect(onOpenAgent).toHaveBeenCalledTimes(1);
});

test('an agent lists its main chat only as the agent row', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [
      row('o0', { title: 'ops main', agent_id: 'ops', is_primary: true }),
      row('o1', { title: 'nas scrub', agent_id: 'ops' }),
    ],
  });
  await page.getByRole('button', { name: /^homelab/ }).click();
  await expect.element(page.getByRole('button', { name: /nas scrub/ })).toBeInTheDocument();
  await expect.element(page.getByText('ops main')).not.toBeInTheDocument();
});

test('a pinned main chat stays in the pinned section', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [row('o0', { title: 'ops main', agent_id: 'ops', is_primary: true, pinned: true })],
  });
  await expect.element(page.getByRole('button', { name: /ops main/ })).toBeInTheDocument();
});

test('an open main chat marks its agent row active and leaves it collapsed', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    selectedId: 'o0',
    rows: [
      row('o0', { title: 'ops main', agent_id: 'ops', is_primary: true }),
      row('o1', { title: 'nas scrub', agent_id: 'ops' }),
    ],
  });
  const open = page.getByRole('button', { name: 'Open homelab main chat' });
  await expect.element(open).toHaveAttribute('aria-current', 'true');
  await expect
    .element(page.getByRole('button', { name: 'Open atlas main chat' }))
    .not.toHaveAttribute('aria-current');
  await expect
    .element(page.getByRole('button', { name: /^homelab/ }))
    .toHaveAttribute('aria-expanded', 'false');
});

test('the chevron and the agent name each show their own hover', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [row('o1', { title: 'nas scrub', agent_id: 'ops' })],
  });
  const toggle = document.querySelector<HTMLElement>('[aria-label^="homelab"]')!;
  const open = document.querySelector<HTMLElement>('[aria-label="Open homelab main chat"]')!;
  const name = open.querySelector<HTMLElement>('.agent-name')!;

  await userEvent.hover(toggle);
  const toggleHover = getComputedStyle(toggle).backgroundColor;
  expect(getComputedStyle(name).textDecorationLine).toBe('none');

  await userEvent.hover(open);
  expect(getComputedStyle(name).textDecorationLine).toBe('underline');
  expect(getComputedStyle(toggle).backgroundColor).not.toBe(toggleHover);
});

test('the all count leaves out main chats the tree shows only as agent rows', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [
      row('m0', { agent_id: 'main', is_primary: true }),
      row('o0', { agent_id: 'ops', is_primary: true, pinned: true }),
      row('o1', { agent_id: 'ops' }),
    ],
  });
  await expect.element(page.getByRole('button', { name: 'all 2' })).toBeInTheDocument();
});

const opsMain = [
  row('o0', { title: 'ops main', agent_id: 'ops', is_primary: true }),
  row('o1', { title: 'nas scrub', agent_id: 'ops' }),
];

test('the needs-you filter keeps the agent row of a waiting main chat', async () => {
  render(SessionsRail, { ...base, agents: twoAgents, attn: new Set(['o0']), rows: opsMain });
  await page.getByRole('button', { name: 'needs you 1' }).click();
  await expect
    .element(page.getByRole('button', { name: 'Open homelab main chat' }))
    .toBeInTheDocument();
});

test('a search matching a main chat keeps its agent row', async () => {
  render(SessionsRail, { ...base, agents: twoAgents, rows: opsMain });
  await page.getByLabelText('Filter sessions').fill('ops main');
  await expect
    .element(page.getByRole('button', { name: 'Open homelab main chat' }))
    .toBeInTheDocument();
  await expect.element(page.getByText('nas scrub')).not.toBeInTheDocument();
});

test('the flat rail counts the main chat', async () => {
  render(SessionsRail, {
    ...base,
    rows: [row('m0', { is_primary: true }), row('m1')],
  });
  await expect.element(page.getByRole('button', { name: 'all 2' })).toBeInTheDocument();
});

const pinRows = [
  row('p1', { title: 'alpha pin', agent_id: 'main', pinned: true, pin_position: 0 }),
  row('p2', { title: 'beta pin', agent_id: 'ops', pinned: true, pin_position: 1 }),
  row('o1', { title: 'nas scrub', agent_id: 'ops' }),
];

test('pins from every agent gather in one section above the agents', async () => {
  render(SessionsRail, {
    ...base,
    agents: [
      agent('main', 'atlas', { is_default: true }),
      agent('ops', 'homelab', { color: 'green' }),
    ],
    rows: pinRows,
  });
  await expect.element(page.getByRole('button', { name: /beta pin/ })).toBeInTheDocument();
  const scroll = document.querySelector('.rail-scroll')!;
  const heads = [...scroll.querySelectorAll('.sb-group, .agent-hd')].map((e) =>
    e.textContent!.trim(),
  );
  expect(heads[0]).toBe('pinned');
  expect(heads.filter((h) => h === 'pinned')).toHaveLength(1);

  const dot = (await page.getByText('beta pin').element())
    .closest('.srow-drag')!
    .querySelector('.agent-dot') as HTMLElement;
  expect(dot.dataset.color).toBe('green');
  expect(dot.title).toBe('homelab');
  await expect
    .element(page.getByRole('button', { name: 'beta pin, homelab, idle' }))
    .toBeInTheDocument();
});

test('pins of different agents reorder as one list', async () => {
  const reorder = vi.spyOn(sessions, 'reorderPins').mockResolvedValue(undefined);
  render(SessionsRail, { ...base, agents: twoAgents, rows: pinRows });

  const first = (await page.getByText('alpha pin').element()).closest('.srow-drag') as HTMLElement;
  const second = (await page.getByText('beta pin').element()).closest('.srow-drag') as HTMLElement;
  dragRowOnto(second, first, 'top');

  expect(reorder).toHaveBeenCalledWith(['p2', 'p1']);
});

test('each agent keeps its own ended section open state', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [
      row('m1', { title: 'old main', agent_id: 'main', status: 'completed' }),
      row('o1', { title: 'old ops', agent_id: 'ops', status: 'completed' }),
    ],
  });
  await page.getByRole('button', { name: /^homelab/ }).click();
  await page.getByRole('button', { name: '1 ended' }).first().click();
  await expect.element(page.getByText('old main')).toBeInTheDocument();
  await expect.element(page.getByText('old ops')).not.toBeInTheDocument();
});

test('an agent lists its sessions without bucket headers', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    attn: new Set(['m2']),
    rows: [
      row('m1', { title: 'quiet chat', agent_id: 'main' }),
      row('m2', { title: 'waiting chat', agent_id: 'main' }),
    ],
  });
  await expect.element(page.getByRole('button', { name: /quiet chat/ })).toBeInTheDocument();
  expect(document.querySelectorAll('.rail-scroll .sb-group')).toHaveLength(0);
  const titles = [...document.querySelectorAll('.rail-scroll .ttl')].map((t) => t.textContent);
  expect(titles).toEqual(['waiting chat', 'quiet chat']);
});

const sevenRows = Array.from({ length: 7 }, (_, i) =>
  row(`m${i}`, { title: `chat ${i}`, agent_id: 'main' }),
);

test('an agent shows five sessions until "show more" expands it', async () => {
  render(SessionsRail, { ...base, agents: twoAgents, rows: sevenRows });
  await expect.element(page.getByRole('button', { name: /chat 0/ })).toBeInTheDocument();
  expect(document.querySelectorAll('.rail-scroll .t-srow')).toHaveLength(5);

  await page.getByRole('button', { name: 'show 2 more' }).click();
  await expect.element(page.getByRole('button', { name: /chat 6/ })).toBeInTheDocument();
  expect(document.querySelectorAll('.rail-scroll .t-srow')).toHaveLength(7);
});

test('a filter shows every matching session of an agent', async () => {
  render(SessionsRail, { ...base, agents: twoAgents, rows: sevenRows });
  await page.getByLabelText('Filter sessions').fill('chat');
  await expect.element(page.getByRole('button', { name: /chat 6/ })).toBeInTheDocument();
  expect(document.querySelectorAll('.rail-scroll .t-srow')).toHaveLength(7);
  await expect
    .element(page.getByRole('button', { name: /^show \d+ more/ }))
    .not.toBeInTheDocument();
});

test('schedule runs collapse to one row inside an agent', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [
      row('r1', { title: 'nightly', agent_id: 'main', source: 'schedule', parent_id: 'nightly' }),
      row('r2', { title: 'nightly', agent_id: 'main', source: 'schedule', parent_id: 'nightly' }),
    ],
  });
  await expect.element(page.getByLabelText('2 runs')).toBeInTheDocument();
  expect(document.querySelectorAll('.rail-scroll .t-srow')).toHaveLength(1);
});

const at = (h: number) => `2026-10-01T${String(h).padStart(2, '0')}:00:00Z`;
const opsRows = Array.from({ length: 7 }, (_, i) =>
  row(`o${i}`, { title: `ops chat ${i}`, agent_id: 'ops', last_active: at(i + 1) }),
);

test('the open session renders even when it is past its agent row cap', async () => {
  render(SessionsRail, { ...base, agents: twoAgents, selectedId: 'o0', rows: opsRows });
  await expect.element(page.getByRole('button', { name: /ops chat 0/ })).toBeInTheDocument();
  await expect.element(page.getByRole('button', { name: 'show 1 more' })).toBeInTheDocument();
});

test('an open ended session opens its agent ended section', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    selectedId: 'o1',
    rows: [row('o1', { title: 'old ops', agent_id: 'ops', status: 'completed' })],
  });
  await expect.element(page.getByRole('button', { name: /old ops/ })).toBeInTheDocument();
});

test('clicking the ended row that holds the open session closes it', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    selectedId: 'o1',
    rows: [row('o1', { title: 'old ops', agent_id: 'ops', status: 'completed' })],
  });
  const toggle = page.getByRole('button', { name: '1 ended' });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
  await toggle.click();
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
  await expect.element(page.getByText('old ops')).not.toBeInTheDocument();
  await toggle.click();
  await expect.element(page.getByText('old ops')).toBeInTheDocument();
});

test('an open pinned session leaves its agent collapsed', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    selectedId: 'p2',
    rows: pinRows,
  });
  await expect.element(page.getByRole('button', { name: /beta pin/ })).toBeInTheDocument();
  await expect
    .element(page.getByRole('button', { name: /^homelab/ }))
    .toHaveAttribute('aria-expanded', 'false');
});

test('opening another session of a collapsed agent expands it again', async () => {
  const { rerender } = await render(SessionsRail, {
    ...base,
    agents: twoAgents,
    selectedId: 'o1',
    rows: opsRows,
  });
  const toggle = page.getByRole('button', { name: /^homelab/ });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
  await toggle.click();
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');

  await rerender({ selectedId: 'o2' });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');
});

test('an open session of an unknown agent expands the unknown group', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    selectedId: 'x1',
    rows: [row('x1', { title: 'stray run', agent_id: 'gone' })],
  });
  await expect.element(page.getByRole('button', { name: /stray run/ })).toBeInTheDocument();
});

test('an agent lists its ended sessions behind a row styled like "show more"', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    rows: [...sevenRows, row('m9', { title: 'old main', agent_id: 'main', status: 'completed' })],
  });
  const more = await page.getByRole('button', { name: 'show 2 more' }).element();
  const ended = await page.getByRole('button', { name: '1 ended' }).element();
  expect(document.querySelector('.rail-scroll .sb-ended-hd')).toBeNull();
  const look = (el: Element) => {
    const st = getComputedStyle(el);
    return [st.paddingLeft, st.font, st.color, st.borderTopWidth];
  };
  expect(look(ended)).toEqual(look(more));

  await page.getByRole('button', { name: '1 ended' }).click();
  await expect.element(page.getByRole('button', { name: /old main/ })).toBeInTheDocument();
});

function rgb(color: string): number[] {
  const ctx = document.createElement('canvas').getContext('2d')!;
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 1, 1);
  return [...ctx.getImageData(0, 0, 1, 1).data.slice(0, 3)];
}

function contrast(a: number[], b: number[]): number {
  const lum = (c: number[]) => {
    const [r, g, bl] = c.map((v) => {
      const s = v / 255;
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r! + 0.7152 * g! + 0.0722 * bl!;
  };
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

test('the agent needs-you count stays readable in every theme', async () => {
  render(SessionsRail, {
    ...base,
    agents: twoAgents,
    attn: new Set(['o1']),
    rows: [row('o1', { title: 'disk alert', agent_id: 'ops' })],
  });
  await expect.element(page.getByRole('button', { name: /^homelab/ })).toBeInTheDocument();
  const badge = document.querySelector('.agent-attn')!;
  try {
    for (const theme of ['mocha', 'macchiato', 'frappe', 'latte', 'gruvbox']) {
      document.documentElement.dataset.theme = theme;
      const style = getComputedStyle(badge);
      expect(contrast(rgb(style.color), rgb(style.backgroundColor)), theme).toBeGreaterThanOrEqual(
        4.5,
      );
    }
  } finally {
    delete document.documentElement.dataset.theme;
  }
});
