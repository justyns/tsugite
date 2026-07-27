import { describe, expect, test, vi } from 'vitest';
import {
  buildPaletteItems,
  buildSessionItems,
  commandArgParam,
  commandNeedsInput,
  commandPaletteAction,
  runPaletteHref,
  type PaletteData,
} from './palette-sources';

const data: PaletteData = {
  views: [
    { id: 'chats', label: 'Chats', icon: 'chat' },
    { id: 'jobs', label: 'Jobs', icon: 'jobs' },
  ],
  themes: ['mocha', 'latte'],
  currentTheme: 'mocha',
  spaces: [
    { id: 's1', name: 'Main' },
    { id: 's2', name: 'Ops' },
  ],
  activeSpaceId: 's1',
  commands: [
    {
      name: 'status',
      description: 'Show agent status and context usage',
      params: [{ name: 'user_id' }, { name: 'session_id' }],
    },
    {
      name: 'model',
      description: "Show or switch this chat's model",
      params: [{ name: 'user_id' }, { name: 'message' }, { name: 'session_id' }],
    },
  ],
};

describe('buildPaletteItems', () => {
  test('encodes every row as a dispatchable command href', () => {
    const items = buildPaletteItems(data);
    expect(items.find((i) => i.label === 'Jobs')?.href).toBe('view:jobs');
    expect(items.find((i) => i.label === 'Ops')?.href).toBe('space:s2');
    expect(items.find((i) => i.label === 'latte')?.href).toBe('theme:latte');
    expect(items.some((i) => i.href === 'action:settings')).toBe(true);
  });

  test('marks the active space and current theme as current', () => {
    const items = buildPaletteItems(data);
    expect(items.find((i) => i.href === 'space:s1')?.meta).toBe('current');
    expect(items.find((i) => i.href === 'theme:mocha')?.meta).toBe('current');
    expect(items.find((i) => i.href === 'theme:latte')?.meta).not.toBe('current');
  });

  test('carries the view glyph so the palette row matches the nav rail', () => {
    const items = buildPaletteItems(data);
    expect(items.find((i) => i.href === 'view:jobs')?.icon).toBe('jobs');
  });

  test('groups in scannable order: views, spaces, theme, actions, commands', () => {
    const groups = [...new Set(buildPaletteItems(data).map((i) => i.group))];
    expect(groups).toEqual(['views', 'spaces', 'theme', 'actions', 'commands']);
  });

  test('leads the actions group with a searchable new-chat row', () => {
    const items = buildPaletteItems(data);
    const it = items.find((i) => i.href === 'action:new-chat');
    expect(it?.label).toBe('New chat');
    expect(it?.group).toBe('actions');
    expect(it?.keywords).toContain('conversation');
    const actions = items.filter((i) => i.group === 'actions');
    expect(actions[0]?.href).toBe('action:new-chat');
  });

  test('offers a keyboard-shortcuts action in the actions group', () => {
    const it = buildPaletteItems(data).find((i) => i.href === 'action:shortcuts');
    expect(it?.label).toBe('Keyboard shortcuts');
    expect(it?.group).toBe('actions');
  });

  test('emits one dispatchable row per slash command, matchable by description', () => {
    const it = buildPaletteItems(data).find((i) => i.href === 'command:status');
    expect(it?.group).toBe('commands');
    expect(it?.label).toBe('/status');
    expect(it?.meta).toBe('Show agent status and context usage');
    expect(it?.keywords).toContain('context');
  });

  test('falls back to a generic meta for a description-less command', () => {
    const items = buildPaletteItems({
      ...data,
      commands: [{ name: 'ping', description: '', params: [] }],
    });
    expect(items.find((i) => i.href === 'command:ping')?.meta).toBe('command');
  });
});

describe('commandNeedsInput', () => {
  test('an all-auto-injected command runs without input', () => {
    expect(commandNeedsInput({ params: [{ name: 'user_id' }, { name: 'session_id' }] })).toBe(
      false,
    );
  });

  test('a paramless command runs without input', () => {
    expect(commandNeedsInput({ params: [] })).toBe(false);
  });

  test('any user-facing param (primary or choice) needs input', () => {
    expect(
      commandNeedsInput({
        params: [{ name: 'user_id' }, { name: 'message' }, { name: 'session_id' }],
      }),
    ).toBe(true);
    expect(commandNeedsInput({ params: [{ name: 'status' }] })).toBe(true);
    expect(commandNeedsInput({ params: [{ name: 'prompt' }] })).toBe(true);
  });
});

describe('commandArgParam', () => {
  test('picks a known free-text param over an auto-injected one', () => {
    const p = commandArgParam({
      params: [{ name: 'user_id' }, { name: 'message', widget: 'model' }, { name: 'session_id' }],
    });
    expect(p?.name).toBe('message');
    expect(p?.widget).toBe('model');
  });

  test('falls back to the first user-facing param when none is a known text field', () => {
    const p = commandArgParam({
      params: [{ name: 'status', choices: ['running', 'completed', 'failed'] }],
    });
    expect(p?.name).toBe('status');
    expect(p?.choices).toEqual(['running', 'completed', 'failed']);
  });

  test('an all-auto-injected command has no argument param', () => {
    expect(
      commandArgParam({ params: [{ name: 'user_id' }, { name: 'session_id' }] }),
    ).toBeUndefined();
  });
});

describe('commandPaletteAction', () => {
  test('a widget=model command opens the model picker rather than a prefill', () => {
    expect(
      commandPaletteAction({
        name: 'model',
        params: [{ name: 'user_id' }, { name: 'message', widget: 'model' }, { name: 'session_id' }],
      }),
    ).toEqual({ kind: 'model-picker' });
  });

  test('a widget=effort command prefills so the composer can offer its choices', () => {
    expect(
      commandPaletteAction({
        name: 'effort',
        params: [
          { name: 'user_id' },
          { name: 'message', widget: 'effort' },
          { name: 'session_id' },
        ],
      }),
    ).toEqual({ kind: 'prefill', text: '/effort ' });
  });

  test('a static-choices command prefills for its inline choices', () => {
    expect(
      commandPaletteAction({
        name: 'sessions',
        params: [{ name: 'status', choices: ['running'] }],
      }),
    ).toEqual({ kind: 'prefill', text: '/sessions ' });
  });

  test('an all-auto-injected command runs outright', () => {
    expect(
      commandPaletteAction({
        name: 'status',
        params: [{ name: 'user_id' }, { name: 'session_id' }],
      }),
    ).toEqual({ kind: 'run', text: '/status' });
  });
});

describe('buildSessionItems', () => {
  const rows = [
    { id: 's1', title: 'refactor sse', ended: false, when: '2m', topic: 'sse reconnect' },
    { id: 's2', title: 'old cleanup', ended: true, when: 'jul 1', topic: '' },
    { id: 's3', title: 'live triage', ended: false, when: 'now', topic: 'inbox' },
  ];

  test('maps each session to a session-scheme jump row', () => {
    const it = buildSessionItems(rows).find((i) => i.label === 'refactor sse');
    expect(it?.group).toBe('sessions');
    expect(it?.href).toBe('session:s1');
    expect(it?.icon).toBe('chat');
    expect(it?.meta).toBe('2m');
    expect(it?.keywords).toBe('sse reconnect');
  });

  test('orders live sessions before ended ones, keeping input recency within each', () => {
    expect(buildSessionItems(rows).map((i) => i.label)).toEqual([
      'refactor sse',
      'live triage',
      'old cleanup',
    ]);
  });

  test('drops superseded sessions so a compacted chat is not listed twice', () => {
    const withSuperseded = [
      { id: 's1', title: 'long thread', ended: false, when: '5m', topic: '', superseded: true },
      { id: 's1b', title: 'long thread', ended: false, when: '2m', topic: '' },
    ];
    const labels = buildSessionItems(withSuperseded).map((i) => i.label);
    expect(labels).toEqual(['long thread']);
    expect(buildSessionItems(withSuperseded).map((i) => i.href)).toEqual(['session:s1b']);
  });
});

describe('runPaletteHref', () => {
  const makeCtx = () => ({
    openView: vi.fn(),
    setTheme: vi.fn(),
    setSpace: vi.fn(),
    openSettings: vi.fn(),
    openSession: vi.fn(),
    newChat: vi.fn(),
    showHelp: vi.fn(),
    runCommand: vi.fn(),
  });

  test('dispatches a view jump', () => {
    const ctx = makeCtx();
    expect(runPaletteHref('view:jobs', ctx)).toBe(true);
    expect(ctx.openView).toHaveBeenCalledWith('jobs');
  });

  test('dispatches a theme set', () => {
    const ctx = makeCtx();
    runPaletteHref('theme:latte', ctx);
    expect(ctx.setTheme).toHaveBeenCalledWith('latte');
  });

  test('dispatches a space switch', () => {
    const ctx = makeCtx();
    runPaletteHref('space:s2', ctx);
    expect(ctx.setSpace).toHaveBeenCalledWith('s2');
  });

  test('dispatches the settings action', () => {
    const ctx = makeCtx();
    runPaletteHref('action:settings', ctx);
    expect(ctx.openSettings).toHaveBeenCalledOnce();
  });

  test('dispatches the new-chat action', () => {
    const ctx = makeCtx();
    expect(runPaletteHref('action:new-chat', ctx)).toBe(true);
    expect(ctx.newChat).toHaveBeenCalledOnce();
  });

  test('dispatches the shortcuts action', () => {
    const ctx = makeCtx();
    expect(runPaletteHref('action:shortcuts', ctx)).toBe(true);
    expect(ctx.showHelp).toHaveBeenCalledOnce();
  });

  test('dispatches a session jump', () => {
    const ctx = makeCtx();
    expect(runPaletteHref('session:abc123', ctx)).toBe(true);
    expect(ctx.openSession).toHaveBeenCalledWith('abc123');
  });

  test('dispatches a slash command by name', () => {
    const ctx = makeCtx();
    expect(runPaletteHref('command:status', ctx)).toBe(true);
    expect(ctx.runCommand).toHaveBeenCalledWith('status');
  });

  test('an id with a colon in it survives the split', () => {
    const ctx = makeCtx();
    runPaletteHref('space:a:b', ctx);
    expect(ctx.setSpace).toHaveBeenCalledWith('a:b');
  });

  test('an unknown or empty href is a no-op', () => {
    const ctx = makeCtx();
    expect(runPaletteHref(undefined, ctx)).toBe(false);
    expect(runPaletteHref('nope:x', ctx)).toBe(false);
    expect(ctx.openView).not.toHaveBeenCalled();
  });
});
