/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import Composer from './Composer.svelte';
import { TESTID } from '$lib/testids';
import type { RefItem, RefSource } from './types';

const REF_ITEMS: RefItem[] = [
  { id: 'f1', kind: 'file', label: '@sse-reconnect.md', detail: 'kb/ops · modified', git: 'm' },
  { id: 'c1', kind: 'chat', label: '@sse-reconnect-backoff', detail: 'chat · working' },
  { id: 'a1', kind: 'agent', label: '@odyn', detail: 'agent · opus-4-8' },
];

test('typing an @ token opens the popover filtered to matching references', async () => {
  render(Composer, { refItems: REF_ITEMS });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'ping @sse');
  await expect.element(page.getByRole('listbox')).toBeInTheDocument();
  // 'sse' matches the two reconnect refs, not @odyn.
  expect(page.getByRole('option').elements()).toHaveLength(2);
});

test('ArrowDown moves the roving highlight to the next option', async () => {
  render(Composer, { refItems: REF_ITEMS });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'ping @sse');
  await expect
    .element(page.getByRole('option', { selected: true }))
    .toHaveTextContent('@sse-reconnect.md');
  await userEvent.keyboard('{ArrowDown}');
  await expect
    .element(page.getByRole('option', { selected: true }))
    .toHaveTextContent('@sse-reconnect-backoff');
  // The textarea keeps focus and points at the active option.
  await expect.element(box).toHaveAttribute('aria-activedescendant');
});

test('Enter selects the highlighted reference, inserts it, and closes the popover', async () => {
  render(Composer, { refItems: REF_ITEMS });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'ping @sse');
  await userEvent.keyboard('{ArrowDown}');
  await userEvent.keyboard('{Enter}');
  await expect.element(box).toHaveValue('ping @sse-reconnect-backoff ');
  // Popover closed: the textarea no longer points at an active option.
  await expect.element(box).not.toHaveAttribute('aria-activedescendant');
});

test('selecting a file ref with onPickRef attaches it and strips the @token, inserting no text', async () => {
  const onPickRef = vi.fn();
  render(Composer, { refItems: REF_ITEMS, onPickRef });
  const box = page.getByRole('textbox', { name: 'Message' });
  // Only the file ref matches this query, so it is the highlighted pick.
  await userEvent.fill(box, 'see @sse-reconnect.md');
  await userEvent.keyboard('{Enter}');
  expect(onPickRef).toHaveBeenCalledWith(REF_ITEMS[0]);
  // Convergence: the ref becomes a chip on the host, so the @query trigger is
  // removed and nothing is inserted in its place.
  await expect.element(box).toHaveValue('see ');
});

test('selecting a non-file ref still inserts its text inline even with onPickRef set', async () => {
  const onPickRef = vi.fn();
  render(Composer, { refItems: REF_ITEMS, onPickRef });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'ping @odyn');
  await userEvent.keyboard('{Enter}');
  // A chat/agent ref is not a workspace file: it inserts inline as before.
  expect(onPickRef).not.toHaveBeenCalled();
  await expect.element(box).toHaveValue('ping @odyn ');
});

test('a session ref matches by id substring, not just its title, and attaches on pick', async () => {
  const onPickRef = vi.fn();
  const items: RefItem[] = [
    { id: 'sess-abc123', kind: 'session', label: 'Nightly backup', detail: 'active · 2h' },
  ];
  render(Composer, { refItems: items, onPickRef });
  const box = page.getByRole('textbox', { name: 'Message' });
  // The id, not the title, carries 'abc123' - the popover still finds it.
  await userEvent.fill(box, 'ref @abc123');
  await expect.element(page.getByRole('option', { name: /Nightly backup/ })).toBeInTheDocument();
  await userEvent.keyboard('{Enter}');
  // A session attaches like a file: onPickRef fires and the @token is stripped.
  expect(onPickRef).toHaveBeenCalledWith(items[0]);
  await expect.element(box).toHaveValue('ref ');
});

test('a plain @query filters the built-in list and never queries a prefix source', async () => {
  const search = vi.fn(async () => []);
  render(Composer, {
    refItems: [{ id: 'notes.md', kind: 'file', label: 'notes.md' }],
    refSources: [{ prefix: 'jira', label: 'Jira', search }] satisfies RefSource[],
  });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'see @notes');
  await expect.element(page.getByRole('option', { name: /notes\.md/ })).toBeInTheDocument();
  // No `<prefix> ` in the query, so the source is left alone.
  expect(search).not.toHaveBeenCalled();
});

test('typing @<prefix> <query> fetches the source and a pick attaches the plugin item', async () => {
  const onPickRef = vi.fn();
  const result: RefItem = {
    id: 'PROJ-1',
    kind: 'plugin',
    label: 'auth login',
    providerKey: 'jira',
  };
  const search = vi.fn(async (q: string) => (q === 'auth' ? [result] : []));
  render(Composer, {
    refItems: [],
    refSources: [{ prefix: 'jira', label: 'Jira', search }] satisfies RefSource[],
    onPickRef,
  });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'see @jira auth');
  // The subquery (not the whole `jira auth`) is what the source receives.
  await vi.waitFor(() => expect(search).toHaveBeenCalledWith('auth'));
  await expect.element(page.getByRole('option', { name: /auth login/ })).toBeInTheDocument();
  await userEvent.keyboard('{Enter}');
  expect(onPickRef).toHaveBeenCalledWith(result);
  // The whole `@jira auth` token is stripped (it attaches, not inserts).
  await expect.element(box).toHaveValue('see ');
});

test('a stale (out-of-order) source response is dropped for the latest query', async () => {
  const resolvers: Record<string, (v: RefItem[]) => void> = {};
  const search = vi.fn(
    (q: string) =>
      new Promise<RefItem[]>((res) => {
        resolvers[q] = res;
      }),
  );
  render(Composer, {
    refItems: [],
    refSources: [{ prefix: 'jira', label: 'Jira', search }] satisfies RefSource[],
  });
  const box = page.getByRole('textbox', { name: 'Message' });

  await userEvent.fill(box, 'x @jira aa');
  await vi.waitFor(() => expect(search).toHaveBeenCalledWith('aa'));
  await userEvent.type(box, 'b'); // -> @jira aab
  await vi.waitFor(() => expect(search).toHaveBeenCalledWith('aab'));

  // Resolve the LATEST query first, then let the earlier (now stale) one land.
  resolvers['aab']?.([{ id: 'FRESH', kind: 'plugin', label: 'fresh hit', providerKey: 'jira' }]);
  resolvers['aa']?.([{ id: 'STALE', kind: 'plugin', label: 'stale hit', providerKey: 'jira' }]);

  await expect.element(page.getByRole('option', { name: /fresh hit/ })).toBeInTheDocument();
  // The stale response for the superseded query must never replace the results.
  expect(page.getByRole('option', { name: /stale hit/ }).elements()).toHaveLength(0);
});

test('Enter while a source shows "No matches" is swallowed, not sent', async () => {
  const onSend = vi.fn();
  const search = vi.fn(async () => []); // always empty -> the No-matches state
  render(Composer, {
    refItems: [],
    refSources: [{ prefix: 'jira', label: 'Jira', search }] satisfies RefSource[],
    onSend,
  });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'go @jira zzz');
  await vi.waitFor(() => expect(search).toHaveBeenCalledWith('zzz'));
  await expect.element(page.getByText('No matches')).toBeInTheDocument();
  // The popover is open mid-search, so Enter dismisses to it, never sending.
  await userEvent.keyboard('{Enter}');
  expect(onSend).not.toHaveBeenCalled();
});

test('a picker provider commits immediately (no inline submenu) so the host opens the Picker', async () => {
  const onPickContext = vi.fn();
  const onRequestChoices = vi.fn().mockResolvedValue([{ value: 'a.md', label: 'a.md' }]);
  const item = {
    key: 'file',
    label: 'Workspace file',
    icon: 'file' as const,
    kind: 'server' as const,
    hasChoices: true,
    picker: true,
  };
  render(Composer, { contextMenu: [item], onPickContext, onRequestChoices });
  await page.getByTestId(TESTID.composerContext).click();
  (page.getByTestId(TESTID.composerContextOption('file')).element() as HTMLElement).click();

  // Picking it commits the whole item (no arg); the host will open the Picker.
  expect(onPickContext).toHaveBeenCalledWith(item);
  // It must not fall into the inline-submenu path.
  expect(onRequestChoices).not.toHaveBeenCalled();
  await expect.element(page.getByTestId(TESTID.composerContextSubmenu)).not.toBeInTheDocument();
});

test('Escape closes the popover without sending', async () => {
  const onSend = vi.fn();
  render(Composer, { refItems: REF_ITEMS, onSend });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'ping @sse');
  await expect.element(page.getByRole('listbox')).toBeInTheDocument();
  await userEvent.keyboard('{Escape}');
  await expect.element(box).not.toHaveAttribute('aria-activedescendant');
  expect(onSend).not.toHaveBeenCalled();
});

test('Enter with no popover sends the trimmed text and clears the input', async () => {
  const onSend = vi.fn();
  render(Composer, { onSend });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, '  hello world  ');
  await userEvent.keyboard('{Enter}');
  expect(onSend).toHaveBeenCalledWith('hello world');
  await expect.element(box).toHaveValue('');
});

test('on a touch device (pointer: coarse) Enter does not send', async () => {
  const onSend = vi.fn();
  const orig = window.matchMedia;
  window.matchMedia = ((q: string) => ({
    matches: q.includes('coarse'),
    media: q,
    addEventListener() {},
    removeEventListener() {},
    // legacy fields some libs read
    onchange: null,
    addListener() {},
    removeListener() {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
  try {
    render(Composer, { onSend });
    const box = page.getByRole('textbox', { name: 'Message' });
    await userEvent.fill(box, 'line one');
    await userEvent.keyboard('{Enter}');
    expect(onSend).not.toHaveBeenCalled();
  } finally {
    window.matchMedia = orig;
  }
});

test('Shift+Enter does not send', async () => {
  const onSend = vi.fn();
  render(Composer, { onSend });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'line one');
  await userEvent.keyboard('{Shift>}{Enter}{/Shift}');
  expect(onSend).not.toHaveBeenCalled();
});

test('while streaming the send button becomes Stop and triggers onStop', async () => {
  const onSend = vi.fn();
  const onStop = vi.fn();
  render(Composer, { streaming: true, value: 'work in progress', onSend, onStop });
  const stop = page.getByRole('button', { name: 'Stop streaming' });
  await expect.element(stop).toBeInTheDocument();
  await userEvent.click(stop);
  expect(onStop).toHaveBeenCalledTimes(1);
  expect(onSend).not.toHaveBeenCalled();
});

test('mid-turn Enter with a draft queues it instead of stopping', async () => {
  const onStop = vi.fn();
  const onQueue = vi.fn();
  render(Composer, { streaming: true, onStop, onQueue });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'follow-up question');
  await userEvent.keyboard('{Enter}');
  expect(onQueue).toHaveBeenCalledWith('follow-up question');
  expect(onStop).not.toHaveBeenCalled();
  await expect.element(box).toHaveValue('');
});

test('mid-turn Enter with an empty draft still stops', async () => {
  const onStop = vi.fn();
  const onQueue = vi.fn();
  render(Composer, { streaming: true, onStop, onQueue });
  (page.getByRole('textbox', { name: 'Message' }).element() as HTMLElement).focus();
  await userEvent.keyboard('{Enter}');
  expect(onStop).toHaveBeenCalledTimes(1);
  expect(onQueue).not.toHaveBeenCalled();
});

test('the Queue button shows only mid-turn, disables on empty, and fires onQueue', async () => {
  const onQueue = vi.fn();
  const idle = await render(Composer, { streaming: false, onQueue });
  expect(idle.container.querySelector('[data-act="queue"]')).toBeNull();
  idle.unmount();

  render(Composer, { streaming: true, onQueue });
  const queueBtn = page.getByRole('button', { name: 'Queue message for after this turn' });
  await expect.element(queueBtn).toBeDisabled();
  await userEvent.fill(page.getByRole('textbox', { name: 'Message' }), 'next task');
  await expect.element(queueBtn).toBeEnabled();
  await queueBtn.click();
  expect(onQueue).toHaveBeenCalledWith('next task');
});

test('the input grows with multi-line content and shrinks back', async () => {
  render(Composer, {});
  const box = page.getByRole('textbox', { name: 'Message' });
  const el = box.element() as HTMLTextAreaElement;
  const base = el.clientHeight;

  await userEvent.fill(box, Array.from({ length: 8 }, (_, i) => `line ${i}`).join('\n'));
  await expect.poll(() => el.clientHeight).toBeGreaterThan(base + 40);

  await userEvent.fill(box, 'short again');
  await expect.poll(() => el.clientHeight).toBeLessThan(base + 20);
});

test('the context button opens a menu of providers; picking one fires onPickContext', async () => {
  const onPickContext = vi.fn();
  render(Composer, {
    contextMenu: [{ key: 'location', label: 'Location', icon: 'pin' }],
    onPickContext,
  });
  await page.getByTestId(TESTID.composerContext).click();
  await expect.element(page.getByTestId(TESTID.composerContextMenu)).toBeInTheDocument();
  // The menu floats above the composer, off the top of an isolated test's
  // viewport, so fire the DOM click on the raw element (as the chip-× test does).
  const opt = page.getByTestId(TESTID.composerContextOption('location')).element() as HTMLElement;
  opt.click();
  // A client/no-choices pick commits the whole menu item with no arg.
  expect(onPickContext).toHaveBeenCalledWith({ key: 'location', label: 'Location', icon: 'pin' });
});

test('a hasChoices provider opens a submenu; picking an option commits it with the value', async () => {
  const onPickContext = vi.fn();
  const onRequestChoices = vi.fn().mockResolvedValue([
    { value: 't1', label: 'npm test' },
    { value: 't2', label: 'server log' },
  ]);
  render(Composer, {
    contextMenu: [
      { key: 'terminal', label: 'Terminal output', icon: 'term', kind: 'server', hasChoices: true },
    ],
    onPickContext,
    onRequestChoices,
  });
  await page.getByTestId(TESTID.composerContext).click();
  (page.getByTestId(TESTID.composerContextOption('terminal')).element() as HTMLElement).click();

  // Picking the choices provider loads its options rather than committing.
  await expect.element(page.getByTestId(TESTID.composerContextSubmenu)).toBeInTheDocument();
  expect(onRequestChoices).toHaveBeenCalledWith('terminal');
  expect(onPickContext).not.toHaveBeenCalled();

  const choice = page
    .getByTestId(TESTID.composerContextChoice('terminal', 't2'))
    .element() as HTMLElement;
  choice.click();
  expect(onPickContext).toHaveBeenCalledWith(
    { key: 'terminal', label: 'Terminal output', icon: 'term', kind: 'server', hasChoices: true },
    't2',
  );
});

test('a context chip shows its short label (not the value) and its X removes it', async () => {
  const onRemoveContext = vi.fn();
  render(Composer, {
    contextItems: [
      { key: 'location', label: 'Location', value: '37.77490, -122.41940 (±20m)', icon: 'pin' },
    ],
    onRemoveContext,
  });
  const chip = page.getByTestId(TESTID.composerContextChip('location'));
  // The chip shows the label; the value (which can be a whole file) stays out of
  // the row and only appears in the preview modal.
  await expect.element(chip).toHaveTextContent('Location');
  await expect.element(chip).not.toHaveTextContent('37.77490');
  // The chip's × is icon-only; its size comes from global icon CSS not loaded in
  // isolation, so fire the DOM click to prove the handler wiring (visibility is
  // covered by the live Playwright verify, where real CSS applies).
  const x = page.getByRole('button', { name: 'Remove Location context' }).element() as HTMLElement;
  x.click();
  expect(onRemoveContext).toHaveBeenCalledWith('location');
});

test('clicking a context chip opens a modal showing the full value', async () => {
  const value = Array.from({ length: 40 }, (_, i) => `line ${i} of the fetched page`).join('\n');
  render(Composer, {
    contextItems: [
      { key: 'page', label: 'https://example.com/very/long/url', value, icon: 'link' },
    ],
  });
  // Closed: the modal dialog is not in the a11y tree (its scrim is display:none).
  expect(page.getByRole('dialog').elements()).toHaveLength(0);
  await page.getByRole('button', { name: 'https://example.com/very/long/url' }).click();
  const dialog = page.getByRole('dialog', { name: 'https://example.com/very/long/url' });
  await expect.element(dialog).toBeInTheDocument();
  await expect.element(dialog).toHaveTextContent('line 0 of the fetched page');
  await expect.element(dialog).toHaveTextContent('line 39 of the fetched page');
  // Focus lands inside the dialog, so Esc closes the preview.
  await userEvent.keyboard('{Escape}');
  await vi.waitFor(() => expect(page.getByRole('dialog').elements()).toHaveLength(0));
});

test('the attach and context buttons stay present alongside two oversized chips', async () => {
  const path = '/very/long/workspace/path/that/would/overflow/the/composer/row/file.md';
  render(Composer, {
    contextItems: [
      { key: 'a', label: `${path}-a`, value: 'A'.repeat(6000), icon: 'file' },
      { key: 'b', label: `${path}-b`, value: 'B'.repeat(6000), icon: 'file' },
    ],
    contextMenu: [{ key: 'location', label: 'Location', icon: 'pin' }],
  });
  // The huge values live behind the preview modal, not in the row, so the row's
  // own controls stay in the DOM (pixel visibility at phone width is the live
  // Playwright verify's job, where real CSS applies).
  await expect.element(page.getByRole('button', { name: 'attach' })).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.composerContext)).toBeInTheDocument();
});

test('an attached context item makes an empty composer sendable', async () => {
  const onSend = vi.fn();
  render(Composer, { contextItems: [{ key: 'location', label: 'Location', value: 'x' }], onSend });
  // No typed text, but Send fires with empty text - the item rides as metadata.
  await page.getByRole('button', { name: 'Send message' }).click();
  expect(onSend).toHaveBeenCalledWith('');
});

test('with no text and no context, Send does nothing', async () => {
  const onSend = vi.fn();
  render(Composer, { onSend });
  await page.getByRole('button', { name: 'Send message' }).click();
  expect(onSend).not.toHaveBeenCalled();
});

test('a huge paste caps the input height and scrolls inside', async () => {
  render(Composer, {});
  const box = page.getByRole('textbox', { name: 'Message' });
  const el = box.element() as HTMLTextAreaElement;

  await userEvent.fill(box, Array.from({ length: 200 }, (_, i) => `line ${i}`).join('\n'));
  await expect.poll(() => el.clientHeight).toBeLessThanOrEqual(window.innerHeight * 0.4);
  expect(el.scrollHeight).toBeGreaterThan(el.clientHeight);
});
