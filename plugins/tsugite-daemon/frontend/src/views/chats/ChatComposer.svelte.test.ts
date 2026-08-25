/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import ChatComposer from './ChatComposer.svelte';
import { api } from '$lib/api/client';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { composerPrefill } from './composerPrefill.svelte';
import { contextAttach } from './contextAttach.svelte';
import { modelPickerRequest } from './modelPickerSignal.svelte';
import { contextProviders } from '$lib/context/contextProviders';
import { autoAttachStore } from '$lib/stores/autoAttach.svelte';
import { sessions, type SessionRow } from '$lib/stores/sessions.svelte';
import { TESTID } from '$lib/testids';

const GEO_KEY = 'tsugite_geo_autoattach';

// The sessions store is a singleton; reset its rows so one test's seeded chats
// can't leak into another's @ popover.
beforeEach(() => {
  localStorage.clear();
  sessions.rows = [];
});

function sessionRow(over: Partial<SessionRow>): SessionRow {
  return {
    id: 'x',
    title: null,
    status: 'active',
    last_active: null,
    pinned: false,
    pin_position: null,
    ...over,
  } as SessionRow;
}
// Restore stacked spies so one test's in-flight upload can't resolve into the
// next test's spy (uploadChosen awaits the image config before calling upload).
// Drain any prefill / model-picker request a test left unconsumed so it can't leak.
// The auto-attach store singletons persist in memory across cases, so reset them.
afterEach(() => {
  vi.restoreAllMocks();
  for (const p of contextProviders)
    if (p.autoAttachStoreKey) autoAttachStore(p.autoAttachStoreKey).set(false);
  composerPrefill.consume(base.sessionId);
  const attach = contextAttach.pending;
  if (attach) contextAttach.consume(attach.sessionId);
  const p = modelPickerRequest.pending;
  if (p) modelPickerRequest.consume(p.sessionId);
});

/** Open the "add context" menu and pick the location provider. The menu floats
 *  off the top of an isolated test's viewport, so fire the raw-element click. */
async function pickLocation() {
  await page.getByTestId(TESTID.composerContext).click();
  const opt = page.getByTestId(TESTID.composerContextOption('location')).element() as HTMLElement;
  opt.click();
}

/** Open the "add context" menu, then click a provider row by key. The menu floats
 *  off the top of an isolated test's viewport, so fire the raw-element click. */
async function openContextMenu() {
  await page.getByTestId(TESTID.composerContext).click();
}
function clickContextOption(key: string) {
  (page.getByTestId(TESTID.composerContextOption(key)).element() as HTMLElement).click();
}

type ServerWire = {
  key: string;
  label: string;
  icon: string;
  has_choices: boolean;
  picker?: boolean;
};
/** Route /api/context-providers (and each provider's /choices) through the GET
 *  stub so the composer's server-provider load + submenu fetch resolve to fixtures. */
function stubServerMenu(
  providers: ServerWire[],
  choices?: Record<string, { value: string; label: string }[]>,
) {
  stubCommands([], (path) => {
    if (path === '/api/context-providers') return Promise.resolve({ providers });
    const m = /\/context-providers\/([^/]+)\/choices/.exec(path);
    const key = m?.[1];
    if (key && choices?.[key]) return Promise.resolve({ choices: choices[key] });
    return undefined;
  });
}

/** Drive navigator.geolocation.getCurrentPosition to a fixed fix or error code. */
function mockGeoSuccess(coords: { latitude: number; longitude: number; accuracy: number }) {
  vi.spyOn(navigator.geolocation, 'getCurrentPosition').mockImplementation((ok) =>
    ok({ coords } as GeolocationPosition),
  );
}
function mockGeoError(code: 1 | 2 | 3) {
  vi.spyOn(navigator.geolocation, 'getCurrentPosition').mockImplementation((_ok, err) =>
    err?.({ code, message: '', PERMISSION_DENIED: 1, POSITION_UNAVAILABLE: 2, TIMEOUT: 3 }),
  );
}

/** Stub the /api/health config fetch so loadImageConfig resolves instantly
 *  (the test proxy black-holes it, which is slow enough to race across tests). */
function stubImageConfig() {
  vi.spyOn(api, 'get').mockResolvedValue({ images: { max_edge: 1568, quality: 0.85 } });
}

type StubParam = { name: string; widget?: string; choices?: string[] };
/** Route the composer's two GETs: its /api/commands list (so a run-request can
 *  resolve the command) and the image config every render pulls. Extra GETs (e.g.
 *  the effort-levels probe) resolve via `extra` when provided, else as image config. */
function stubCommands(
  commands: { name: string; description: string; params: StubParam[] }[],
  extra?: (path: string) => Promise<unknown> | undefined,
) {
  vi.spyOn(api, 'get').mockImplementation((path: string) => {
    if (path === '/api/commands') return Promise.resolve({ commands });
    const e = extra?.(path);
    if (e) return e as Promise<unknown>;
    return Promise.resolve({ images: { max_edge: 1568, quality: 0.85 } });
  });
}

const base = {
  sessionId: 's1',
  onSend: vi.fn(),
  onStop: vi.fn(),
};

async function makePng(w: number, h: number): Promise<File> {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  c.getContext('2d')!.fillRect(0, 0, w, h);
  const blob = await new Promise<Blob>((res) => c.toBlob((b) => res(b!), 'image/png'));
  return new File([blob], 'photo.png', { type: 'image/png' });
}

/** Build a `paste` whose clipboardData is our DataTransfer. Some engines ignore
 *  `clipboardData` in the init dict, so force it on when the constructor drops it. */
function pasteEvent(dt: DataTransfer): ClipboardEvent {
  const e = new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true });
  if (e.clipboardData !== dt)
    Object.defineProperty(e, 'clipboardData', { value: dt, configurable: true });
  return e;
}

test('a failed send restores its text into an empty composer', async () => {
  const { rerender } = await render(ChatComposer, { ...base, restoreFailed: null });
  const box = page.getByRole('textbox', { name: 'Message' });
  await expect.element(box).toHaveValue('');

  await rerender({ restoreFailed: { text: 'lost message', seq: 1 } });
  await expect.element(box).toHaveValue('lost message');
});

test('a failed send never clobbers text typed since', async () => {
  const { rerender } = await render(ChatComposer, { ...base, restoreFailed: null });
  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'newer thought');

  await rerender({ restoreFailed: { text: 'older failed text', seq: 1 } });
  await expect.element(box).toHaveValue('newer thought');
});

test('the same failure restores at most once', async () => {
  const { rerender } = await render(ChatComposer, {
    ...base,
    restoreFailed: { text: 'first', seq: 1 },
  });
  const box = page.getByRole('textbox', { name: 'Message' });
  await expect.element(box).toHaveValue('first');

  await userEvent.fill(box, '');
  await rerender({ restoreFailed: { text: 'first', seq: 1 }, streaming: false });
  await expect.element(box).toHaveValue('');
});

test('re-encodes a chosen image to JPEG before upload', async () => {
  stubImageConfig();
  const upload = vi
    .spyOn(api, 'uploadFiles')
    .mockResolvedValue({ files: [{ name: 'photo.jpg', size: 1000 }] });
  await render(ChatComposer, { ...base });
  await userEvent.upload(page.getByTestId(TESTID.composerFileInput), await makePng(2400, 1600));
  await vi.waitFor(() => expect(upload).toHaveBeenCalled());
  const sent = Array.from((upload.mock.calls[0]?.[1] ?? []) as Iterable<File>);
  expect(sent[0]?.type).toBe('image/jpeg');
});

test('uploads a non-image file unchanged (not re-encoded to JPEG)', async () => {
  // userEvent.upload round-trips the File through disk, so assert by type/name
  // rather than reference identity: a non-image must not be touched.
  stubImageConfig();
  const upload = vi.spyOn(api, 'uploadFiles').mockResolvedValue({ files: [{ name: 'notes.txt' }] });
  await render(ChatComposer, { ...base });
  const txt = new File(['hello'], 'notes.txt', { type: 'text/plain' });
  await userEvent.upload(page.getByTestId(TESTID.composerFileInput), txt);
  await vi.waitFor(() => expect(upload).toHaveBeenCalled());
  const sent = Array.from((upload.mock.calls[0]?.[1] ?? []) as Iterable<File>);
  expect(sent[0]?.type).toBe('text/plain');
  expect(sent[0]?.name).toBe('notes.txt');
});

test('the generic attach input stays accept-less; the camera input targets the camera', async () => {
  await render(ChatComposer, { ...base });
  await expect.element(page.getByTestId(TESTID.composerFileInput)).not.toHaveAttribute('accept');
  const camera = page.getByTestId(TESTID.composerCameraInput);
  await expect.element(camera).toHaveAttribute('accept', 'image/*');
  await expect.element(camera).toHaveAttribute('capture', 'environment');
});

test('pasting an image attaches it and leaves the draft untouched', async () => {
  stubImageConfig();
  const upload = vi
    .spyOn(api, 'uploadFiles')
    .mockResolvedValue({ files: [{ name: 'pasted.jpg', size: 900 }] });
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });
  const el = box.element() as HTMLTextAreaElement;

  const dt = new DataTransfer();
  dt.items.add(await makePng(20, 20));
  // preventDefault (native paste blocked) is what dispatchEvent === false reports.
  const cancelled = !el.dispatchEvent(pasteEvent(dt));

  expect(cancelled).toBe(true);
  await vi.waitFor(() => expect(upload).toHaveBeenCalled());
  const sent = Array.from((upload.mock.calls[0]?.[1] ?? []) as Iterable<File>);
  expect(sent).toHaveLength(1);
  await expect.element(box).toHaveValue('');
});

test('a plain-text paste is left to native behavior (no attach)', async () => {
  stubImageConfig();
  const upload = vi.spyOn(api, 'uploadFiles').mockResolvedValue({ files: [] });
  await render(ChatComposer, { ...base });
  const el = page.getByRole('textbox', { name: 'Message' }).element() as HTMLTextAreaElement;

  const dt = new DataTransfer();
  dt.setData('text/plain', 'hello world');
  const notCancelled = el.dispatchEvent(pasteEvent(dt));

  expect(notCancelled).toBe(true);
  expect(upload).not.toHaveBeenCalled();
});

test('a paste carrying both text and an image prefers the image', async () => {
  stubImageConfig();
  const upload = vi
    .spyOn(api, 'uploadFiles')
    .mockResolvedValue({ files: [{ name: 'pasted.jpg' }] });
  await render(ChatComposer, { ...base });
  const el = page.getByRole('textbox', { name: 'Message' }).element() as HTMLTextAreaElement;

  const dt = new DataTransfer();
  dt.setData('text/html', '<img src="blob:...">');
  dt.items.add(await makePng(20, 20));
  const cancelled = !el.dispatchEvent(pasteEvent(dt));

  expect(cancelled).toBe(true);
  await vi.waitFor(() => expect(upload).toHaveBeenCalled());
});

function pasteText(el: HTMLTextAreaElement, text: string): boolean {
  const dt = new DataTransfer();
  dt.setData('text/plain', text);
  return !el.dispatchEvent(pasteEvent(dt));
}

const hasChooser = () =>
  Array.from(document.querySelectorAll('button')).some((b) =>
    b.textContent?.includes('Attach as .txt'),
  );

test('a large text paste shows the attach/inline chooser and preventDefaults', async () => {
  await render(ChatComposer, { ...base });
  const el = page.getByRole('textbox', { name: 'Message' }).element() as HTMLTextAreaElement;

  const cancelled = pasteText(el, 'x'.repeat(600));

  expect(cancelled).toBe(true);
  await expect.element(page.getByRole('button', { name: 'Attach as .txt' })).toBeInTheDocument();
});

test('a many-line paste under the char limit still shows the chooser', async () => {
  await render(ChatComposer, { ...base });
  const el = page.getByRole('textbox', { name: 'Message' }).element() as HTMLTextAreaElement;

  // 12 lines, well under 500 chars — the line count alone crosses the threshold.
  const cancelled = pasteText(el, Array.from({ length: 12 }, (_, i) => `line ${i}`).join('\n'));

  expect(cancelled).toBe(true);
  await expect.element(page.getByRole('button', { name: 'Attach as .txt' })).toBeInTheDocument();
});

test('choosing "attach as .txt" uploads the paste as a text file, draft untouched', async () => {
  stubImageConfig();
  const upload = vi
    .spyOn(api, 'uploadFiles')
    .mockResolvedValue({ files: [{ name: 'pasted.txt' }] });
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });
  const el = box.element() as HTMLTextAreaElement;

  pasteText(el, 'y'.repeat(600));
  await page.getByRole('button', { name: 'Attach as .txt' }).click();

  await vi.waitFor(() => expect(upload).toHaveBeenCalled());
  const sent = Array.from((upload.mock.calls[0]?.[1] ?? []) as Iterable<File>);
  expect(sent).toHaveLength(1);
  expect(sent[0]?.type).toBe('text/plain');
  expect(sent[0]?.name).toMatch(/^pasted-.*\.txt$/);
  await expect.element(box).toHaveValue('');
});

test('choosing "paste inline" inserts the text into the draft', async () => {
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });
  const el = box.element() as HTMLTextAreaElement;
  const big = 'z'.repeat(600);

  pasteText(el, big);
  await page.getByRole('button', { name: 'Paste inline' }).click();

  await expect.element(box).toHaveValue(big);
});

test('dismissing the chooser with Escape defaults to inline (text not lost)', async () => {
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });
  const el = box.element() as HTMLTextAreaElement;
  const big = 'w'.repeat(600);

  pasteText(el, big);
  await expect.element(page.getByRole('button', { name: 'Attach as .txt' })).toBeInTheDocument();
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

  await expect.element(box).toHaveValue(big);
  expect(hasChooser()).toBe(false);
});

test('a small text paste inserts natively without the chooser', async () => {
  await render(ChatComposer, { ...base });
  const el = page.getByRole('textbox', { name: 'Message' }).element() as HTMLTextAreaElement;

  const cancelled = pasteText(el, 'a short paragraph, nothing special');

  expect(cancelled).toBe(false);
  expect(hasChooser()).toBe(false);
});

test('a run prefill request for this session dispatches the command', async () => {
  stubCommands([
    {
      name: 'status',
      description: 'Show status',
      params: [{ name: 'user_id' }, { name: 'session_id' }],
    },
  ]);
  const post = vi.spyOn(api, 'post').mockResolvedValue({ result: 'ok' });
  await render(ChatComposer, { ...base });

  composerPrefill.request(base.sessionId, '/status', true);

  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toContain('/commands/status');
  // A run must not leave the command text sitting in the composer.
  await expect.element(page.getByRole('textbox', { name: 'Message' })).toHaveValue('');
});

test('a successful command echoes its result inline instead of toasting it', async () => {
  stubCommands([{ name: 'status', description: 'Show status', params: [] }]);
  vi.spyOn(api, 'post').mockResolvedValue({ result: 'Model: claude_code:haiku' });
  const onCommandResult = vi.fn();
  const push = vi.spyOn(toasts, 'push');
  await render(ChatComposer, { ...base, onCommandResult });

  composerPrefill.request(base.sessionId, '/status', true);

  await vi.waitFor(() => expect(onCommandResult).toHaveBeenCalled());
  expect(onCommandResult).toHaveBeenCalledWith(
    '/status',
    'Model: claude_code:haiku',
    true,
    undefined,
  );
  // The result is NOT surfaced as a success toast anymore.
  expect(push).not.toHaveBeenCalledWith('ok', expect.anything(), expect.anything());
});

test('a failed command echoes the error inline instead of an error toast', async () => {
  stubCommands([{ name: 'status', description: 'Show status', params: [] }]);
  vi.spyOn(api, 'post').mockRejectedValue(new Error('daemon unreachable'));
  const onCommandResult = vi.fn();
  const push = vi.spyOn(toasts, 'push');
  await render(ChatComposer, { ...base, onCommandResult });

  composerPrefill.request(base.sessionId, '/status', true);

  await vi.waitFor(() => expect(onCommandResult).toHaveBeenCalled());
  expect(onCommandResult).toHaveBeenCalledWith('/status', 'daemon unreachable', false);
  expect(push).not.toHaveBeenCalledWith('err', '/status failed', expect.anything());
});

test('the /job command carries an "open jobs" affordance in its echo', async () => {
  stubCommands([{ name: 'job', description: 'Spawn a job', params: [] }]);
  vi.spyOn(api, 'post').mockResolvedValue({ result: 'Job started' });
  const onCommandResult = vi.fn();
  await render(ChatComposer, { ...base, onCommandResult });

  composerPrefill.request(base.sessionId, '/job build the thing', true);

  await vi.waitFor(() => expect(onCommandResult).toHaveBeenCalled());
  expect(onCommandResult).toHaveBeenCalledWith('/job build the thing', 'Job started', true, {
    label: 'Open jobs',
    href: '#jobs',
  });
});

test('an unknown command echoes an inline error rather than a toast', async () => {
  stubCommands([{ name: 'status', description: 'Show status', params: [] }]);
  const post = vi.spyOn(api, 'post');
  const onCommandResult = vi.fn();
  await render(ChatComposer, { ...base, onCommandResult });

  composerPrefill.request(base.sessionId, '/bogus', true);

  await vi.waitFor(() => expect(onCommandResult).toHaveBeenCalled());
  expect(onCommandResult).toHaveBeenCalledWith(
    '/bogus',
    expect.stringContaining('Unknown command'),
    false,
  );
  expect(post).not.toHaveBeenCalled();
});

test('a non-run prefill request for this session fills and focuses the composer', async () => {
  stubCommands([]);
  const post = vi.spyOn(api, 'post');
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });

  composerPrefill.request(base.sessionId, '/model ', false);

  await expect.element(box).toHaveValue('/model ');
  await expect.element(box).toHaveFocus();
  expect(post).not.toHaveBeenCalled();
});

test('a prefill request for another session is ignored here', async () => {
  stubCommands([]);
  const post = vi.spyOn(api, 'post');
  await render(ChatComposer, { ...base, sessionId: 's1' });
  const box = page.getByRole('textbox', { name: 'Message' });

  // A non-run request (no command-list gate) so only the session guard can hold it.
  composerPrefill.request('other-session', '/model ', false);

  await expect.element(box).toHaveValue('');
  expect(post).not.toHaveBeenCalled();
});

test('a static-choices arg shows an inline options list and submits the pick', async () => {
  stubCommands([
    {
      name: 'sessions',
      description: 'List sessions',
      params: [{ name: 'status', choices: ['running', 'completed', 'failed'] }],
    },
  ]);
  const post = vi.spyOn(api, 'post').mockResolvedValue({ result: 'ok' });
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });

  // Entering the command's argument surfaces its choices inline.
  await userEvent.fill(box, '/sessions ');
  await expect.element(page.getByRole('option', { name: 'running' })).toBeInTheDocument();
  await expect.element(page.getByRole('option', { name: 'completed' })).toBeInTheDocument();

  // Arrow to the second choice and Enter to pick: it completes the command text and
  // submits it (the pick maps onto `status`). (Keyboard, since the menu floats above
  // the composer and sits off the top of an isolated test's viewport for a click.)
  await userEvent.keyboard('{ArrowDown}{Enter}');
  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toContain('/commands/sessions');
  expect(post.mock.calls[0]?.[1]).toMatchObject({ status: 'completed' });
  await expect.element(box).toHaveValue('');
});

test('typing narrows the inline choices to the partial argument', async () => {
  stubCommands([
    {
      name: 'sessions',
      description: 'List sessions',
      params: [{ name: 'status', choices: ['running', 'completed', 'failed'] }],
    },
  ]);
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });

  await userEvent.fill(box, '/sessions comp');
  await expect.element(page.getByRole('option', { name: 'completed' })).toBeInTheDocument();
  await expect.element(page.getByRole('option', { name: 'running' })).not.toBeInTheDocument();
});

test('a widget=effort arg fetches the model levels and offers them inline', async () => {
  stubCommands(
    [
      {
        name: 'effort',
        description: 'Set effort',
        params: [
          { name: 'user_id' },
          { name: 'message', widget: 'effort' },
          { name: 'session_id' },
        ],
      },
    ],
    (path) =>
      path.includes('/effort-levels')
        ? Promise.resolve({ model: 'gpt-5.4', supported_effort_levels: ['low', 'medium', 'high'] })
        : undefined,
  );
  const post = vi.spyOn(api, 'post').mockResolvedValue({ result: 'ok' });
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });

  await userEvent.fill(box, '/effort ');
  await expect.element(page.getByRole('option', { name: 'high' })).toBeInTheDocument();
  // Arrow to the third level (low, medium, high) and Enter to submit it.
  await userEvent.keyboard('{ArrowDown}{ArrowDown}{Enter}');

  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toContain('/commands/effort');
  expect(post.mock.calls[0]?.[1]).toMatchObject({ session_id: 's1', message: 'high' });
});

test('picking /model from the inline menu opens the model picker, not a text field', async () => {
  stubCommands([
    {
      name: 'model',
      description: 'Switch model',
      params: [{ name: 'user_id' }, { name: 'message', widget: 'model' }, { name: 'session_id' }],
    },
  ]);
  const req = vi.spyOn(modelPickerRequest, 'request');
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });

  await userEvent.fill(box, '/model');
  await expect.element(page.getByRole('option', { name: /\/model/ })).toBeInTheDocument();
  await userEvent.keyboard('{Enter}');

  expect(req).toHaveBeenCalledWith('s1');
  // The command text is not left sitting in the composer as a plain field.
  await expect.element(box).toHaveValue('');
});

test('picking location from the context menu chips it and sends it as metadata, not text', async () => {
  stubCommands([]);
  mockGeoSuccess({ latitude: 37.7749, longitude: -122.4194, accuracy: 20 });
  const onSend = vi.fn();
  await render(ChatComposer, { ...base, onSend });

  await pickLocation();
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('location')))
    .toBeInTheDocument();

  await userEvent.fill(page.getByRole('textbox', { name: 'Message' }), 'where am i');
  await userEvent.keyboard('{Enter}');

  await vi.waitFor(() => expect(onSend).toHaveBeenCalled());
  // The message text stays clean - the location rides as structured metadata.
  expect(onSend.mock.calls[0]?.[0]).toBe('where am i');
  expect(onSend.mock.calls[0]?.[1]?.contextMetadata).toEqual([
    { key: 'location', label: 'Location', value: '37.77490, -122.41940 (±20m)' },
  ]);
  // The chip clears after a successful send.
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('location')))
    .not.toBeInTheDocument();
});

test('a denied capture toasts and attaches no chip', async () => {
  stubCommands([]);
  mockGeoError(1);
  const push = vi.spyOn(toasts, 'push');
  await render(ChatComposer, { ...base });

  await pickLocation();

  await vi.waitFor(() => expect(push).toHaveBeenCalled());
  expect(push.mock.calls[0]?.[0]).toBe('warn');
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('location')))
    .not.toBeInTheDocument();
});

test('with auto-attach on, a send carries the location as metadata without the menu', async () => {
  stubCommands([]);
  mockGeoSuccess({ latitude: 51.5074, longitude: -0.1278, accuracy: 12 });
  autoAttachStore(GEO_KEY).set(true);
  const onSend = vi.fn();
  await render(ChatComposer, { ...base, onSend });

  await userEvent.fill(page.getByRole('textbox', { name: 'Message' }), 'ping');
  await userEvent.keyboard('{Enter}');

  await vi.waitFor(() => expect(onSend).toHaveBeenCalled());
  expect(onSend.mock.calls[0]?.[0]).toBe('ping');
  expect(onSend.mock.calls[0]?.[1]?.contextMetadata).toEqual([
    { key: 'location', label: 'Location', value: '51.50740, -0.12780 (±12m)' },
  ]);
});

test('with auto-attach off, a send carries no context_metadata', async () => {
  stubCommands([]);
  mockGeoSuccess({ latitude: 51.5074, longitude: -0.1278, accuracy: 12 });
  const onSend = vi.fn();
  await render(ChatComposer, { ...base, onSend });

  await userEvent.fill(page.getByRole('textbox', { name: 'Message' }), 'ping');
  await userEvent.keyboard('{Enter}');

  await vi.waitFor(() => expect(onSend).toHaveBeenCalled());
  expect(onSend.mock.calls[0]?.[0]).toBe('ping');
  expect(onSend.mock.calls[0]?.[1]?.contextMetadata).toBeUndefined();
});

test('a daemon-provided server provider shows in the menu, alongside the client one', async () => {
  stubServerMenu([{ key: 'webpage', label: 'Web page', icon: 'link', has_choices: false }]);
  await render(ChatComposer, { ...base });

  await openContextMenu();
  await expect
    .element(page.getByTestId(TESTID.composerContextOption('webpage')))
    .toBeInTheDocument();
  await expect
    .element(page.getByTestId(TESTID.composerContextOption('location')))
    .toBeInTheDocument();
});

test('picking a no-choices server provider captures on the daemon and chips the result', async () => {
  stubServerMenu([{ key: 'webpage', label: 'Web page', icon: 'link', has_choices: false }]);
  const post = vi.spyOn(api, 'post').mockResolvedValue({
    items: [{ key: 'webpage', label: 'Web page', value: 'Example Domain — a page' }],
  });
  await render(ChatComposer, { ...base });

  await openContextMenu();
  await expect
    .element(page.getByTestId(TESTID.composerContextOption('webpage')))
    .toBeInTheDocument();
  clickContextOption('webpage');

  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toBe('/api/context-providers/webpage/capture');
  expect(post.mock.calls[0]?.[1]).toEqual({ session_id: 's1', arg: null });
  await expect.element(page.getByTestId(TESTID.composerContextChip('webpage'))).toBeInTheDocument();
});

test('a choices server provider opens a submenu, then captures the chosen value', async () => {
  stubServerMenu([{ key: 'terminal', label: 'Terminal output', icon: 'term', has_choices: true }], {
    terminal: [
      { value: 't1', label: 'npm test' },
      { value: 't2', label: 'server log' },
    ],
  });
  const post = vi.spyOn(api, 'post').mockResolvedValue({
    items: [{ key: 'terminal:t2', label: 'server log', value: '... tail of output ...' }],
  });
  await render(ChatComposer, { ...base });

  await openContextMenu();
  await expect
    .element(page.getByTestId(TESTID.composerContextOption('terminal')))
    .toBeInTheDocument();
  clickContextOption('terminal');

  // The submenu of the session's terminals opens; capture waits for a pick.
  await expect.element(page.getByTestId(TESTID.composerContextSubmenu)).toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
  (
    page.getByTestId(TESTID.composerContextChoice('terminal', 't2')).element() as HTMLElement
  ).click();

  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toBe('/api/context-providers/terminal/capture');
  expect(post.mock.calls[0]?.[1]).toEqual({ session_id: 's1', arg: 't2' });
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('terminal:t2')))
    .toBeInTheDocument();
});

test('a picker provider opens the Picker overlay and a pick attaches a chip', async () => {
  stubServerMenu(
    [{ key: 'file', label: 'Workspace file', icon: 'file', has_choices: true, picker: true }],
    {
      file: [
        { value: 'notes.md', label: 'notes.md' },
        { value: 'kb/ops/sse.md', label: 'kb/ops/sse.md' },
      ],
    },
  );
  const post = vi.spyOn(api, 'post').mockResolvedValue({
    items: [{ key: 'file:kb/ops/sse.md', label: 'kb/ops/sse.md', value: '# sse notes' }],
  });
  await render(ChatComposer, { ...base });

  await openContextMenu();
  clickContextOption('file');

  // A picker provider opens the searchable Picker, not the inline submenu, and
  // captures nothing until a row is chosen.
  await expect.element(page.getByTestId(TESTID.picker)).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.composerContextSubmenu)).not.toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();

  (page.getByTestId(TESTID.pickerOption('kb/ops/sse.md')).element() as HTMLElement).click();

  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toBe('/api/context-providers/file/capture');
  expect(post.mock.calls[0]?.[1]).toEqual({ session_id: 's1', arg: 'kb/ops/sse.md' });
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('file:kb/ops/sse.md')))
    .toBeInTheDocument();
  // The Picker closes once a pick is made.
  await expect.element(page.getByTestId(TESTID.picker)).not.toBeInTheDocument();
});

test('picking a workspace file from the @ popover attaches a file: chip and strips the @token', async () => {
  const get = vi.spyOn(api, 'get').mockImplementation((path: string) => {
    if (path.includes('/workspace'))
      return Promise.resolve({
        entries: [
          { path: 'notes.md', name: 'notes.md', is_dir: false },
          { path: 'kb/ops/sse.md', name: 'sse.md', is_dir: false },
        ],
        workspace_dir: '/ws',
      });
    // /api/commands, /api/context-providers, etc: nothing this test needs.
    return Promise.resolve({});
  });
  const post = vi.spyOn(api, 'post').mockResolvedValue({
    items: [{ key: 'file:kb/ops/sse.md', label: 'kb/ops/sse.md', value: '# sse notes' }],
  });
  await render(ChatComposer, { ...base });
  // Let the one-shot workspace load settle so the @ popover has files to list.
  await vi.waitFor(() => expect(get).toHaveBeenCalledWith(expect.stringContaining('/workspace')));

  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'see @sse');
  // 'sse' matches only the nested file; the popover lists it by its full path.
  await expect.element(page.getByRole('option', { name: /kb\/ops\/sse\.md/ })).toBeInTheDocument();
  await userEvent.keyboard('{Enter}');

  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toBe('/api/context-providers/file/capture');
  expect(post.mock.calls[0]?.[1]).toEqual({ session_id: 's1', arg: 'kb/ops/sse.md' });
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('file:kb/ops/sse.md')))
    .toBeInTheDocument();
  // Convergence: the ref became a chip, so the @query trigger is stripped and no
  // @path text is left inline.
  await expect.element(box).toHaveValue('see ');
});

test('an @ file whose capture returns no items strips the token but attaches no chip, silently', async () => {
  const get = vi.spyOn(api, 'get').mockImplementation((path: string) => {
    if (path.includes('/workspace'))
      return Promise.resolve({
        entries: [{ path: 'image.bin', name: 'image.bin', is_dir: false }],
        workspace_dir: '/ws',
      });
    return Promise.resolve({});
  });
  // A binary/oversized/out-of-tree file captures as 200 with an empty items array
  // (not an error): no chip, and no error toast.
  const post = vi.spyOn(api, 'post').mockResolvedValue({ items: [] });
  const push = vi.spyOn(toasts, 'push');
  await render(ChatComposer, { ...base });
  await vi.waitFor(() => expect(get).toHaveBeenCalledWith(expect.stringContaining('/workspace')));

  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'see @image');
  await expect.element(page.getByRole('option', { name: /image\.bin/ })).toBeInTheDocument();
  await userEvent.keyboard('{Enter}');

  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('file:image.bin')))
    .not.toBeInTheDocument();
  expect(push).not.toHaveBeenCalled();
  // The trigger text is still cleaned up even though nothing attached.
  await expect.element(box).toHaveValue('see ');
});

test('a server capture error toasts and attaches no chip', async () => {
  stubServerMenu([{ key: 'webpage', label: 'Web page', icon: 'link', has_choices: false }]);
  vi.spyOn(api, 'post').mockRejectedValue(new Error('fetch failed'));
  const push = vi.spyOn(toasts, 'push');
  await render(ChatComposer, { ...base });

  await openContextMenu();
  await expect
    .element(page.getByTestId(TESTID.composerContextOption('webpage')))
    .toBeInTheDocument();
  clickContextOption('webpage');

  await vi.waitFor(() => expect(push).toHaveBeenCalled());
  expect(push.mock.calls[0]?.[0]).toBe('err');
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('webpage')))
    .not.toBeInTheDocument();
});

test('a contextAttach request for this session chips the captured items', async () => {
  stubCommands([]);
  await render(ChatComposer, { ...base });

  contextAttach.request(base.sessionId, [
    { key: 'session:s9', label: 'a referenced session', value: 'status: active' },
  ]);

  await expect
    .element(page.getByTestId(TESTID.composerContextChip('session:s9')))
    .toBeInTheDocument();
});

test('a contextAttach request for another session is ignored here', async () => {
  stubCommands([]);
  await render(ChatComposer, { ...base, sessionId: 's1' });

  contextAttach.request('other-session', [{ key: 'session:x', label: 'x', value: 'v' }]);

  await expect
    .element(page.getByTestId(TESTID.composerContextChip('session:x')))
    .not.toBeInTheDocument();
  contextAttach.consume('other-session');
});

test('a reference-marker paste captures the record, chips it, and preventDefaults', async () => {
  stubCommands([]);
  const post = vi.spyOn(api, 'post').mockResolvedValue({
    items: [{ key: 'session:s9', label: 'a referenced session', value: 'status: active' }],
  });
  await render(ChatComposer, { ...base });
  const el = page.getByRole('textbox', { name: 'Message' }).element() as HTMLTextAreaElement;

  const dt = new DataTransfer();
  dt.setData('text/html', '<span data-tsugite-ref="session:s9">session s9</span>');
  dt.setData('text/plain', 'session s9');
  const cancelled = !el.dispatchEvent(pasteEvent(dt));

  expect(cancelled).toBe(true);
  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toBe('/api/context-providers/session/capture');
  expect(post.mock.calls[0]?.[1]).toEqual({ session_id: 's1', arg: 's9' });
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('session:s9')))
    .toBeInTheDocument();
});

test('a paste whose html is not a reference marker keeps native behavior', async () => {
  stubCommands([]);
  const post = vi.spyOn(api, 'post');
  await render(ChatComposer, { ...base });
  const el = page.getByRole('textbox', { name: 'Message' }).element() as HTMLTextAreaElement;

  const dt = new DataTransfer();
  dt.setData('text/html', '<p>pasted rich text</p>');
  dt.setData('text/plain', 'pasted rich text');
  const cancelled = !el.dispatchEvent(pasteEvent(dt));

  // No marker -> the composer never preventDefaults or captures; native paste stands.
  expect(cancelled).toBe(false);
  expect(post).not.toHaveBeenCalled();
});

test('recent chats show in the @ popover, excluding the chat you are in', async () => {
  stubCommands([]);
  // base.sessionId is 's1' - it must never list itself.
  sessions.rows = [
    sessionRow({ id: 's1', title: 'selfchat' }),
    sessionRow({ id: 's2', title: 'backuprun' }),
  ];
  await render(ChatComposer, { ...base });
  const box = page.getByRole('textbox', { name: 'Message' });

  await userEvent.fill(box, 'ref @backuprun');
  await expect.element(page.getByRole('option', { name: /backuprun/ })).toBeInTheDocument();

  // The current chat's own title never surfaces in its composer's @ source.
  await userEvent.fill(box, 'ref @selfchat');
  expect(page.getByRole('option', { name: /selfchat/ }).elements()).toHaveLength(0);
});

test('an @<prefix> query hits the provider search route and a pick captures via that provider', async () => {
  const get = vi.spyOn(api, 'get').mockImplementation((path: string) => {
    if (path === '/api/context-providers')
      return Promise.resolve({
        providers: [
          {
            key: 'demo',
            label: 'Demo docs',
            icon: 'sparkle',
            has_choices: false,
            in_menu: false,
            autocomplete_prefix: 'demo',
          },
        ],
      });
    if (path.includes('/search'))
      return Promise.resolve({ results: [{ value: 'runbook', label: 'runbook' }] });
    // /api/commands, /workspace, image config: nothing this test needs.
    return Promise.resolve({});
  });
  const post = vi.spyOn(api, 'post').mockResolvedValue({
    items: [{ key: 'demo:runbook', label: 'demo/runbook', value: 'restart the daemon' }],
  });
  await render(ChatComposer, { ...base });
  // Let the provider list load so the prefix registry is built.
  await vi.waitFor(() => expect(get).toHaveBeenCalledWith('/api/context-providers'));

  const box = page.getByRole('textbox', { name: 'Message' });
  await userEvent.fill(box, 'see @demo run');
  await vi.waitFor(() =>
    expect(get).toHaveBeenCalledWith(expect.stringContaining('/context-providers/demo/search?')),
  );
  await expect.element(page.getByRole('option', { name: /runbook/ })).toBeInTheDocument();
  await userEvent.keyboard('{Enter}');

  // The pick captures through the demo provider with the picked value as arg.
  await vi.waitFor(() => expect(post).toHaveBeenCalled());
  expect(post.mock.calls[0]?.[0]).toBe('/api/context-providers/demo/capture');
  expect(post.mock.calls[0]?.[1]).toEqual({ session_id: 's1', arg: 'runbook' });
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('demo:runbook')))
    .toBeInTheDocument();
  // An autocomplete-only source (in_menu:false) stays out of the add-context menu.
  await openContextMenu();
  await expect
    .element(page.getByTestId(TESTID.composerContextOption('demo')))
    .not.toBeInTheDocument();
});
