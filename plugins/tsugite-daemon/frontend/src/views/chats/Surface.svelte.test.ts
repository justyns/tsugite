/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import { tick } from 'svelte';
import Surface from './Surface.svelte';
import { sessions } from '$lib/stores/sessions.svelte';
import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
import { api } from '$lib/api/client';
import { routeHistory } from '$lib/router.svelte';

const realGetInfo = sessions.getInfo.bind(sessions);

function stubInfo(agent: string | null, metadata: Record<string, unknown>) {
  sessions.getInfo = async () => ({ agent, metadata, contextLimit: null, cumulativeTokens: null });
}

function roster(...names: string[]) {
  agentsMeta.runtime = names.length
    ? {
        agent_file: names[0] as string,
        workspace_dir: '/ws',
        model: null,
        context_limit: null,
        running_tasks: 0,
      }
    : null;
}

/** Resolve /api/health (the image-config source) fast so uploadChosen reaches
 *  the upload call; everything else keeps its real (test-proxy) behavior. */
function stubHealthGet() {
  const realGet = api.get.bind(api);
  vi.spyOn(api, 'get').mockImplementation((path: string, ...rest: unknown[]) =>
    path.startsWith('/api/health')
      ? Promise.resolve({ images: { max_edge: 1568, quality: 0.85 } } as never)
      : (realGet as (p: string, ...r: unknown[]) => Promise<never>)(path, ...rest),
  );
}

function dragEvent(type: 'dragover' | 'drop', dt: DataTransfer): DragEvent {
  const e = new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true });
  if (e.dataTransfer !== dt)
    Object.defineProperty(e, 'dataTransfer', { value: dt, configurable: true });
  return e;
}

const surfaceEl = () => document.querySelector('.chat-surface') as HTMLElement;
const composerTa = () =>
  document.querySelector('.chat-surface textarea') as HTMLTextAreaElement | null;

afterEach(async () => {
  sessions.getInfo = realGetInfo;
  vi.restoreAllMocks();
  await page.viewport(1440, 900);
});

test('a job worker/verifier session (metadata.job_id) renders read-only, not a composer', async () => {
  // The session's true agent (smoke) IS a chat adapter, but a spawned job
  // artifact is inspect-only - chatting into it is a footgun; the parent chat is
  // where the conversation continues.
  roster('smoke');
  stubInfo('smoke', { job_id: 'job-a08493b1' });
  render(Surface, { params: { sessionId: 'session-worker' } });
  await expect.element(page.getByTestId('chat-readonly')).toBeInTheDocument();
  expect(document.querySelector('[data-testid="chat-composer"]')).toBeFalsy();
});

test('a normal (non-artifact) session renders the composer', async () => {
  roster('smoke');
  stubInfo('smoke', { job_host: true });
  render(Surface, { params: { sessionId: 'session-parent' } });
  await expect.element(page.getByTestId('chat-composer')).toBeInTheDocument();
  expect(document.querySelector('[data-testid="chat-readonly"]')).toBeFalsy();
});

test('dropping OS files on the chat surface attaches them (multiple files)', async () => {
  roster('smoke');
  stubInfo('smoke', { job_host: true });
  stubHealthGet();
  const upload = vi
    .spyOn(api, 'uploadFiles')
    .mockResolvedValue({ files: [{ name: 'a.txt' }, { name: 'b.txt' }] });
  render(Surface, { params: { sessionId: 'session-parent' } });
  await expect.element(page.getByTestId('chat-composer')).toBeInTheDocument();

  const dt = new DataTransfer();
  dt.items.add(new File(['a'], 'a.txt', { type: 'text/plain' }));
  dt.items.add(new File(['b'], 'b.txt', { type: 'text/plain' }));
  surfaceEl().dispatchEvent(dragEvent('drop', dt));

  await vi.waitFor(() => expect(upload).toHaveBeenCalled());
  const sent = Array.from((upload.mock.calls[0]?.[1] ?? []) as Iterable<File>);
  expect(sent).toHaveLength(2);
});

test('a file dragover shows the drop overlay', async () => {
  roster('smoke');
  stubInfo('smoke', { job_host: true });
  render(Surface, { params: { sessionId: 'session-parent' } });
  await expect.element(page.getByTestId('chat-composer')).toBeInTheDocument();

  const dt = new DataTransfer();
  dt.items.add(new File(['a'], 'a.txt', { type: 'text/plain' }));
  surfaceEl().dispatchEvent(dragEvent('dragover', dt));

  await vi.waitFor(() => expect(document.querySelector('.chat-drop')).toBeTruthy());
});

test('an internal surface drag (no Files type) is ignored - no overlay, no attach', async () => {
  roster('smoke');
  stubInfo('smoke', { job_host: true });
  const upload = vi.spyOn(api, 'uploadFiles').mockResolvedValue({ files: [] });
  render(Surface, { params: { sessionId: 'session-parent' } });
  await expect.element(page.getByTestId('chat-composer')).toBeInTheDocument();

  const dt = new DataTransfer();
  dt.setData('application/x-tsugite-surface', JSON.stringify({ kind: 'chat', params: {} }));
  surfaceEl().dispatchEvent(dragEvent('dragover', dt));
  await tick();
  expect(document.querySelector('.chat-drop')).toBeFalsy();

  surfaceEl().dispatchEvent(dragEvent('drop', dt));
  await tick();
  expect(upload).not.toHaveBeenCalled();
});

test('a read-only (job artifact) surface ignores file drops', async () => {
  roster('smoke');
  stubInfo('smoke', { job_id: 'job-x' });
  const upload = vi.spyOn(api, 'uploadFiles').mockResolvedValue({ files: [] });
  render(Surface, { params: { sessionId: 'session-worker' } });
  await expect.element(page.getByTestId('chat-readonly')).toBeInTheDocument();

  const dt = new DataTransfer();
  dt.items.add(new File(['a'], 'a.txt', { type: 'text/plain' }));
  surfaceEl().dispatchEvent(dragEvent('dragover', dt));
  await tick();
  expect(document.querySelector('.chat-drop')).toBeFalsy();

  surfaceEl().dispatchEvent(dragEvent('drop', dt));
  await tick();
  expect(upload).not.toHaveBeenCalled();
});

test('a fresh session paints a context meter from its resolved limit (0 / agent default)', async () => {
  // The meter's fallback is built from getInfo().contextLimit. A brand-new
  // session's raw limit is null, but getInfo now returns the RESOLVED limit
  // (agent default), so the header must show a `0 / <default>` meter from open
  // rather than no context readout until the first turn.
  roster('smoke');
  sessions.getInfo = async () => ({
    metadata: { job_host: true },
    contextLimit: 200_000,
    cumulativeTokens: 0,
  });
  render(Surface, { params: { sessionId: 'session-fresh' } });
  await expect.element(page.getByText(/0\/200k/)).toBeInTheDocument();
});

test('navigating to a composable chat auto-focuses the composer (desktop)', async () => {
  // Landing on a session should let the user type immediately - focus the
  // composer textarea once the surface knows it's editable.
  await page.viewport(1440, 900);
  roster('smoke');
  stubInfo('smoke', { job_host: true });
  render(Surface, { params: { sessionId: 'session-parent' } });
  await expect.element(page.getByTestId('chat-composer')).toBeInTheDocument();
  await vi.waitFor(() => expect(document.activeElement).toBe(composerTa()));
});

test('switching the selected session refocuses the composer (desktop)', async () => {
  // Changing the selected session is a navigation: refocus even if the user had
  // clicked away to read. (A stream frame or resync would NOT change the id, so
  // it can never steal focus - that is what keying on the session id buys.)
  await page.viewport(1440, 900);
  roster('smoke');
  stubInfo('smoke', { job_host: true });
  const { rerender } = await render(Surface, { params: { sessionId: 'session-a' } });
  await vi.waitFor(() => expect(document.activeElement).toBe(composerTa()));
  const ta = composerTa()!;
  ta.blur();
  expect(document.activeElement).not.toBe(ta);
  await rerender({ params: { sessionId: 'session-b' } });
  await vi.waitFor(() => expect(document.activeElement).toBe(composerTa()));
});

test('auto-focus is suppressed at phone width (the keyboard would cover the conversation)', async () => {
  await page.viewport(390, 780);
  roster('smoke');
  let infoLoaded: Promise<unknown> = Promise.resolve();
  sessions.getInfo = () => {
    infoLoaded = Promise.resolve({
      metadata: { job_host: true },
      contextLimit: null,
      cumulativeTokens: null,
    });
    return infoLoaded as ReturnType<typeof realGetInfo>;
  };
  const focusSpy = vi.spyOn(HTMLTextAreaElement.prototype, 'focus');
  render(Surface, { params: { sessionId: 'session-parent' } });
  await expect.element(page.getByTestId('chat-composer')).toBeInTheDocument();
  await infoLoaded; // getInfo settled -> sessionInfo applied -> the (skipped) focus effect ran
  await tick();
  expect(focusSpy).not.toHaveBeenCalled();
});

test('auto-focus never fires for a read-only (job artifact) surface', async () => {
  await page.viewport(1440, 900);
  roster('smoke');
  stubInfo('smoke', { job_id: 'job-x' });
  const focusSpy = vi.spyOn(HTMLTextAreaElement.prototype, 'focus');
  render(Surface, { params: { sessionId: 'session-worker' } });
  await expect.element(page.getByTestId('chat-readonly')).toBeInTheDocument();
  expect(composerTa()).toBeFalsy();
  expect(focusSpy).not.toHaveBeenCalled();
});

test('the phone back affordance clears the sessionId to the list hash', async () => {
  // At phone width the header back control returns to the sessions list by clearing
  // the hash's sessionId. With no list entry behind us (prev = null) it pushes the
  // bare #chats hash rather than escaping the app.
  await page.viewport(390, 780);
  routeHistory.prev = null;
  location.hash = '#chats?sessionId=session-parent';
  roster('smoke');
  stubInfo('smoke', { job_host: true });
  render(Surface, { params: { sessionId: 'session-parent' } });
  await page.getByTestId('phone-back').click();
  expect(location.hash).toBe('#chats');
});
