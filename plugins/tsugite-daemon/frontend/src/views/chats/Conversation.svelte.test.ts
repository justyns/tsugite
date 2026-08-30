/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import Conversation from './Conversation.svelte';
import { ConversationController } from './conversation.svelte';
import type { SessionRow } from '$lib/stores/sessions.svelte';
import { TESTID } from '$lib/testids';
import { sessionRow } from './__fixtures__/sessionRow';
import { expandThinking } from '$lib/stores/expandThinking.svelte';

const noop = () => {};
const callbacks = {
  onToggleRail: noop,
  onBack: noop,
  onRenameCommit: noop,
  onAliasCommit: noop,
  onTopicCommit: noop,
  onComplete: noop,
  onCancel: noop,
  onRestart: noop,
  onPin: noop,
  onUnpin: noop,
  onSetPrimary: noop,
  onCopyId: noop,
  onOpenSession: noop,
  onRetry: noop,
  onDismissAttention: noop,
};

// Reset to a desktop width so a narrow-viewport test never leaks into the next
// file (the browser instance is shared and viewport persists).
afterEach(async () => {
  await page.viewport(1440, 900);
});

// The thinking pref is a module singleton shared with every other browser test.
afterEach(() => expandThinking.set(true));

function controllerWith(events: Record<string, unknown>[]): ConversationController {
  const ctrl = new ConversationController();
  ctrl.sessionId = 'sess-1';
  ctrl.events = events;
  return ctrl;
}

// --- Auto-follow (pin-to-bottom) scroll helpers -----------------------------
// A mid-stream conversation: a sent prompt with the AI turn open, so appended
// stream_chunk frames grow the transcript the way a live token stream does.
function streamingController(): ConversationController {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'list 40 things', timestamp: '2026-07-18T10:00:00Z' },
    { type: 'turn_start' },
  ]);
  ctrl.streaming = true;
  return ctrl;
}

// Multi-paragraph filler tall enough to overflow the (force-shrunk) scroll box.
function tall(n: number): string {
  return Array.from({ length: n }, (_, i) => `paragraph ${i} lorem ipsum dolor sit amet`).join(
    '\n\n',
  );
}

// The middle pane renders with no bounded ancestor in a component test, so pin
// the scroll container to a short fixed height - otherwise it grows to content
// and never overflows, making every scrollTop assertion vacuous.
function scrollBox(): HTMLElement {
  const el = document.querySelector('.convo-scroll') as HTMLElement;
  el.style.flex = 'none';
  el.style.height = '150px';
  return el;
}

function distFromBottom(el: HTMLElement): number {
  return el.scrollHeight - el.scrollTop - el.clientHeight;
}

function appendChunk(ctrl: ConversationController, text: string): void {
  ctrl.events = [...ctrl.events, { type: 'stream_chunk', chunk: text }];
}

function wheelUp(el: HTMLElement): void {
  el.dispatchEvent(new WheelEvent('wheel', { deltaY: -160, bubbles: true }));
}

test('renders a replayed user + ai turn from the controller timeline', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'add exponential backoff', timestamp: '2026-07-14T15:00:00Z' },
    { type: 'model_response', raw_content: 'Here is the plan.', thought: 'Here is the plan.' },
    { type: 'final_result', result: 'Here is the plan.', tokens: 40 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('add exponential backoff')).toBeInTheDocument();
  await expect.element(page.getByText('Here is the plan.')).toBeInTheDocument();
  // Author gutters distinguish the two roles.
  await expect.element(page.getByText('you')).toBeInTheDocument();
  await expect.element(page.getByText('tsugite')).toBeInTheDocument();
});

test('a client_context injection renders in the gutter as label:value rows, not the bubble', async () => {
  const ctrl = controllerWith([
    {
      type: 'user_input',
      text: 'where am i',
      display_text: 'where am i',
      injected: [
        {
          tag: 'client_context',
          items: [{ key: 'location', label: 'Location', value: '37.77490, -122.41940 (±20m)' }],
        },
      ],
      timestamp: '2026-07-18T10:00:00Z',
    },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });

  const row = page.getByTestId(TESTID.chatContextRow('location'));
  await expect.element(row).toHaveTextContent('Location');
  await expect.element(row).toHaveTextContent('37.77490, -122.41940 (±20m)');
  // The typed words still render as the user's message.
  await expect.element(page.getByText('where am i')).toBeInTheDocument();
});

test('a single-step ai turn footer surfaces cached tokens (reads headline, rd/wr in the tooltip)', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'compute totals', timestamp: '2026-07-14T15:00:00Z' },
    {
      type: 'model_response',
      raw_content: 'Done.',
      thought: 'Done.',
      usage: { cache_read_input_tokens: 8100, cache_creation_input_tokens: 1200 },
    },
    { type: 'final_result', result: 'Done.', tokens: 40 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const footer = page.getByText('8.1k cached');
  await expect.element(footer).toBeInTheDocument();
  await expect
    .element(footer)
    .toHaveAttribute('title', '8.1k cache reads / 1.2k writes across 1 step');
});

test('a multi-step ai turn footer headlines the LAST step read, summed totals in the tooltip', async () => {
  // Two steps each re-read the ~60k cached prefix. The headline must show the LAST
  // step's read (60k - the current cached prefix, matching the context meter's
  // scale), NOT the 120k cross-step sum; the sum + step count live in the tooltip
  // so "120k" can never be mistaken for the current context next to a 60k meter.
  const ctrl = controllerWith([
    { type: 'user_input', text: 'long task', timestamp: '2026-07-14T15:00:00Z' },
    {
      type: 'model_response',
      raw_content: 's1',
      thought: 's1',
      usage: { cache_read_input_tokens: 60000, cache_creation_input_tokens: 8000 },
    },
    { type: 'code_execution', code: 'x=1' },
    {
      type: 'model_response',
      raw_content: 's2',
      thought: 's2',
      usage: { cache_read_input_tokens: 60000, cache_creation_input_tokens: 4000 },
    },
    { type: 'final_result', result: 'done', tokens: 40 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const footer = page.getByText('60k cached');
  await expect.element(footer).toBeInTheDocument();
  await expect
    .element(footer)
    .toHaveAttribute('title', '120k cache reads / 12k writes across 2 steps');
});

test('an ai turn with no reported cache shows no cached footer (honest absence)', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'hi', timestamp: '2026-07-14T15:00:00Z' },
    { type: 'model_response', raw_content: 'hey', thought: 'hey' },
    { type: 'final_result', result: 'hey', tokens: 10 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('hey')).toBeInTheDocument();
  expect(page.getByText(/cached/).query()).toBeNull();
});

test('an ai turn footer shows the serving model, fully qualified on hover', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'hi', timestamp: '2026-07-14T15:00:00Z' },
    {
      type: 'model_response',
      raw_content: 'hey',
      thought: 'hey',
      provider: 'openai',
      model: 'gpt-4o-mini',
    },
    { type: 'final_result', result: 'hey', tokens: 10 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });

  const badge = page.getByTestId(TESTID.chatTurnModel);
  await expect.element(badge).toHaveTextContent('gpt-4o-mini');
  await expect.element(badge).toHaveAttribute('title', 'openai:gpt-4o-mini');
});

test('a transcript whose model changes mid-session labels each turn with its own', async () => {
  // The second turn records no provider, so it also covers the tooltip falling
  // back to the bare id rather than a half-qualified one.
  const ctrl = controllerWith([
    { type: 'user_input', text: 'first', timestamp: '2026-07-14T15:00:00Z' },
    {
      type: 'model_response',
      raw_content: 'a',
      thought: 'a',
      provider: 'openai',
      model: 'gpt-4o-mini',
    },
    { type: 'final_result', result: 'a', tokens: 10 },
    { type: 'user_input', text: 'second', timestamp: '2026-07-14T15:05:00Z' },
    { type: 'model_response', raw_content: 'b', thought: 'b', model: 'claude-opus-4-5' },
    { type: 'final_result', result: 'b', tokens: 10 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });

  const badges = document.querySelectorAll(`[data-testid="${TESTID.chatTurnModel}"]`);
  expect(Array.from(badges, (b) => b.textContent?.trim())).toEqual([
    'gpt-4o-mini',
    'claude-opus-4-5',
  ]);
  expect(Array.from(badges, (b) => b.getAttribute('title'))).toEqual([
    'openai:gpt-4o-mini',
    'claude-opus-4-5',
  ]);
});

test('an ai turn from history with no recorded model shows no model footer', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'hi', timestamp: '2026-07-14T15:00:00Z' },
    { type: 'model_response', raw_content: 'hey', thought: 'hey' },
    { type: 'final_result', result: 'hey', tokens: 10 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });

  await expect.element(page.getByText('hey')).toBeInTheDocument();
  expect(page.getByTestId(TESTID.chatTurnModel).query()).toBeNull();
});

test('an unfinished ai turn is a live region (aria-busy) for streaming a11y', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: '2026-07-14T15:00:00Z' },
    { type: 'thought', content: 'streaming…' }, // no final_result -> still streaming
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  expect(document.querySelector('[aria-busy="true"]')).toBeTruthy();
});

test('a pending ask_user renders the Ask prompt wired to respond', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'ship it', timestamp: 'z' }]);
  ctrl.ask = {
    question: 'Push to origin?',
    questionType: 'yes_no',
    options: [],
    answered: false,
    answer: '',
  };
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('Push to origin?')).toBeInTheDocument();
});

test('a pending ask brought in by resync is scrolled into view (not left below the fold)', async () => {
  // A tall thread the reader has scrolled up through (unpinned): when resync sets
  // a pending ask, the blocking prompt renders at the tail and must be re-pinned
  // into view rather than left below the fold (the pasted-link approval report).
  const ctrl = controllerWith([
    { type: 'user_input', text: 'summarize this', timestamp: '2026-07-18T10:00:00Z' },
    { type: 'model_response', raw_content: tall(40), thought: tall(40) },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6); // opens at the tail
  wheelUp(el);
  el.scrollTop = 0; // reading history, unpinned - the prompt would be below the fold
  await expect.element(page.getByText('jump to live')).toBeInTheDocument();
  // The resync loader sets a pending ask (a durable ask_user with no answer).
  ctrl.ask = {
    askId: 'ask-1',
    question: 'Fetch content from example.com?',
    questionType: 'approval',
    options: ['Approve', 'Deny'],
    answered: false,
    answer: '',
  };
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6); // re-pinned to the prompt
  await expect.element(page.getByText('Fetch content from example.com?')).toBeInTheDocument();
});

test('a replayed native tool call renders its name, args, and result (no stuck spinner)', async () => {
  // Captured from GET /api/sessions/{id}/events: a native tool completes via
  // tool_result_audit, which must resolve the exec and surface args + result.
  const ctrl = controllerWith([
    { type: 'user_input', text: 'set the topic', timestamp: '2026-07-14T15:00:00Z' },
    {
      type: 'tool_call',
      tool: 'session_metadata',
      arguments: { key: 'topic', value: 'ledger totals' },
      step: 0,
    },
    {
      type: 'tool_result_audit',
      tool: 'session_metadata',
      success: true,
      duration_ms: 12,
      summary: 'metadata updated',
      step: 0,
    },
    { type: 'final_result', result: 'Done.' },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const header = page.getByRole('button', { name: /session_metadata/ });
  await expect.element(header).toBeInTheDocument();
  await header.click();
  await expect.element(page.getByText('ledger totals')).toBeInTheDocument();
  await expect.element(page.getByText('metadata updated')).toBeInTheDocument();
  // Defect 1: the completed turn leaves nothing aria-busy.
  expect(document.querySelector('[aria-busy="true"]')).toBeFalsy();
});

test('a failed last turn shows a prominent Retry that re-sends the last user message', async () => {
  // The real failed-turn shape (from the :18461 repro): a user_input, the error as
  // model content, session_end, then the error frame. The reducer folds this into
  // one AI turn carrying a kind:'error' block, which drives the prominent Retry.
  const onRetry = vi.fn();
  const ERR =
    "There's an issue with the selected model (bogus-model-xyz-999). Run --model to pick another. (subtype=success)";
  const ctrl = controllerWith([
    { type: 'user_input', text: 'hello', timestamp: '2026-07-21T06:00:00Z' },
    { type: 'model_response', raw_content: `[Error: ${ERR}]`, thought: `[Error: ${ERR}]` },
    { type: 'session_end', status: 'success' },
    { type: 'error', error: ERR },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks, onRetry });

  const retry = page.getByTestId('chat-retry');
  await expect.element(retry).toBeVisible(); // not hover-gated
  await retry.click();
  // Re-sends the last user message via the existing onRetry path.
  expect(onRetry).toHaveBeenCalledWith('hello');
});

test('a cancelled last turn shows the prominent Retry too (no error block, keyed off row status)', async () => {
  // Hitting Stop cancels the turn: it carries no kind:'error' block, so the
  // prominent Retry leans on the session's failed/cancelled status instead. The
  // last AI turn (a partial reply that got cut off) becomes the retry target.
  const onRetry = vi.fn();
  const ctrl = controllerWith([
    { type: 'user_input', text: 'do the thing', timestamp: '2026-07-21T06:00:00Z' },
    { type: 'model_response', raw_content: 'Starting…', thought: 'Starting…' },
    { type: 'session_cancelled', status: 'cancelled' },
  ]);
  const row = {
    status: 'cancelled',
    title: 'c',
    metadata: {},
    pinned: false,
    is_primary: false,
    busy: false,
  } as unknown as SessionRow;
  render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks, onRetry });

  const retry = page.getByTestId('chat-retry');
  await expect.element(retry).toBeVisible();
  await retry.click();
  expect(onRetry).toHaveBeenCalledWith('do the thing');
});

test('a resume_reset renders a calm notice line, not a red error alert', async () => {
  // The backend severed an unresumable provider session and continued from saved
  // history. The user should see a subdued informational line at the head of the
  // healed turn - never the role="alert" red error treatment.
  const RESET =
    "The chat's resumable model session was reset because it could no longer be " +
    'resumed; continuing from saved history.';
  const ctrl = controllerWith([
    { type: 'user_input', text: 'carry on', timestamp: '2026-07-18T10:00:00Z' },
    { type: 'resume_reset', reason: 'poisoned_transcript', message: RESET },
    { type: 'model_response', raw_content: 'Back on track.', thought: 'Back on track.' },
    { type: 'final_result', result: 'Back on track.' },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText(RESET)).toBeInTheDocument();
  // Calm status role, and the answer still rendered beneath it.
  const notice = document.querySelector('.t-turnnotice');
  expect(notice).toBeTruthy();
  expect(notice!.getAttribute('role')).toBe('status');
  await expect.element(page.getByText('Back on track.')).toBeInTheDocument();
  // Not an error: no alert role and no error-styled bar.
  expect(document.querySelector('.t-turnerr')).toBeFalsy();
  expect(document.querySelector('[role="alert"]')).toBeFalsy();
});

test('a compaction event renders the compaction banner', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'final_result', result: 'a' },
    { type: 'compaction', replaced_count: 40, retained_count: 6 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  expect(document.querySelector('[data-testid="chat-compaction-banner"]')).toBeTruthy();
});

test('a compaction with no summary shows the counts line and no disclosure', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'final_result', result: 'a' },
    { type: 'compaction', replaced_count: 40, retained_count: 6 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText(/40 turns/)).toBeInTheDocument();
  // Older compactions carry no summary text: the banner stays static, never an
  // expandable disclosure.
  expect(page.getByRole('button', { name: /context compacted/ }).query()).toBeNull();
});

test('a compaction summary is hidden until its disclosure is expanded', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'final_result', result: 'a' },
    {
      type: 'compaction',
      replaced_count: 40,
      retained_count: 6,
      summary: 'Refactored the auth module and added backoff to the retry client.',
    },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const toggle = page.getByRole('button', { name: /context compacted/ });
  await expect.element(toggle).toBeInTheDocument();
  // Collapsed by default: the summary prose isn't rendered yet.
  expect(page.getByText(/Refactored the auth module/).query()).toBeNull();
  await toggle.click();
  await expect.element(page.getByText(/Refactored the auth module/)).toBeVisible();
});

test('the context meter opens a prompt inspector with per-category tokens and total', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    {
      type: 'prompt_snapshot',
      token_breakdown: {
        categories: [
          { name: 'tools', tokens: 5000, items: [{ name: 'read_file', tokens: 10 }] },
          { name: 'history', tokens: 3000, items: [] },
          { name: 'attachments', tokens: 0, items: [] },
        ],
        total: 8000,
      },
    },
    { type: 'final_result', result: 'a' },
    { type: 'session_info', tokens: 8000, context_limit: 200000 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await page.getByRole('button', { name: /context breakdown/i }).click();
  await expect
    .element(page.getByRole('dialog', { name: /context breakdown/i }))
    .toBeInTheDocument();
  await expect.element(page.getByText('tools', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('history', { exact: true })).toBeInTheDocument();
  // Zero-token categories are dropped (nothing to inspect).
  expect(page.getByText('attachments', { exact: true }).query()).toBeNull();
  // The snapshot total is surfaced ("8k").
  await expect.element(page.getByText('8k', { exact: true })).toBeInTheDocument();
});

test('with no prompt snapshot the meter is inert (no inspector to open)', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'final_result', result: 'a' },
    { type: 'session_info', tokens: 8000, context_limit: 200000 },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  // The meter still renders its readout...
  await expect.element(page.getByText('8k/200k')).toBeInTheDocument();
  // ...but with no snapshot there's no breakdown trigger to open.
  expect(page.getByRole('button', { name: /context breakdown/i }).query()).toBeNull();
});

test('an in-flight send with no visible activity shows the waiting-on-model line', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'do the thing', timestamp: 'z' }]);
  ctrl.streaming = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('waiting')).toBeInTheDocument();
  await expect.element(page.getByText('on the model')).toBeInTheDocument();
  // The composer owns the single Stop control; the waiting line has none.
  expect(document.querySelectorAll('.t-work button').length).toBe(0);
});

test('the waiting line shows the agent-loop position while the turn runs', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'do the thing', timestamp: 'z' },
    { type: 'turn_start', turn: 3, max_turns: 20 },
  ]);
  ctrl.streaming = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('turn 3 / 20')).toBeInTheDocument();
});

test('the waiting line shows the turn alone when the run reports no limit', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'do the thing', timestamp: 'z' },
    { type: 'turn_start', turn: 3 },
  ]);
  ctrl.streaming = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText(/turn 3$/)).toBeInTheDocument();
});

test('a running tool suppresses the waiting line (its own spinner carries the signal)', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'grep', timestamp: 'z' },
    { type: 'tool_call', tool: 'run', command: 'rg foo', call_id: 'c1' },
  ]);
  ctrl.streaming = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText(/rg foo/)).toBeInTheDocument();
  await expect.element(page.getByText('on the model')).not.toBeInTheDocument();
});

test('no waiting line renders for a settled conversation', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'final_result', result: 'a' },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('a', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('on the model')).not.toBeInTheDocument();
});

test('switching back to a still-running session restores the waiting line from server busy', async () => {
  // On switch-back the local per-chat stream is gone (streaming=false) and the
  // in-flight turn's live-only frames (turn_start/stream_chunk) were never
  // persisted, so replay ends at the user_input. The session's server-side busy
  // status is the truth that restores the working indicator.
  const ctrl = controllerWith([{ type: 'user_input', text: 'do the thing', timestamp: 'z' }]);
  ctrl.serverBusy = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('waiting')).toBeInTheDocument();
  await expect.element(page.getByText('on the model')).toBeInTheDocument();
});

test('a session whose turn ended while away shows no waiting line despite a lagging busy flag', async () => {
  // Turn-end clears row.busy via a debounced list revalidate, so busy can still
  // read true for a beat after final_result lands in replay. The closed turn in
  // the timeline must win - no stranded spinner.
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'final_result', result: 'done' },
  ]);
  ctrl.serverBusy = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('done', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('on the model')).not.toBeInTheDocument();
});

test('completing the turn while viewing clears the waiting line even if busy still lags', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'q', timestamp: 'z' }]);
  ctrl.streaming = true;
  ctrl.serverBusy = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('on the model')).toBeInTheDocument();
  // The turn ends: final_result closes it and the local stream stops, but the
  // busy broadcast hasn't cleared row.busy yet.
  ctrl.events = [...ctrl.events, { type: 'final_result', result: 'done' }];
  ctrl.streaming = false;
  await expect.element(page.getByText('on the model')).not.toBeInTheDocument();
});

test('token-stream deltas render live in the AI turn and replace the waiting line', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'turn_start' },
    { type: 'stream_chunk', chunk: 'Let me check the ' },
    { type: 'stream_chunk', chunk: 'workspace first.' },
  ]);
  ctrl.streaming = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('Let me check the workspace first.')).toBeInTheDocument();
  await expect.element(page.getByText('on the model')).not.toBeInTheDocument();
});

test('an open streamed fence renders in the code panel with the streaming flag', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'stream_chunk', chunk: 'Computing.\n\n```python-exec\nx = 6 * 7\nprint(x)' },
  ]);
  ctrl.streaming = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('Computing.')).toBeInTheDocument();
  const code = document.querySelector('.t-code.is-streaming');
  expect(code?.textContent).toContain('x = 6 * 7');
  expect(code?.querySelector('.streamflag')?.textContent).toContain('streaming');
});

test('a hook_status tick names the waiting line while the turn streams', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'deploy it', timestamp: '2026-07-17T10:00:00Z' },
    { type: 'turn_start' },
    { type: 'hook_status', message: 'Running precommit...' },
  ]);
  ctrl.streaming = true;
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('hook · precommit')).toBeInTheDocument();
});

test('a replayed hook_execution renders as a hook exec row', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'deploy it', timestamp: '2026-07-17T10:00:00Z' },
    { type: 'hook_execution', name: 'precommit', exit_code: 1, stderr: 'lint failed' },
    {
      type: 'model_response',
      raw_content: 'Fixing the lint error.',
      thought: 'Fixing the lint error.',
    },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByRole('button', { name: /hook:precommit/ })).toBeInTheDocument();
  await expect.element(page.getByText('exit 1')).toBeInTheDocument();
  await expect.element(page.getByText('lint failed')).toBeInTheDocument();
});

test('at phone width the header shows a back affordance and hides the rail toggle', async () => {
  // Phone drilldown: the conversation is a full screen reached from the list, so
  // its header leads with a back control (‹), not the desktop rail-collapse toggle.
  await page.viewport(390, 780);
  const ctrl = controllerWith([{ type: 'user_input', text: 'q', timestamp: 'z' }]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByTestId('phone-back')).toBeVisible();
  await expect.element(page.getByTestId('chat-rail-toggle')).not.toBeVisible();
});

test('the back affordance invokes onBack (returns to the sessions list)', async () => {
  await page.viewport(390, 780);
  const onBack = vi.fn();
  const ctrl = controllerWith([{ type: 'user_input', text: 'q', timestamp: 'z' }]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks, onBack });
  await page.getByTestId('phone-back').click();
  expect(onBack).toHaveBeenCalledTimes(1);
});

test('at desktop width the rail toggle shows and the back affordance is hidden', async () => {
  // Desktop/tablet keep the side-by-side rail + its collapse toggle, unchanged.
  await page.viewport(1440, 900);
  const ctrl = controllerWith([{ type: 'user_input', text: 'q', timestamp: 'z' }]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByTestId('chat-rail-toggle')).toBeVisible();
  await expect.element(page.getByTestId('phone-back')).not.toBeVisible();
});

test('a scheduled-task injection renders as a folded sched turn, not as "you"', async () => {
  const ctrl = controllerWith([
    {
      type: 'user_input',
      text: '<scheduled_task id="nightly-backup">\nThis task ran in the background and the result was sent as a notification to the user.\n</scheduled_task>',
      injected: [
        {
          tag: 'scheduled_task',
          id: 'nightly-backup',
          body: 'This task ran in the background and the result was sent as a notification to the user.',
        },
      ],
      display_text: '',
      timestamp: '2026-07-17T05:30:00Z',
    },
    {
      type: 'model_response',
      raw_content: 'Backup verified, 12 files.',
      thought: 'Backup verified, 12 files.',
    },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('sched', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('you', { exact: true })).not.toBeInTheDocument();
  // The wrapper boilerplate lives inside a collapsed panel titled by tag + id.
  await expect.element(page.getByText('scheduled_task', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('nightly-backup', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('Backup verified, 12 files.')).toBeInTheDocument();
});

test('while pinned, streamed frames keep the transcript scrolled to the tail', async () => {
  const ctrl = streamingController();
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  appendChunk(ctrl, tall(40));
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6);
  expect(document.querySelector('.jumplive')).toBeFalsy();
});

test('a wheel-up gesture unpins so later frames hold the viewport instead of yanking it down', async () => {
  const ctrl = streamingController();
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  appendChunk(ctrl, tall(40));
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6); // pinned at the tail
  wheelUp(el);
  await expect.element(page.getByText('jump to live')).toBeInTheDocument(); // unpinned
  // More stream lands: the viewport must stay where the user left it, so the new
  // content grows below and the distance from the bottom increases (no snap-back).
  appendChunk(ctrl, tall(40));
  await expect.poll(() => distFromBottom(el)).toBeGreaterThan(80);
});

test('a touch drag upward unpins during a stream', async () => {
  await page.viewport(390, 780);
  const ctrl = streamingController();
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  appendChunk(ctrl, tall(40));
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6);
  const touch = (y: number) => new Touch({ identifier: 1, target: el, clientX: 20, clientY: y });
  el.dispatchEvent(
    new TouchEvent('touchstart', {
      touches: [touch(120)],
      changedTouches: [touch(120)],
      bubbles: true,
    }),
  );
  el.dispatchEvent(
    new TouchEvent('touchmove', {
      touches: [touch(260)],
      changedTouches: [touch(260)],
      bubbles: true,
    }),
  );
  await expect.element(page.getByText('jump to live')).toBeInTheDocument();
});

test('scrolling back to the bottom re-pins and following resumes', async () => {
  const ctrl = streamingController();
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  appendChunk(ctrl, tall(40));
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6);
  wheelUp(el);
  await expect.element(page.getByText('jump to live')).toBeInTheDocument();
  // The gesture carried the viewport up (a synthetic wheel doesn't scroll, so
  // move it by hand); being away from the tail must NOT re-pin on its own.
  el.scrollTop = 0;
  await expect.element(page.getByText('jump to live')).toBeInTheDocument();
  el.scrollTop = el.scrollHeight; // user scrolls back to the tail
  await expect.element(page.getByText('jump to live')).not.toBeInTheDocument(); // re-pinned
  appendChunk(ctrl, tall(20));
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6); // following again
});

test('the jump-to-latest affordance re-pins and snaps to the tail on click', async () => {
  const ctrl = streamingController();
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  appendChunk(ctrl, tall(40));
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6);
  wheelUp(el);
  appendChunk(ctrl, tall(40));
  await expect.poll(() => distFromBottom(el)).toBeGreaterThan(80); // held, unpinned
  await page.getByText('jump to live').click();
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6); // snapped to the tail
  expect(document.querySelector('.jumplive')).toBeFalsy();
});

test("this surface's own send re-pins even when the reader had scrolled up", async () => {
  // Idle, tall thread the user has scrolled up through.
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'model_response', raw_content: tall(40), thought: tall(40) },
    { type: 'final_result', result: 'done' },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6); // opens at the tail
  wheelUp(el);
  el.scrollTop = 0; // reading history
  await expect.element(page.getByText('jump to live')).toBeInTheDocument();
  // Mimic a local send: ConversationController.send flips streaming false->true
  // and appends the optimistic user turn. That own-send signal must re-pin.
  ctrl.events = [...ctrl.events, { type: 'user_input', text: 'and again', timestamp: 'z' }];
  ctrl.streaming = true;
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6);
  expect(document.querySelector('.jumplive')).toBeFalsy();
});

test('a local command echo renders at the conversation tail (local-only footer)', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'final_result', result: 'a' },
  ]);
  ctrl.pushEcho('/status', 'Model: claude_code:haiku · 0/128k tokens', true);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('/status')).toBeInTheDocument();
  await expect.element(page.getByText(/Model: claude_code:haiku/)).toBeInTheDocument();
  await expect.element(page.getByText(/not saved/)).toBeInTheDocument();
});

test('a local command echo can be dismissed from its footer', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'final_result', result: 'a' },
  ]);
  ctrl.pushEcho('/status', 'Model: haiku', true);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  await expect.element(page.getByText('/status')).toBeInTheDocument();

  await page.getByRole('button', { name: 'Dismiss' }).click();

  expect(ctrl.localEcho).toHaveLength(0);
  await expect.element(page.getByText('/status')).not.toBeInTheDocument();
});

test('an echo pushed while scrolled up never force-scrolls (its own channel, not the follow trigger)', async () => {
  // The follow effect watches events/turns length only; localEcho is a separate
  // array, so appending an echo must not yank an unpinned reader to the tail.
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'model_response', raw_content: tall(40), thought: tall(40) },
    { type: 'final_result', result: 'done' },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6);
  wheelUp(el);
  el.scrollTop = 0; // reading history, unpinned
  await expect.element(page.getByText('jump to live')).toBeInTheDocument();
  const before = el.scrollTop;
  ctrl.pushEcho('/status', 'Model: haiku', true);
  await new Promise((r) => setTimeout(r, 80));
  expect(Math.abs(el.scrollTop - before)).toBeLessThan(6); // no yank to the tail
  await expect.element(page.getByText('jump to live')).toBeInTheDocument(); // still unpinned
});

test('a background resync that replaces events leaves an unpinned reader where they were', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'q', timestamp: 'z' },
    { type: 'model_response', raw_content: tall(40), thought: tall(40) },
    { type: 'final_result', result: 'done' },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const el = scrollBox();
  await expect.poll(() => distFromBottom(el)).toBeLessThan(6);
  wheelUp(el);
  await expect.element(page.getByText('jump to live')).toBeInTheDocument();
  el.scrollTop = 40; // parked mid-thread
  const before = el.scrollTop;
  // resync replaces the events array wholesale; turn ids are a deterministic
  // sequence, so the keyed DOM is preserved and an unpinned reader must not jump.
  ctrl.events = ctrl.events.map((e) => ({ ...e }));
  await new Promise((r) => setTimeout(r, 80));
  expect(Math.abs(el.scrollTop - before)).toBeLessThan(6);
  await expect.element(page.getByText('jump to live')).toBeInTheDocument(); // still unpinned
});

// --- Status pill: compaction ------------------------------------------------
// The pill's `compacting` state must come from the server's authoritative
// per-session flag, never from matching text in the progress label.

function pillState(): string | null {
  return document.querySelector('.t-pill')?.getAttribute('data-st') ?? null;
}

/** A live session row; the two fields the pill tests vary are the arguments. */
function pillRow(compacting: boolean, statusText: string): SessionRow {
  return {
    id: 'sess-1',
    status: 'active',
    title: 'c',
    metadata: {},
    pinned: false,
    is_primary: false,
    busy: true,
    compacting,
    progress: { turn_count: 3, tool_count: 1, status_text: statusText, last_event_time: null },
  } as unknown as SessionRow;
}

test('a compaction in flight lights the pill even with no compaction hooks configured', async () => {
  // No workspace hook means the progress label never contains the word
  // "compact" - it reads whatever the last mid-turn event said.
  await page.viewport(1440, 900);
  const ctrl = controllerWith([{ type: 'user_input', text: 'q', timestamp: 'z' }]);
  const row = pillRow(true, 'Waiting on LLM...');
  render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks });
  await page.screenshot({ path: '__screenshots__/compacting-pill.png' });
  await expect.poll(pillState).toBe('compacting');
});

test('a hook message mentioning compaction does not by itself light the pill', async () => {
  // hook_status text is free-form user configuration; it must not be read as a
  // session state signal.
  const ctrl = controllerWith([{ type: 'user_input', text: 'q', timestamp: 'z' }]);
  const row = pillRow(false, 'pre_compact hook: archiving transcript');
  render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks });
  await expect.poll(pillState).toBe('busy');
});

function metadataRow(metadata: Record<string, unknown>): SessionRow {
  return { ...pillRow(false, ''), metadata } as unknown as SessionRow;
}

test('url-valued session metadata renders as header chips, labelled by key or label', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'review', timestamp: '2026-08-09T10:00:00Z' },
  ]);
  const row = metadataRow({
    pr: { url: 'https://forge.example/justyns/tsugite/pulls/608', label: 'PR #608' },
    task: 'https://vikunja.example/tasks/42',
  });
  render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks });

  const pr = page.getByRole('link', { name: 'PR #608' });
  await expect
    .element(pr)
    .toHaveAttribute('href', 'https://forge.example/justyns/tsugite/pulls/608');

  // No `label`, so the key names the chip.
  const task = page.getByRole('link', { name: 'task' });
  await expect.element(task).toHaveAttribute('href', 'https://vikunja.example/tasks/42');
});

test('metadata that is not an http(s) url gets no chip', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'review', timestamp: '2026-08-09T10:00:00Z' },
  ]);
  const row = metadataRow({
    pr: '608',
    notes: 'ready for review',
    evil: 'javascript:alert(1)',
    sneaky: '  javascript:alert(1)',
    type: { url: 42 },
  });
  render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks });

  expect(document.querySelectorAll(`[data-testid="${TESTID.chatMetaLink}"]`)).toHaveLength(0);
});

test('session menu shows raw session metadata, with the links reachable from it', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'review', timestamp: '2026-08-09T10:00:00Z' },
  ]);
  const row = metadataRow({
    pr: 'https://forge.example/justyns/tsugite/pulls/608',
    status_text: 'review posted',
  });
  render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks });

  await page.getByTestId(TESTID.chatSessionMenuTrigger).click();
  await page.getByRole('menuitem', { name: /view metadata/i }).click();

  const dialog = page.getByTestId(TESTID.chatRawMetadata);
  await expect.element(dialog).toBeInTheDocument();
  await expect.element(dialog).toHaveTextContent('"status_text"');
  await expect.element(dialog).toHaveTextContent('https://forge.example/justyns/tsugite/pulls/608');

  // Narrow screens hide the header chips, so the dialog carries the only link a
  // phone can reach.
  await expect
    .element(dialog.getByRole('link', { name: 'pr' }))
    .toHaveAttribute('href', 'https://forge.example/justyns/tsugite/pulls/608');
});

test('raw metadata shows the session fields that live outside the metadata block', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'review', timestamp: '2026-08-09T10:00:00Z' },
  ]);
  const row = metadataRow({ status_text: 'review posted' });
  const detail = {
    id: 'sess-1',
    title: 'Nightly deploy check',
    alias: 'daily',
    model: 'codex_cli:gpt-5.5',
    model_override: 'claude_code:opus',
    reasoning_effort: 'high',
    agent_file: 'odyn-daemon',
    metadata: { status_text: 'review posted' },
    prompt: 'SHOULD NOT RENDER',
    result: 'SHOULD NOT RENDER',
    deferred_deliveries: [{ message: 'SHOULD NOT RENDER' }],
  };
  // A Response body reads once, so every call needs its own.
  const fetchSpy = vi
    .spyOn(globalThis, 'fetch')
    .mockImplementation(async (input) =>
      String(input).includes('/api/sessions/')
        ? new Response(JSON.stringify(detail), { headers: { 'content-type': 'application/json' } })
        : new Response('{}', { headers: { 'content-type': 'application/json' } }),
    );
  try {
    render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks });

    await page.getByTestId(TESTID.chatSessionMenuTrigger).click();
    await page.getByRole('menuitem', { name: /view metadata/i }).click();

    const dialog = page.getByTestId(TESTID.chatRawMetadata);
    await expect.element(dialog).toHaveTextContent('"alias"');
    await expect.element(dialog).toHaveTextContent('daily');
    await expect.element(dialog).toHaveTextContent('"model_override"');
    await expect.element(dialog).toHaveTextContent('claude_code:opus');
    await expect.element(dialog).toHaveTextContent('"title"');
    // The metadata block is still its own section.
    await expect.element(dialog).toHaveTextContent('"status_text"');
    // Content fields have their own surfaces and would swamp the overlay.
    await expect.element(dialog).not.toHaveTextContent('SHOULD NOT RENDER');
    // The effect must not read what it writes, or it refetches forever.
    const detailCalls = fetchSpy.mock.calls.filter(([input]) =>
      /\/api\/sessions\/[^/]+$/.test(String(input)),
    );
    expect(detailCalls).toHaveLength(1);
  } finally {
    fetchSpy.mockRestore();
  }
});

test('raw metadata still renders when the session detail fetch fails', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'review', timestamp: '2026-08-09T10:00:00Z' },
  ]);
  const row = metadataRow({ status_text: 'review posted' });
  const fetchSpy = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('offline'));
  try {
    render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks });

    await page.getByTestId(TESTID.chatSessionMenuTrigger).click();
    await page.getByRole('menuitem', { name: /view metadata/i }).click();

    const dialog = page.getByTestId(TESTID.chatRawMetadata);
    await expect.element(dialog).toBeInTheDocument();
    await expect.element(dialog).toHaveTextContent('"status_text"');
  } finally {
    fetchSpy.mockRestore();
  }
});

test('the jobs chip links to the jobs board filtered to this session', async () => {
  // The chip is a shortcut to "the jobs this chat spawned" - landing on the
  // unfiltered board makes the user re-find them by hand.
  await page.viewport(1440, 900);
  const ctrl = controllerWith([
    { type: 'user_input', text: 'ship it', timestamp: '2026-08-01T10:00:00Z' },
    { type: 'job_status', job_id: 'job-1', state: 'running', prompt: 'ship it' },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });
  const chip = document.querySelector('.hd-chip') as HTMLAnchorElement | null;
  expect(chip?.textContent).toContain('1 job');
  expect(chip?.getAttribute('href')).toBe('#jobs?q=session%3Asess-1');
});

// --- Delivery cards ---------------------------------------------------------
// `needsAck` is baked into the history event and never changes; whether THAT
// delivery is still outstanding is live truth on the session row.
const NEEDS_ACK = {
  type: 'delivery',
  source: 'schedule',
  kind: 'needs_ack',
  delivery_id: 'dlv-1',
  title: 'Rent is due',
  message: 'The rent run found an unpaid invoice.',
  timestamp: '2026-08-14T15:10:00Z',
  id: 9,
};

function attentionRow(pending: string[]): SessionRow {
  return {
    status: 'active',
    title: 'rent',
    metadata: {},
    pinned: false,
    is_primary: false,
    busy: false,
    needs_attention: pending.length > 0,
    pending_deliveries: pending,
  } as unknown as SessionRow;
}

test('an unacknowledged needs-ack card offers the dismiss control', async () => {
  const ctrl = controllerWith([NEEDS_ACK]);
  render(Conversation, { ctrl, row: attentionRow(['dlv-1']), railCollapsed: false, ...callbacks });

  await expect.element(page.getByTestId(TESTID.chatDelivery)).toHaveTextContent('needs you');
  await expect.element(page.getByTestId(TESTID.chatDeliveryDismiss)).toBeVisible();
});

test('a dismissed session stops offering dismiss on its needs-ack cards', async () => {
  // The event still says needs_ack forever; the row says the obligation is
  // discharged. Rendering the stale event leaves a live Dismiss button on every
  // card in the session, across reloads.
  const ctrl = controllerWith([NEEDS_ACK]);
  render(Conversation, { ctrl, row: attentionRow([]), railCollapsed: false, ...callbacks });

  const card = page.getByTestId(TESTID.chatDelivery);
  await expect.element(card).toHaveTextContent('The rent run found an unpaid invoice.');
  expect(card.element().textContent).not.toContain('needs you');
  expect(document.querySelector(`[data-testid="${TESTID.chatDeliveryDismiss}"]`)).toBeNull();
});

test('dismissing one of two cards leaves the other one outstanding', async () => {
  // One bool for N cards made the first dismiss silence every other obligation
  // in the chat.
  const ctrl = controllerWith([
    { ...NEEDS_ACK, delivery_id: 'dlv-1', message: 'rent is due friday', id: 9 },
    { ...NEEDS_ACK, delivery_id: 'dlv-2', message: 'approve the deploy?', id: 10 },
  ]);
  render(Conversation, { ctrl, row: attentionRow(['dlv-2']), railCollapsed: false, ...callbacks });

  const cards = document.querySelectorAll(`[data-testid="${TESTID.chatDelivery}"]`);
  expect(cards).toHaveLength(2);
  expect(cards[0]!.textContent).not.toContain('needs you');
  expect(cards[1]!.textContent).toContain('needs you');
  expect(document.querySelectorAll(`[data-testid="${TESTID.chatDeliveryDismiss}"]`)).toHaveLength(
    1,
  );
});

test('dismissing a card names the delivery it discharges', async () => {
  const dismissed: (string | undefined)[] = [];
  const ctrl = controllerWith([NEEDS_ACK]);
  render(Conversation, {
    ctrl,
    row: attentionRow(['dlv-1']),
    railCollapsed: false,
    ...callbacks,
    onDismissAttention: (id?: string) => dismissed.push(id),
  });

  await page.getByTestId(TESTID.chatDeliveryDismiss).click();

  expect(dismissed).toEqual(['dlv-1']);
});

test('two cards from different schedules show each schedule id', async () => {
  // Without this every card reads "SCHEDULE", so a session fed by several
  // schedules gives the reader no way to tell them apart.
  const ctrl = controllerWith([
    { ...NEEDS_ACK, delivery_id: 'dlv-1', schedule_id: 'nightly-backup', id: 9 },
    { ...NEEDS_ACK, delivery_id: 'dlv-2', schedule_id: 'weekly-digest', id: 10 },
  ]);
  render(Conversation, { ctrl, row: attentionRow([]), railCollapsed: false, ...callbacks });

  const cards = document.querySelectorAll(`[data-testid="${TESTID.chatDelivery}"]`);
  expect(cards).toHaveLength(2);
  expect(cards[0]!.textContent).toContain('nightly-backup');
  expect(cards[1]!.textContent).toContain('weekly-digest');
});

test('a card carrying no origin id keeps the plain source header', async () => {
  const ctrl = controllerWith([NEEDS_ACK]);
  render(Conversation, { ctrl, row: attentionRow([]), railCollapsed: false, ...callbacks });

  const card = page.getByTestId(TESTID.chatDelivery);
  await expect.element(card).toHaveTextContent('schedule');
  expect(card.element().querySelector('.dlv-origin')).toBeNull();
});

test('the alias renders as its own chip, distinct from the title', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'hi', timestamp: 'z' }]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks, alias: 'daily' });

  const chip = page.getByTestId(TESTID.chatAlias);
  await expect.element(chip).toBeInTheDocument();
  await expect.element(chip).toHaveTextContent('daily');
});

test('a session holding no alias renders no chip', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'hi', timestamp: 'z' }]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });

  expect(document.querySelector(`[data-testid="${TESTID.chatAlias}"]`)).toBeFalsy();
});

test('removing the alias from its chip commits an empty value', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'hi', timestamp: 'z' }]);
  const committed: string[] = [];
  render(Conversation, {
    ctrl,
    row: null,
    railCollapsed: false,
    ...callbacks,
    alias: 'daily',
    onAliasCommit: (value: string) => committed.push(value),
  });

  await page.getByLabelText('Remove alias').click();

  expect(committed).toEqual(['']);
});

async function openAliasEditor(menuItem: string) {
  await page.getByTestId(TESTID.chatSessionMenuTrigger).click();
  await page.getByRole('menuitem', { name: menuItem }).click();
  return page.getByLabelText('Set session alias');
}

test('set alias prefills a slug of the display name', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'hi', timestamp: 'z' }]);
  render(Conversation, {
    ctrl,
    row: sessionRow('s1', { title: 'Nightly deploy check' }),
    railCollapsed: false,
    ...callbacks,
  });

  const field = await openAliasEditor('Set alias');

  await expect.element(field).toHaveValue('nightly-deploy-check');
});

test('change alias prefills the alias it already holds, not a fresh suggestion', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'hi', timestamp: 'z' }]);
  render(Conversation, {
    ctrl,
    row: sessionRow('s1', { title: 'Nightly deploy check' }),
    railCollapsed: false,
    ...callbacks,
    alias: 'daily',
  });

  const field = await openAliasEditor('Change alias');

  await expect.element(field).toHaveValue('daily');
});

test('a malformed alias never commits, and says why', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'hi', timestamp: 'z' }]);
  const committed: string[] = [];
  render(Conversation, {
    ctrl,
    row: sessionRow('s1', { title: 'Nightly deploy check' }),
    railCollapsed: false,
    ...callbacks,
    onAliasCommit: (value: string) => committed.push(value),
  });

  const field = await openAliasEditor('Set alias');
  await field.fill('bad alias!');
  await userEvent.keyboard('{Enter}');

  expect(committed).toEqual([]);
  await expect.element(field).toBeInTheDocument();
  await expect.element(page.getByText(/Start with a letter or digit/)).toBeInTheDocument();
});

test('clearing the field commits an empty value, which releases the alias', async () => {
  const ctrl = controllerWith([{ type: 'user_input', text: 'hi', timestamp: 'z' }]);
  const committed: string[] = [];
  render(Conversation, {
    ctrl,
    row: sessionRow('s1', { title: 'Nightly deploy check' }),
    railCollapsed: false,
    ...callbacks,
    alias: 'daily',
    onAliasCommit: (value: string) => committed.push(value),
  });

  const field = await openAliasEditor('Change alias');
  await field.fill('');
  await expect.element(page.getByText('Clearing this removes the alias.')).toBeInTheDocument();
  await userEvent.keyboard('{Enter}');

  expect(committed).toEqual(['']);
});

test('a resumable background chat reads as idle in the header, not completed', async () => {
  const ctrl = controllerWith([
    { type: 'user_input', text: 'hi', timestamp: '2026-07-14T15:00:00Z' },
  ]);
  const row = sessionRow('sess-1', {
    source: 'background',
    status: 'completed',
    resumable: true,
  });
  render(Conversation, { ctrl, row, railCollapsed: false, ...callbacks });

  await expect.element(page.getByText('idle', { exact: true })).toBeInTheDocument();
  expect(await page.getByText('completed', { exact: true }).elements()).toHaveLength(0);
});

// --- Cross-session attribution ---------------------------------------------

function crossSessionController(): ConversationController {
  return controllerWith([
    {
      type: 'user_input',
      text: 'status on the migration?',
      timestamp: '2026-07-14T15:00:00Z',
      channel: { source: 'session', from_session: 'lead-agent' },
    },
    { type: 'model_response', raw_content: 'Two files left.', thought: 'Two files left.' },
    { type: 'final_result', result: 'Two files left.' },
  ]);
}

test('a message from another session is badged with its sender, not shown as yours', async () => {
  render(Conversation, {
    ctrl: crossSessionController(),
    row: null,
    railCollapsed: false,
    ...callbacks,
  });

  const badge = page.getByTestId(TESTID.chatTurnOrigin);
  await expect.element(badge).toHaveTextContent('lead-agent');
  // The gutter names the sender instead of claiming the person typed it.
  await expect.element(page.getByText('session', { exact: true })).toBeInTheDocument();
  expect(await page.getByText('you', { exact: true }).elements()).toHaveLength(0);
});

test('the sender badge links to the session it came from', async () => {
  render(Conversation, {
    ctrl: crossSessionController(),
    row: null,
    railCollapsed: false,
    ...callbacks,
  });

  await expect
    .element(page.getByTestId(TESTID.chatTurnOrigin))
    .toHaveAttribute('href', '#chats?sessionId=lead-agent');
});

test('the answer says it went back to the sender', async () => {
  render(Conversation, {
    ctrl: crossSessionController(),
    row: null,
    railCollapsed: false,
    ...callbacks,
  });

  await expect
    .element(page.getByTestId(TESTID.chatTurnReplyTo))
    .toHaveTextContent('replied to lead-agent');
});

test('a message the person typed carries no sender badge', async () => {
  const ctrl = controllerWith([
    {
      type: 'user_input',
      text: 'status on the migration?',
      timestamp: '2026-07-14T15:00:00Z',
      channel: { source: 'http', user_id: 'web-alice' },
    },
    { type: 'final_result', result: 'Two files left.' },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });

  await expect.element(page.getByText('you', { exact: true })).toBeInTheDocument();
  expect(await page.getByTestId(TESTID.chatTurnOrigin).elements()).toHaveLength(0);
  expect(await page.getByTestId(TESTID.chatTurnReplyTo).elements()).toHaveLength(0);
});

test('a session-completion wake-up folds instead of rendering raw XML as your message', async () => {
  const ctrl = controllerWith([
    {
      type: 'user_input',
      text: '<session_finished id="child" status="completed" title="Do the thing">\nall done\n</session_finished>',
      display_text: '',
      injected: [
        {
          tag: 'session_finished',
          id: 'child',
          status: 'completed',
          title: 'Do the thing',
          body: 'all done',
        },
      ],
      timestamp: '2026-07-14T15:00:00Z',
      channel: { source: 'session_completion', from_session: 'child' },
    },
  ]);
  render(Conversation, { ctrl, row: null, railCollapsed: false, ...callbacks });

  expect(await page.getByText('<session_finished', { exact: false }).elements()).toHaveLength(0);
  await expect.element(page.getByTestId(TESTID.chatTurnOrigin)).toHaveTextContent('child');
});

function reasoningController(): ConversationController {
  return controllerWith([
    { type: 'user_input', text: 'which option?', timestamp: '2026-07-14T15:00:00Z' },
    { type: 'reasoning', content: 'weighing the options' },
    { type: 'model_response', raw_content: 'Option A.', thought: 'Option A.' },
    { type: 'final_result', result: 'Option A.' },
  ]);
}

test('a thinking block renders expanded by default', async () => {
  expandThinking.set(true);
  render(Conversation, {
    ctrl: reasoningController(),
    row: null,
    railCollapsed: false,
    ...callbacks,
  });
  await expect
    .element(page.getByRole('button', { name: /thinking/ }))
    .toHaveAttribute('aria-expanded', 'true');
});

test('turning the thinking pref off renders the block collapsed', async () => {
  expandThinking.set(false);
  render(Conversation, {
    ctrl: reasoningController(),
    row: null,
    railCollapsed: false,
    ...callbacks,
  });
  await expect
    .element(page.getByRole('button', { name: /thinking/ }))
    .toHaveAttribute('aria-expanded', 'false');
});

test('flipping the thinking pref collapses a block already on screen', async () => {
  expandThinking.set(true);
  render(Conversation, {
    ctrl: reasoningController(),
    row: null,
    railCollapsed: false,
    ...callbacks,
  });
  const toggle = page.getByRole('button', { name: /thinking/ });
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'true');

  expandThinking.set(false);
  await expect.element(toggle).toHaveAttribute('aria-expanded', 'false');
});
