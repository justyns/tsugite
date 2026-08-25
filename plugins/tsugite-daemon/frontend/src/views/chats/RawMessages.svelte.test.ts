/// <reference types="vitest/browser" />
import { page } from 'vitest/browser';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import RawMessages from './RawMessages.svelte';
import type { RawMessages as RawMessagesData } from './rawMessages';

function mockRaw(payload: RawMessagesData | null) {
  vi.mocked(api.get).mockResolvedValue({ raw_messages: payload });
}

afterEach(cleanup);
beforeEach(() => vi.clearAllMocks());

const twoTurns: RawMessagesData = {
  system_prompt: null,
  turns: [
    {
      index: 1,
      turn: 0,
      provider: 'anthropic',
      model: 'claude-x',
      request: [{ role: 'user', content: 'hello there' }],
      new_messages: [{ role: 'user', content: 'hello there' }],
      reset_before: false,
      response: { raw_content: 'first reply' },
    },
    {
      index: 2,
      turn: 1,
      provider: 'anthropic',
      model: 'claude-x',
      request: [
        { role: 'user', content: 'hello there' },
        { role: 'assistant', content: 'first reply' },
        { role: 'user', content: 'again please' },
      ],
      new_messages: [
        { role: 'assistant', content: 'first reply' },
        { role: 'user', content: 'again please' },
      ],
      reset_before: false,
      response: { raw_content: 'second reply' },
    },
  ],
};

test('renders one collapsible section per model call, showing role + content', async () => {
  mockRaw(twoTurns);
  render(RawMessages, { sessionId: 's1', onClose: () => {} });

  await expect.element(page.getByRole('dialog', { name: 'raw messages' })).toBeInTheDocument();
  // A summary per call, labelled by the monotonic index (not the repeating turn).
  await expect.element(page.getByText('call 1')).toBeInTheDocument();
  await expect.element(page.getByText('call 2')).toBeInTheDocument();
  // Each call's delta content, a role label, and the raw response are in the DOM.
  // ("again please" also appears in the full-prompt disclosure, hence .first().)
  await expect.element(page.getByText('again please').first()).toBeInTheDocument();
  await expect.element(page.getByText('assistant').first()).toBeInTheDocument();
  await expect.element(page.getByText('second reply')).toBeInTheDocument();
});

test('a later call shows just its delta, with the full prompt behind a disclosure', async () => {
  mockRaw(twoTurns);
  render(RawMessages, { sessionId: 's1', onClose: () => {} });
  // Call 2 added two messages over call 1: the label says so, and the whole
  // 3-message prompt is available without being the default view.
  await expect.element(page.getByText('request · new this call')).toBeInTheDocument();
  await expect.element(page.getByText('full prompt · 3 messages')).toBeInTheDocument();
});

test('a null system prompt shows the muted not-shown note', async () => {
  mockRaw(twoTurns);
  render(RawMessages, { sessionId: 's1', onClose: () => {} });
  await expect.element(page.getByText('system prompt not shown')).toBeInTheDocument();
});

test('an empty turn list renders a truthful note, not a blank body', async () => {
  mockRaw({ system_prompt: null, turns: [] });
  render(RawMessages, { sessionId: 's1', onClose: () => {} });
  await expect.element(page.getByText('no model turns recorded yet')).toBeInTheDocument();
});

test('the close button fires onClose', async () => {
  mockRaw(twoTurns);
  const onClose = vi.fn();
  render(RawMessages, { sessionId: 's1', onClose });
  await expect.element(page.getByRole('dialog')).toBeInTheDocument();
  await page.getByRole('button', { name: 'Close' }).click();
  expect(onClose).toHaveBeenCalled();
});
