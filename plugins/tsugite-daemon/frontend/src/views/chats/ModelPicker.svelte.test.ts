/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn(), patch: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { sessions } from '$lib/stores/sessions.svelte';
import { modelPickerRequest } from './modelPickerSignal.svelte';
import ModelPicker from './ModelPicker.svelte';

const MODELS = {
  models: [
    {
      id: 'acp:claude-opus-4-7',
      provider: 'acp',
      context_window: 1_000_000,
      input_cost_per_million: 5,
      output_cost_per_million: 25,
      supports_vision: true,
      supports_reasoning: true,
    },
    {
      id: 'openai:gpt-5.4-mini',
      provider: 'openai',
      context_window: 400_000,
      input_cost_per_million: 0.15,
      output_cost_per_million: 0.6,
      supports_vision: true,
      supports_reasoning: false,
    },
    {
      id: 'openai:gpt-5.4',
      provider: 'openai',
      context_window: 400_000,
      input_cost_per_million: 3,
      output_cost_per_million: 15,
      supports_reasoning: true,
    },
  ],
};

beforeEach(() => {
  vi.mocked(api.get).mockImplementation((path: string) => {
    if (path.endsWith('/settings'))
      return Promise.resolve({
        model: 'acp:claude-opus-4-7',
        reasoning_effort: null,
        agent: 'smoke',
      });
    if (path === '/api/models') return Promise.resolve(MODELS);
    return Promise.reject(new Error(`unexpected GET ${path}`));
  });
  vi.mocked(api.patch).mockImplementation((_path: string, body: unknown) =>
    Promise.resolve({
      model: (body as { model: string }).model,
      reasoning_effort: null,
      agent: 'smoke',
    }),
  );
});
// Drain any signal a test left pending (a mismatched request stays set) so the
// module-scoped store can't leak an open-picker request into the next test.
afterEach(() => {
  cleanup();
  const p = modelPickerRequest.pending;
  if (p) modelPickerRequest.consume(p.sessionId);
});

test('shows the current model as a short label and opens a filterable popover', async () => {
  render(ModelPicker, { sessionId: 's1', agent: 'smoke' });
  const trigger = page.getByTestId('chat-model-trigger');
  // The provider prefix is dropped in the compact chip.
  await expect.element(trigger).toHaveTextContent('claude-opus-4-7');

  await trigger.click();
  await expect.element(page.getByTestId('chat-model-popover')).toBeInTheDocument();
  await expect.element(page.getByTestId('chat-model-opt-openai:gpt-5.4-mini')).toBeInTheDocument();

  // type=search keeps Chromium's built-in password manager off the autofocused
  // filter; autocomplete=off (from pwmIgnore) covers the extension managers.
  const search = page.getByTestId('chat-model-search');
  await expect.element(search).toHaveAttribute('type', 'search');
  await expect.element(search).toHaveAttribute('autocomplete', 'off');

  // Filtering narrows the list.
  await page.getByTestId('chat-model-search').fill('mini');
  await expect.element(page.getByTestId('chat-model-opt-openai:gpt-5.4-mini')).toBeInTheDocument();
  await expect
    .element(page.getByTestId('chat-model-opt-acp:claude-opus-4-7'))
    .not.toBeInTheDocument();
});

test('selecting a model PATCHes the session settings', async () => {
  render(ModelPicker, { sessionId: 's1', agent: 'smoke' });
  await page.getByTestId('chat-model-trigger').click();
  await page.getByTestId('chat-model-opt-openai:gpt-5.4').click();
  expect(vi.mocked(api.patch)).toHaveBeenCalledWith('/api/sessions/s1/settings', {
    model: 'openai:gpt-5.4',
  });
});

test('flips the popover left when a clipping ancestor would cut it off', async () => {
  // In the app the conversation pane body scrolls (overflow:auto), so a
  // right-anchored popover hanging left past the pane edge is clipped under
  // the sessions rail - the flip must measure against that ancestor, not the
  // viewport. A short chip label (e.g. "gpt-5.5") is what pulls the popover
  // left enough to cross the boundary.
  render(ModelPicker, { sessionId: 's1', agent: 'smoke' });
  const trigger = page.getByTestId('chat-model-trigger');
  const host = (trigger.element() as HTMLElement).closest('div')!.parentElement as HTMLElement;
  host.style.cssText +=
    ';display:block;position:relative;overflow:auto;margin-left:400px;width:220px;';

  await trigger.click();
  await expect
    .element(page.getByTestId('chat-model-popover'))
    .toHaveAttribute('data-align', 'left');
});

test('a settings broadcast refetches and updates the model chip live', async () => {
  let model = 'acp:claude-opus-4-7';
  vi.mocked(api.get).mockImplementation((path: string) => {
    if (path.endsWith('/settings'))
      return Promise.resolve({ model, reasoning_effort: null, agent: 'smoke' });
    if (path === '/api/models') return Promise.resolve(MODELS);
    return Promise.reject(new Error(`unexpected GET ${path}`));
  });
  render(ModelPicker, { sessionId: 's-bcast', agent: 'smoke' });
  const trigger = page.getByTestId('chat-model-trigger');
  await expect.element(trigger).toHaveTextContent('claude-opus-4-7');

  // Another tab (or /model from Discord) changed the model - the broadcast bumps
  // settingsRev, and the chip refetches without a manual reopen.
  model = 'openai:gpt-5.4';
  sessions.applySessionUpdate({ action: 'settings', id: 's-bcast', model, reasoning_effort: null });
  await expect.element(trigger).toHaveTextContent('gpt-5.4');
});

test('opens the popover when a model-picker signal targets this session', async () => {
  render(ModelPicker, { sessionId: 's-open', agent: 'smoke' });
  // Starts closed - the trigger is present but no popover.
  await expect.element(page.getByTestId('chat-model-trigger')).toBeInTheDocument();
  await expect.element(page.getByTestId('chat-model-popover')).not.toBeInTheDocument();

  // A /model pick (palette or the inline `/` menu) opens this header's picker.
  modelPickerRequest.request('s-open');

  await expect.element(page.getByTestId('chat-model-popover')).toBeInTheDocument();
  await expect.element(page.getByTestId('chat-model-search')).toBeInTheDocument();
});

test('ignores a model-picker signal aimed at another session', async () => {
  render(ModelPicker, { sessionId: 's-mine', agent: 'smoke' });
  modelPickerRequest.request('s-other');
  // The request never matches, so the popover stays closed (drained in afterEach).
  await expect.element(page.getByTestId('chat-model-popover')).not.toBeInTheDocument();
});

test('groups by provider and shows per-model context, price, and capability badges', async () => {
  render(ModelPicker, { sessionId: 's1', agent: 'smoke' });
  await page.getByTestId('chat-model-trigger').click();

  // Provider header rows name each group.
  await expect.element(page.getByText('openai', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('acp', { exact: true })).toBeInTheDocument();

  // A row carries its context window and input/output price.
  const row = page.getByTestId('chat-model-opt-openai:gpt-5.4');
  await expect.element(row).toHaveTextContent('400k');
  await expect.element(row).toHaveTextContent('$3 / $15');
  // The provider prefix is dropped in the row (the group header names it).
  await expect.element(row).toHaveTextContent('gpt-5.4');

  // Badges track the model's capabilities: gpt-5.4-mini is vision-only.
  const mini = page.getByTestId('chat-model-opt-openai:gpt-5.4-mini');
  await expect.element(mini.getByTitle('vision input')).toBeInTheDocument();
  await expect.element(mini.getByTitle('reasoning')).not.toBeInTheDocument();
  // gpt-5.4 reasons but declares no vision.
  await expect.element(row.getByTitle('reasoning')).toBeInTheDocument();
  await expect.element(row.getByTitle('vision input')).not.toBeInTheDocument();

  // Unpriced models omit price gracefully - covered by formatPrice's unit test;
  // here every model is priced so the row always shows one.
});

test('keyboard navigation crosses provider group boundaries', async () => {
  render(ModelPicker, { sessionId: 's1', agent: 'smoke' });
  await page.getByTestId('chat-model-trigger').click();
  const search = page.getByTestId('chat-model-search');
  (search.element() as HTMLInputElement).focus();
  // Options group as [openai: mini, gpt-5.4][acp: opus]; the current model
  // (acp:claude-opus-4-7) lands the selection in the acp group. One ArrowUp must
  // step into the openai group above it - the flat index walks across the header.
  await userEvent.keyboard('{ArrowUp}{Enter}');
  expect(vi.mocked(api.patch)).toHaveBeenCalledWith('/api/sessions/s1/settings', {
    model: 'openai:gpt-5.4',
  });
});

test('keyboard: arrow + Enter selects a filtered model', async () => {
  render(ModelPicker, { sessionId: 's1', agent: 'smoke' });
  await page.getByTestId('chat-model-trigger').click();
  const search = page.getByTestId('chat-model-search');
  await search.fill('openai');
  (search.element() as HTMLInputElement).focus();
  await userEvent.keyboard('{ArrowDown}{Enter}');
  // First filtered row (gpt-5.4-mini, sorted as returned) is chosen.
  expect(vi.mocked(api.patch)).toHaveBeenCalledWith('/api/sessions/s1/settings', {
    model: 'openai:gpt-5.4',
  });
});
