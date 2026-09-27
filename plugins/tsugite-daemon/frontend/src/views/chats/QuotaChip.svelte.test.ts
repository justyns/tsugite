/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { usage, type UsageProviderRow } from '$lib/stores/usage.svelte';
import { TESTID } from '$lib/testids';
import QuotaChip from './QuotaChip.svelte';

const CLAUDE: UsageProviderRow = {
  provider: 'claude_code',
  label: 'Claude Code',
  error: null,
  windows: [
    { key: 'five_hour', label: '5h', used_pct: 11, resets_at: '2026-07-19T21:59:00Z' },
    { key: 'weekly_all', label: 'week', used_pct: 40.25, resets_at: null },
  ],
};
const CODEX: UsageProviderRow = {
  provider: 'codex_cli',
  label: 'Codex',
  error: null,
  windows: [{ key: 'primary', label: '5h', used_pct: 90, resets_at: null }],
};

let refresh: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  refresh = vi.spyOn(usage, 'refreshProviders').mockResolvedValue();
  usage.providers = [CODEX, CLAUDE];
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const chip = () => page.getByTestId(TESTID.chatQuota);

test('shows the most-used window of the chat provider', async () => {
  render(QuotaChip, { provider: 'claude_code' });
  await expect.element(chip()).toHaveTextContent('week 40.3%');
  await expect.element(chip()).not.toHaveClass('is-warn');
});

test('renders nothing when the provider reports no quota', async () => {
  render(QuotaChip, { provider: 'openai' });
  await vi.waitFor(() => expect(refresh).toHaveBeenCalled());
  expect(chip().query()).toBeNull();
});

test('warns at 80% used', async () => {
  usage.providers = [{ ...CLAUDE, windows: [{ ...CLAUDE.windows[0]!, used_pct: 80 }] }];
  render(QuotaChip, { provider: 'claude_code' });
  await expect.element(chip()).toHaveClass('is-warn');
});

test('an errored report shows an error state and its message', async () => {
  usage.providers = [{ ...CLAUDE, windows: [], error: 'usage limits could not be fetched' }];
  render(QuotaChip, { provider: 'claude_code' });
  await expect.element(chip()).toHaveClass('is-err');
  await chip().click();
  await expect
    .element(page.getByTestId(TESTID.chatQuotaPopover))
    .toHaveTextContent('usage limits could not be fetched');
});

test('the popover lists every window and links to the Usage view', async () => {
  render(QuotaChip, { provider: 'claude_code' });
  await chip().click();
  const pop = page.getByTestId(TESTID.chatQuotaPopover);
  await expect.element(pop.getByRole('meter', { name: 'Claude Code 5h' })).toBeInTheDocument();
  await expect.element(pop.getByRole('meter', { name: 'Claude Code week' })).toBeInTheDocument();
  await expect.element(pop).toHaveTextContent('resets jul 19 21:59 UTC');
  await expect.element(pop.getByRole('link')).toHaveAttribute('href', '#usage');

  await userEvent.keyboard('{Escape}');
  await expect.element(pop).not.toBeInTheDocument();
  (chip().element() as HTMLElement).focus();
  await userEvent.keyboard('{Enter}');
  await expect.element(pop).toBeInTheDocument();
});

test('refreshes when the provider resolves and again when a turn finishes', async () => {
  const { rerender } = await render(QuotaChip, { provider: null, working: false });
  expect(refresh).not.toHaveBeenCalled();
  await rerender({ provider: 'claude_code' });
  await vi.waitFor(() => expect(refresh).toHaveBeenCalledTimes(1));
  await rerender({ working: true });
  expect(refresh).toHaveBeenCalledTimes(1);
  await rerender({ working: false });
  await vi.waitFor(() => expect(refresh).toHaveBeenCalledTimes(2));
});
