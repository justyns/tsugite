/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import PromptInspector from './PromptInspector.svelte';

const base = {
  value: 8000,
  max: 200000,
  label: 'Context 8k of 200k tokens',
  displayText: '8k/200k',
  warn: false,
};

test('with no breakdown it is a plain, non-interactive meter', async () => {
  render(PromptInspector, { ...base, breakdown: null });
  await expect.element(page.getByText('8k/200k')).toBeInTheDocument();
  // No snapshot -> nothing to click open.
  expect(page.getByRole('button').query()).toBeNull();
});

test('opens a popover listing non-zero categories and the total', async () => {
  render(PromptInspector, {
    ...base,
    breakdown: {
      categories: [
        { name: 'history', tokens: 3000, items: [] },
        { name: 'tools', tokens: 5000, items: [{ name: 'read_file', tokens: 1 }] },
        { name: 'skills', tokens: 0, items: [] },
      ],
      total: 8000,
    },
  });
  await page.getByRole('button', { name: /context breakdown/i }).click();
  await expect
    .element(page.getByRole('dialog', { name: /context breakdown/i }))
    .toBeInTheDocument();
  await expect.element(page.getByText('tools', { exact: true })).toBeInTheDocument();
  await expect.element(page.getByText('history', { exact: true })).toBeInTheDocument();
  // Zero-token categories are omitted.
  expect(page.getByText('skills', { exact: true }).query()).toBeNull();
  // Total surfaced.
  await expect.element(page.getByText('8k', { exact: true })).toBeInTheDocument();
});

test('shows breakdown staleness (turn + relative time) so it is not read as current', async () => {
  const twoMinAgo = new Date(Date.now() - 2 * 60 * 1000).toISOString();
  render(PromptInspector, {
    ...base,
    breakdown: { categories: [{ name: 'tools', tokens: 5000, items: [] }], total: 5000 },
    turn: 4,
    at: twoMinAgo,
  });
  await page.getByRole('button', { name: /context breakdown/i }).click();
  // turn is 0-indexed in the log; shown 1-indexed to match the turn bubbles.
  await expect.element(page.getByText(/as of turn 5/i)).toBeInTheDocument();
  await expect.element(page.getByText(/2m ago/i)).toBeInTheDocument();
});

test('the "view raw messages" footer button shows only with onViewRaw and fires it', async () => {
  const onViewRaw = vi.fn();
  render(PromptInspector, {
    ...base,
    breakdown: { categories: [{ name: 'tools', tokens: 5000, items: [] }], total: 5000 },
    onViewRaw,
  });
  await page.getByRole('button', { name: /context breakdown/i }).click();
  const btn = page.getByRole('button', { name: /view raw messages/i });
  await expect.element(btn).toBeInTheDocument();
  await btn.click();
  expect(onViewRaw).toHaveBeenCalledOnce();
});

test('without onViewRaw the popover carries no raw-messages affordance', async () => {
  render(PromptInspector, {
    ...base,
    breakdown: { categories: [{ name: 'tools', tokens: 5000, items: [] }], total: 5000 },
  });
  await page.getByRole('button', { name: /context breakdown/i }).click();
  await expect.element(page.getByRole('dialog')).toBeInTheDocument();
  expect(page.getByRole('button', { name: /view raw messages/i }).query()).toBeNull();
});

test('the popover closes on an outside mousedown', async () => {
  render(PromptInspector, {
    ...base,
    breakdown: { categories: [{ name: 'tools', tokens: 5000, items: [] }], total: 5000 },
  });
  await page.getByRole('button', { name: /context breakdown/i }).click();
  await expect.element(page.getByRole('dialog')).toBeInTheDocument();
  document.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
  await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
});
