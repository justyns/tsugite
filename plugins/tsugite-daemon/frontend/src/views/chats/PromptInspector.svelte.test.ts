/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import PromptInspector from './PromptInspector.svelte';

afterEach(async () => {
  await page.viewport(1440, 900);
});

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

test('in a narrow clipping pane the popover stays inside that pane', async () => {
  // A mux pane clips its overflow and can be as narrow as 260px. The meter sits
  // near the pane's right edge, with less room than the popover on either side.
  await page.viewport(1280, 800);
  const { container } = await render(PromptInspector, {
    ...base,
    breakdown: { categories: [{ name: 'tools', tokens: 5000, items: [] }], total: 5000 },
  });
  container.style.cssText =
    'box-sizing:border-box;position:fixed;top:0;left:200px;width:340px;overflow:hidden;display:flex;justify-content:flex-end;padding-right:44px;';

  // The trigger is pinned into the fixture's fixed-position pane, out of reach of
  // the runner's synthetic pointer, so fire the DOM click on the raw element.
  (page.getByRole('button', { name: /context breakdown/i }).element() as HTMLElement).click();
  await expect.element(page.getByRole('dialog')).toBeInTheDocument();

  const pane = container.getBoundingClientRect();
  const r = page.getByRole('dialog').element().getBoundingClientRect();
  expect(r.left).toBeGreaterThanOrEqual(pane.left);
  expect(r.right).toBeLessThanOrEqual(pane.right);
});
