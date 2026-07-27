/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import SpacesRow from './SpacesRow.svelte';

test('renders the state word, title, who line, and meter readout', async () => {
  render(SpacesRow, {
    title: 'refactor: sse reconnect backoff',
    who: 'odyn · sonnet-4.6',
    state: 'working',
    contextPct: 3,
    contextTokens: '34k',
  });
  await expect.element(page.getByText('working')).toBeInTheDocument();
  await expect.element(page.getByText('refactor: sse reconnect backoff')).toBeInTheDocument();
  await expect.element(page.getByText('odyn · sonnet-4.6')).toBeInTheDocument();
  await expect.element(page.getByText('3% · 34k')).toBeInTheDocument();
});

test('exposes the context meter with meter role semantics', async () => {
  render(SpacesRow, {
    title: 'nightly backup prune policy',
    who: 'ops-runner · waiting 12m',
    state: 'blocked',
    contextPct: 41,
    contextTokens: '82k',
  });
  const meter = page.getByRole('meter');
  await expect.element(meter).toHaveAttribute('aria-valuenow', '41');
  await expect.element(meter).toHaveAttribute('aria-valuemin', '0');
  await expect.element(meter).toHaveAttribute('aria-valuemax', '100');
});

test('the bar width is clamped into [0,100] even with an out-of-range pct', async () => {
  const { container } = await render(SpacesRow, {
    title: 'add jitter to backoff',
    who: 'code-worker · 5/5 criteria',
    state: 'done',
    contextPct: 140,
    contextTokens: '18k',
  });
  const bar = container.querySelector('.bar i') as HTMLElement;
  expect(bar.style.getPropertyValue('--w')).toBe('100%');
});

test('contextWarn is opt-in and off by default', async () => {
  const { container, rerender } = await render(SpacesRow, {
    title: 'research: local whisper models',
    who: 'odyn · opus-4.6',
    state: 'idle',
    contextPct: 12,
    contextTokens: '24k',
  });
  expect(container.querySelector('.t-meter')!.className).not.toContain('is-warn');

  await rerender({
    title: 'research: local whisper models',
    who: 'odyn · opus-4.6',
    state: 'idle',
    contextPct: 12,
    contextTokens: '24k',
    contextWarn: true,
  });
  expect(container.querySelector('.t-meter')!.className).toContain('is-warn');
});

test('data-st drives the blocked warm-edge styling hook', async () => {
  const { container } = await render(SpacesRow, {
    title: 'nightly backup prune policy',
    who: 'ops-runner · waiting 12m',
    state: 'blocked',
    contextPct: 41,
    contextTokens: '82k',
  });
  expect(container.querySelector('.sp-row')!.getAttribute('data-st')).toBe('blocked');
});

test('clicking the row fires onSelect', async () => {
  const onSelect = vi.fn();
  render(SpacesRow, {
    title: 'refactor: sse reconnect backoff',
    who: 'odyn · sonnet-4.6',
    state: 'working',
    contextPct: 3,
    contextTokens: '34k',
    onSelect,
  });
  await userEvent.click(page.getByRole('button'));
  expect(onSelect).toHaveBeenCalledOnce();
});

test('Enter activates the focused row from the keyboard', async () => {
  const onSelect = vi.fn();
  render(SpacesRow, {
    title: 'refactor: sse reconnect backoff',
    who: 'odyn · sonnet-4.6',
    state: 'working',
    contextPct: 3,
    contextTokens: '34k',
    onSelect,
  });
  await userEvent.click(page.getByRole('button'));
  onSelect.mockClear();
  await userEvent.keyboard('{Enter}');
  expect(onSelect).toHaveBeenCalledOnce();
});

test('isActive adds the is-active class', async () => {
  const { container } = await render(SpacesRow, {
    title: 'refactor: sse reconnect backoff',
    who: 'odyn · sonnet-4.6',
    state: 'working',
    contextPct: 3,
    contextTokens: '34k',
    isActive: true,
  });
  expect(container.querySelector('.sp-row')!.className).toContain('is-active');
});
