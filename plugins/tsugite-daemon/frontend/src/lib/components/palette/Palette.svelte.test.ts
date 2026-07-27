/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import Palette from './Palette.svelte';
import type { PaletteItem } from './palette-match';

const items: PaletteItem[] = [
  { group: 'sessions', icon: 'chat', label: 'refactor sse reconnect', meta: 'code' },
  { group: 'sessions', icon: 'chat', label: 'ops backup prune', meta: 'ops' },
  { group: 'jobs', icon: 'jobs', label: 'fix flaky sse test', meta: 'running' },
];

test('opens focused with the first result active and groups shown', async () => {
  render(Palette, { open: true, items });
  const input = page.getByRole('combobox');
  await expect.element(input).toBeInTheDocument();
  const options = page.getByRole('option');
  await expect.element(options.first()).toHaveAttribute('aria-selected', 'true');
  // group headers render when there is no query
  await expect.element(page.getByText('sessions')).toBeInTheDocument();
});

test('the search field is type=search and opts out of password managers', async () => {
  render(Palette, { open: true, items });
  const input = page.getByRole('combobox');
  // type=search excludes the field from Chromium's built-in credential
  // heuristics; autocomplete=off (from pwmIgnore) handles the extensions.
  await expect.element(input).toHaveAttribute('type', 'search');
  await expect.element(input).toHaveAttribute('autocomplete', 'off');
});

test('ArrowDown / ArrowUp move the active option', async () => {
  render(Palette, { open: true, items });
  await page.getByRole('combobox').click();
  await userEvent.keyboard('{ArrowDown}');
  await expect.element(page.getByRole('option').nth(1)).toHaveAttribute('aria-selected', 'true');
  await userEvent.keyboard('{ArrowUp}');
  await expect.element(page.getByRole('option').first()).toHaveAttribute('aria-selected', 'true');
});

test('Ctrl+J / Ctrl+K also move the active option', async () => {
  render(Palette, { open: true, items });
  await page.getByRole('combobox').click();
  await userEvent.keyboard('{Control>}j{/Control}');
  await expect.element(page.getByRole('option').nth(1)).toHaveAttribute('aria-selected', 'true');
  await userEvent.keyboard('{Control>}k{/Control}');
  await expect.element(page.getByRole('option').first()).toHaveAttribute('aria-selected', 'true');
});

test('Enter selects the active item and closes', async () => {
  const onSelect = vi.fn();
  render(Palette, { open: true, items, onSelect });
  await page.getByRole('combobox').click();
  await userEvent.keyboard('{ArrowDown}{Enter}');
  expect(onSelect).toHaveBeenCalledTimes(1);
  expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ label: 'ops backup prune' }));
  await expect.element(page.getByRole('combobox')).not.toBeInTheDocument();
});

test('Escape closes without selecting', async () => {
  const onSelect = vi.fn();
  render(Palette, { open: true, items, onSelect });
  await page.getByRole('combobox').click();
  await userEvent.keyboard('{Escape}');
  await expect.element(page.getByRole('combobox')).not.toBeInTheDocument();
  expect(onSelect).not.toHaveBeenCalled();
});

test('typing filters to matches and highlights the run', async () => {
  render(Palette, { open: true, items });
  const input = page.getByRole('combobox');
  await input.click();
  await userEvent.type(input, 'sse');
  const options = page.getByRole('option');
  await expect.element(options.nth(1)).toBeInTheDocument();
  await expect.element(options.nth(2)).not.toBeInTheDocument(); // 'ops backup prune' dropped
  // both matches highlight their 'sse' run in a <b>
  await expect.element(page.getByText('sse', { exact: true }).first()).toBeInTheDocument();
});

test('no matches shows the empty hint', async () => {
  render(Palette, { open: true, items });
  await page.getByRole('combobox').click();
  await userEvent.type(page.getByRole('combobox'), 'zzzzz');
  await expect.element(page.getByText(/No matches for/)).toBeInTheDocument();
});

test('clicking a row selects it', async () => {
  const onSelect = vi.fn();
  render(Palette, { open: true, items, onSelect });
  await page.getByText('fix flaky sse test').click();
  expect(onSelect).toHaveBeenCalledTimes(1);
  expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ label: 'fix flaky sse test' }));
});

test('the footer carries no leaked mobile-sheet dev note', async () => {
  render(Palette, { open: true, items });
  await expect.element(page.getByRole('combobox')).toBeInTheDocument();
  await expect.element(page.getByText(/full-screen sheet on phones/)).not.toBeInTheDocument();
});

const sessionItems: PaletteItem[] = [
  {
    group: 'sessions',
    icon: 'chat',
    label: 'weekly planning sync',
    meta: '2m',
    href: 'session:s1',
  },
  { group: 'sessions', icon: 'chat', label: 'nightly ops backup', meta: '1h', href: 'session:s2' },
];

test('typing a session title reveals the sessions group and its row', async () => {
  render(Palette, { open: true, items, sessionItems });
  const input = page.getByRole('combobox');
  await input.click();
  await userEvent.type(input, 'planning');
  await expect.element(page.getByText('sessions')).toBeInTheDocument(); // group header
  await expect.element(page.getByText('weekly planning sync')).toBeInTheDocument();
});

test('picking a session row fires onSelect with its session href', async () => {
  const onSelect = vi.fn();
  render(Palette, { open: true, items, sessionItems, onSelect });
  const input = page.getByRole('combobox');
  await input.click();
  await userEvent.type(input, 'planning');
  await page.getByText('weekly planning sync').click();
  expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ href: 'session:s1' }));
});

test('a query matching no session shows no sessions group', async () => {
  render(Palette, { open: true, items: [], sessionItems });
  const input = page.getByRole('combobox');
  await input.click();
  await userEvent.type(input, 'zzzzz');
  await expect.element(page.getByText(/No matches for/)).toBeInTheDocument();
  await expect.element(page.getByText('sessions')).not.toBeInTheDocument();
});

test('renders live sessions above ended ones for the same match', async () => {
  // Order mirrors buildSessionItems' live-first output; ties keep source order.
  const ranked: PaletteItem[] = [
    {
      group: 'sessions',
      icon: 'chat',
      label: 'sse live triage',
      meta: 'now',
      href: 'session:live',
    },
    {
      group: 'sessions',
      icon: 'chat',
      label: 'sse old thread',
      meta: 'jul 1',
      href: 'session:old',
    },
  ];
  render(Palette, { open: true, items: [], sessionItems: ranked });
  const input = page.getByRole('combobox');
  await input.click();
  await userEvent.type(input, 'sse');
  await expect.element(page.getByRole('option').first()).toHaveTextContent('sse live triage');
});
