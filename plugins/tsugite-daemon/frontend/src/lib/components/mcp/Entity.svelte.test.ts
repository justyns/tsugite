/// <reference types="@vitest/browser/context" />
import { userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import Entity from './Entity.svelte';

const props = {
  entityKey: 'JIRA-1234',
  statusLabel: 'In Progress',
  status: 'working' as const,
  title: 'SSE client never recovers after laptop sleep',
  assignee: 'you',
  priority: 'High',
  sprint: 'Sprint 24',
  via: 'Jira',
};

test('clicking the chiplet opens the detail popover', async () => {
  const onOpen = vi.fn();
  const screen = await render(Entity, { ...props, onOpen });
  const chip = screen.getByRole('button', { name: /JIRA-1234/ });
  await expect.element(chip).toHaveAttribute('aria-expanded', 'false');
  await chip.click();
  await expect.element(screen.getByRole('dialog')).toBeVisible();
  await expect.element(chip).toHaveAttribute('aria-expanded', 'true');
  expect(onOpen).toHaveBeenCalledTimes(1);
});

test('Escape closes the popover and restores focus to the chiplet', async () => {
  const screen = await render(Entity, props);
  const chip = screen.getByRole('button', { name: /JIRA-1234/ });
  await chip.click();
  await expect.element(screen.getByRole('dialog')).toBeInTheDocument();
  await userEvent.keyboard('{Escape}');
  await expect.element(screen.getByRole('dialog')).not.toBeInTheDocument();
  await expect.element(chip).toHaveFocus();
});

test('the popover surfaces the entity detail fields', async () => {
  const screen = await render(Entity, { ...props, open: true });
  await expect
    .element(screen.getByText('SSE client never recovers after laptop sleep'))
    .toBeVisible();
  await expect.element(screen.getByText('you')).toBeVisible();
  await expect.element(screen.getByText('Sprint 24')).toBeVisible();
});

test('status is exposed as a data attribute (dot + text, not colour alone)', async () => {
  const screen = await render(Entity, { ...props, status: 'blocked' });
  await expect
    .element(screen.getByRole('button', { name: /JIRA-1234/ }))
    .toHaveAttribute('data-st', 'blocked');
});

test('detail popover: status is a t-pill badge and provenance is a plug icon', async () => {
  const screen = await render(Entity, { ...props, status: 'working', open: true });
  await expect.element(screen.getByRole('dialog')).toBeInTheDocument();
  const dialog = screen.getByRole('dialog').element();

  // Header status: bordered/filled pill (colour + text, never colour alone),
  // mapped to the pill's data-st vocabulary (working -> running).
  const pill = dialog.querySelector('.eph .t-pill')!;
  expect(pill).not.toBeNull();
  expect(pill.getAttribute('data-st')).toBe('running');
  expect(pill.textContent).toContain('In Progress');
  expect(dialog.querySelector('.eph .dot')).toBeNull(); // no downgraded bare dot

  // Footer provenance: plug icon + full "auto-linked by {via} plugin" copy,
  // carrying the plugin-provenance cue instead of a generic dot.
  const evia = dialog.querySelector('.evia')!;
  expect(evia.querySelector('svg.ic')).not.toBeNull();
  expect(evia.querySelector('.dot')).toBeNull();
  expect(evia.textContent).toContain('auto-linked by Jira plugin');
});
