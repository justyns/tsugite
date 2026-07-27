/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import ArtifactPanel from './ArtifactPanel.svelte';

test('mode switch flips the active view and reports it', async () => {
  const onViewChange = vi.fn();
  render(ArtifactPanel, {
    props: { title: 'Backup retention plan', kind: 'plan', onViewChange },
  });

  const rendered = page.getByRole('button', { name: 'rendered' });
  const raw = page.getByRole('button', { name: 'raw' });
  const body = page.getByRole('region', { name: /content/ });

  // default view is rendered
  await expect.element(rendered).toHaveAttribute('aria-pressed', 'true');
  await expect.element(body).toHaveAttribute('data-view', 'rendered');

  await raw.click();

  await expect.element(raw).toHaveAttribute('aria-pressed', 'true');
  await expect.element(rendered).toHaveAttribute('aria-pressed', 'false');
  await expect.element(body).toHaveAttribute('data-view', 'raw');
  expect(onViewChange).toHaveBeenCalledWith('raw');
});

test('arrow keys move focus and selection across the mode switch, wrapping', async () => {
  render(ArtifactPanel, { props: { title: 'Backup retention plan', kind: 'plan' } });

  const rendered = page.getByRole('button', { name: 'rendered' });
  const diff = page.getByRole('button', { name: 'diff' });
  const json = page.getByRole('button', { name: 'json' });
  const body = page.getByRole('region', { name: /content/ });

  await rendered.element().focus();

  await userEvent.keyboard('{ArrowRight}');
  await expect.element(diff).toHaveFocus();
  await expect.element(diff).toHaveAttribute('aria-pressed', 'true');
  await expect.element(body).toHaveAttribute('data-view', 'diff');

  // ArrowLeft off the first segment wraps to the last one
  await userEvent.keyboard('{ArrowLeft}{ArrowLeft}');
  await expect.element(json).toHaveFocus();
  await expect.element(json).toHaveAttribute('aria-pressed', 'true');
  await expect.element(body).toHaveAttribute('data-view', 'json');

  await userEvent.keyboard('{Home}');
  await expect.element(rendered).toHaveAttribute('aria-pressed', 'true');
  await userEvent.keyboard('{End}');
  await expect.element(json).toHaveAttribute('aria-pressed', 'true');
});

test('only the active segment is a tab stop (roving tabindex)', async () => {
  const { container } = await render(ArtifactPanel, {
    props: { title: 'Backup retention plan', kind: 'plan', view: 'diff' },
  });
  const buttons = [...container.querySelectorAll<HTMLButtonElement>('[role="group"] button')];
  expect(buttons.map((b) => b.tabIndex)).toEqual([-1, 0, -1, -1]);
});

test('launch variant opens on click', async () => {
  const onOpen = vi.fn();
  render(ArtifactPanel, {
    props: {
      variant: 'launch',
      title: 'Backup retention plan',
      subtitle: 'plan · markdown',
      openLabel: 'Review',
      onOpen,
    },
  });

  await expect.element(page.getByText('plan · markdown')).toBeInTheDocument();
  await page.getByRole('button', { name: 'Review' }).click();
  expect(onOpen).toHaveBeenCalledOnce();
});
