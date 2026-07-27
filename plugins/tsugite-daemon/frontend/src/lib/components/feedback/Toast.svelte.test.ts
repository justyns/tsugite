/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import Toast from './Toast.svelte';
import { AUTO_DISMISS_MS, EXIT_DURATION_MS, RESUME_DISMISS_MS } from './toast-store.svelte';

test('renders title, body, and the variant on data-v', async () => {
  const { container } = await render(Toast, {
    variant: 'warn',
    title: 'Job needs an answer',
    body: 'nightly-backup prune is blocked on a retention question.',
    onDismiss: () => {},
  });
  expect(container.querySelector('.t-toast')?.getAttribute('data-v')).toBe('warn');
  await expect.element(page.getByText('Job needs an answer')).toBeInTheDocument();
  await expect
    .element(page.getByText('nightly-backup prune is blocked on a retention question.'))
    .toBeInTheDocument();
});

test('an action button renders only when both actionLabel and onAction are given', async () => {
  const onAction = vi.fn();
  await render(Toast, {
    variant: 'warn',
    title: 'Job needs an answer',
    actionLabel: 'Answer',
    onAction,
    onDismiss: () => {},
  });
  await userEvent.click(page.getByRole('button', { name: 'Answer' }));
  expect(onAction).toHaveBeenCalledOnce();
});

test('omitting the action callback omits the action button', async () => {
  await render(Toast, {
    variant: 'ok',
    title: 'Job done',
    actionLabel: 'Answer',
    onDismiss: () => {},
  });
  await expect.element(page.getByRole('button', { name: 'Answer' })).not.toBeInTheDocument();
});

test('clicking Dismiss fades out then calls onDismiss after the exit transition', async () => {
  vi.useFakeTimers();
  try {
    const onDismiss = vi.fn();
    const { container } = await render(Toast, {
      variant: 'ok',
      title: 'Job done',
      sticky: true,
      onDismiss,
    });
    await userEvent.click(page.getByRole('button', { name: 'Dismiss' }));
    expect(container.querySelector('.t-toast')?.classList.contains('is-out')).toBe(true);
    expect(onDismiss).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(EXIT_DURATION_MS);
    expect(onDismiss).toHaveBeenCalledOnce();
  } finally {
    vi.useRealTimers();
  }
});

test('a non-sticky toast auto-dismisses on its own after the 6s window', async () => {
  vi.useFakeTimers();
  try {
    const onDismiss = vi.fn();
    await render(Toast, { variant: 'info', title: 'Compaction complete', onDismiss });
    await vi.advanceTimersByTimeAsync(AUTO_DISMISS_MS + EXIT_DURATION_MS);
    expect(onDismiss).toHaveBeenCalledOnce();
  } finally {
    vi.useRealTimers();
  }
});

test('err toasts never auto-dismiss, even without an explicit sticky prop', async () => {
  vi.useFakeTimers();
  try {
    const onDismiss = vi.fn();
    await render(Toast, { variant: 'err', title: 'Schedule failed', onDismiss });
    await vi.advanceTimersByTimeAsync(AUTO_DISMISS_MS * 5);
    expect(onDismiss).not.toHaveBeenCalled();
  } finally {
    vi.useRealTimers();
  }
});

test('hovering pauses the auto-dismiss timer; leaving resumes it at the shorter delay', async () => {
  vi.useFakeTimers();
  try {
    const onDismiss = vi.fn();
    const { container } = await render(Toast, { variant: 'ok', title: 'Job done', onDismiss });
    const toastLocator = page.elementLocator(container.querySelector('.t-toast')!);

    await toastLocator.hover();
    await vi.advanceTimersByTimeAsync(AUTO_DISMISS_MS * 3);
    expect(onDismiss).not.toHaveBeenCalled();

    await toastLocator.unhover();
    await vi.advanceTimersByTimeAsync(RESUME_DISMISS_MS + EXIT_DURATION_MS);
    expect(onDismiss).toHaveBeenCalledOnce();
  } finally {
    vi.useRealTimers();
  }
});
