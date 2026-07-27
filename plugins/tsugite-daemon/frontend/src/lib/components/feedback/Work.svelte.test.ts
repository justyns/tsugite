/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import Work from './Work.svelte';

test('names the operation and shows elapsed time computed from startedAt', async () => {
  const { container } = await render(Work, {
    operation: 'npm test -w @tsugite/sse',
    startedAt: Date.now() - 7000,
    onStop: () => {},
  });
  await expect.element(page.getByText('npm test -w @tsugite/sse')).toBeInTheDocument();
  expect(container.querySelector('.el')?.textContent).toBe('00:07');
});

test('defaults the verb to "running" but accepts an override', async () => {
  const { container: c1 } = await render(Work, {
    operation: 'thing',
    startedAt: Date.now(),
    onStop: () => {},
  });
  expect(c1.querySelector('.t-work')?.textContent).toContain('running');

  const { container: c2 } = await render(Work, {
    operation: 'thing',
    verb: 'waiting on',
    startedAt: Date.now(),
    onStop: () => {},
  });
  expect(c2.querySelector('.t-work')?.textContent).toContain('waiting on');
});

test('renders the progress-detail extension only when provided', async () => {
  const { container } = await render(Work, {
    operation: 'bash',
    detail: 'turn 3 · 2 tools · tool: bash',
    startedAt: Date.now(),
    onStop: () => {},
  });
  expect(container.querySelector('.wk-detail')?.textContent).toBe(
    ' · turn 3 · 2 tools · tool: bash',
  );
});

test('omits the detail span entirely with no detail prop', async () => {
  const { container } = await render(Work, {
    operation: 'bash',
    startedAt: Date.now(),
    onStop: () => {},
  });
  expect(container.querySelector('.wk-detail')).toBeNull();
});

test('clicking Stop calls onStop', async () => {
  const onStop = vi.fn();
  await render(Work, { operation: 'bash', startedAt: Date.now(), onStop });
  await userEvent.click(page.getByRole('button', { name: 'Stop' }));
  expect(onStop).toHaveBeenCalledOnce();
});

test('reconnecting flips the container to .is-re and the spinner to the warn color', async () => {
  const { container } = await render(Work, {
    operation: 'bash',
    startedAt: Date.now(),
    reconnecting: true,
    onStop: () => {},
  });
  expect(container.querySelector('.t-work')?.classList.contains('is-re')).toBe(true);
  expect(container.querySelector('.t-spin')?.getAttribute('style')).toBe(
    '--spin-c: var(--st-warn);',
  );
});

test('reconnecting is announced as text in a status region, not signaled by color alone', async () => {
  const { container } = await render(Work, {
    operation: 'bash',
    startedAt: Date.now(),
    reconnecting: true,
    onStop: () => {},
  });
  const flag = container.querySelector('.re-flag');
  expect(flag).not.toBeNull();
  expect(flag?.getAttribute('role')).toBe('status');
  expect(flag?.textContent?.toLowerCase()).toContain('reconnecting');
});

test('the reconnecting flag is absent while running normally', async () => {
  const { container } = await render(Work, {
    operation: 'bash',
    startedAt: Date.now(),
    onStop: () => {},
  });
  expect(container.querySelector('.re-flag')).toBeNull();
});

test('running and reconnecting render different text, not just different color', async () => {
  const startedAt = Date.now();
  const { container: running } = await render(Work, {
    operation: 'npm test -w @tsugite/sse',
    startedAt,
    onStop: () => {},
  });
  const { container: reconnecting } = await render(Work, {
    operation: 'npm test -w @tsugite/sse',
    startedAt,
    reconnecting: true,
    onStop: () => {},
  });
  expect(reconnecting.querySelector('.t-work')?.textContent).not.toBe(
    running.querySelector('.t-work')?.textContent,
  );
});

test('the ok-state spinner is colored via a prop, since Spin is a child component CSS cannot reach', async () => {
  const { container } = await render(Work, {
    operation: 'bash',
    startedAt: Date.now(),
    onStop: () => {},
  });
  expect(container.querySelector('.t-spin')?.getAttribute('style')).toBe('--spin-c: var(--st-ok);');
});

test('the elapsed readout ticks upward on its own timer', async () => {
  vi.useFakeTimers();
  try {
    const startedAt = Date.now();
    const { container } = await render(Work, { operation: 'bash', startedAt, onStop: () => {} });
    expect(container.querySelector('.el')?.textContent).toBe('00:00');
    await vi.advanceTimersByTimeAsync(3000);
    expect(container.querySelector('.el')?.textContent).toBe('00:03');
  } finally {
    vi.useRealTimers();
  }
});
