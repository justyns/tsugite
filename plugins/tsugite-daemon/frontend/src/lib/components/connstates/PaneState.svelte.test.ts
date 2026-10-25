/// <reference types="vitest/browser" />
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import PaneState from './PaneState.svelte';

test('loading pane is announced as "Loading" by default', async () => {
  const { container } = await render(PaneState, { kind: 'loading' });
  expect(container.querySelector('[aria-busy="true"]')?.getAttribute('aria-label')).toBe('Loading');
});

test('loading pane announces the label it is given', async () => {
  const { container } = await render(PaneState, { kind: 'loading', label: 'Loading jobs' });
  expect(container.querySelector('[aria-busy="true"]')?.getAttribute('aria-label')).toBe(
    'Loading jobs',
  );
});
