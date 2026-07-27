/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import CheckItem from './CheckItem.svelte';

test('renders the label and a screen-reader state prefix for a pending item', async () => {
  const { container } = await render(CheckItem, {
    label: 'restore test passes on staging',
    state: 'pending',
  });
  await expect.element(page.getByText('restore test passes on staging')).toBeInTheDocument();
  expect(container.querySelector('.t-check')!.getAttribute('data-st')).toBe('pending');
  expect(container.querySelector('.box')!.textContent).toBe('');
  await expect.element(page.getByText('Pending:', { exact: false })).toBeInTheDocument();
});

test('an active item shows the spinner and is prefixed Verifying', async () => {
  const { container } = await render(CheckItem, {
    label: 'no orphaned blobs in object store',
    state: 'active',
  });
  expect(container.querySelector('.box .t-spin')).not.toBeNull();
  await expect.element(page.getByText('Verifying:', { exact: false })).toBeInTheDocument();
});

test('a pass item shows the check icon and is prefixed Passed', async () => {
  const { container } = await render(CheckItem, {
    label: 'backup completes in under 10 minutes',
    state: 'pass',
  });
  expect(container.querySelector('.box svg')).not.toBeNull();
  await expect.element(page.getByText('Passed:', { exact: false })).toBeInTheDocument();
});

test('a fail item shows the verifier note, colored via the fail state hook', async () => {
  const { container } = await render(CheckItem, {
    label: 'disk usage below 80% after prune',
    state: 'fail',
    note: 'verifier: 84% — prune kept 22 weeklies',
  });
  await expect
    .element(page.getByText('verifier: 84% — prune kept 22 weeklies'))
    .toBeInTheDocument();
  expect(container.querySelector('.note')).not.toBeNull();
  await expect.element(page.getByText('Failed:', { exact: false })).toBeInTheDocument();
});

test('the note is omitted entirely when not given', async () => {
  const { container } = await render(CheckItem, {
    label: 'backup completes in under 10 minutes',
    state: 'pass',
  });
  expect(container.querySelector('.note')).toBeNull();
});
