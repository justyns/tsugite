/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import ScheduleDrawer from './ScheduleDrawer.svelte';
import { TESTID } from '$lib/testids';

test('create mode picks the first agent once the roster loads', async () => {
  const { rerender } = await render(ScheduleDrawer, { open: true, schedule: null, agents: [] });
  await page.getByRole('textbox', { name: 'name' }).fill('nightly-backup');
  await page.getByRole('textbox', { name: 'cadence (cron)' }).fill('0 3 * * *');
  await page.getByLabelText('task prompt required').fill('sweep the logs');

  const create = page.getByTestId(TESTID.scheduleSave);
  await expect.element(create).toBeDisabled();

  await rerender({ agents: ['coder', 'ops'] });
  await expect.element(create).toBeEnabled();
  await expect.element(page.getByRole('textbox', { name: 'name' })).toHaveValue('nightly-backup');
});
