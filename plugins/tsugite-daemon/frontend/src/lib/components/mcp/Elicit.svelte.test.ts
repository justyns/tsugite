/// <reference types="@vitest/browser/context" />
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import Elicit from './Elicit.svelte';

function fields() {
  return [
    {
      kind: 'enum' as const,
      name: 'environment',
      label: 'environment',
      required: true,
      description: 'enum',
      value: 'staging',
      options: [
        { value: 'staging', label: 'staging', hint: '— auto-verifies' },
        { value: 'production', label: 'production', hint: '— needs approval' },
      ],
    },
    {
      kind: 'string' as const,
      name: 'version',
      label: 'version',
      required: true,
      description: 'string',
      value: 'refactor/sse-backoff',
    },
    {
      kind: 'boolean' as const,
      name: 'runSmokeTests',
      label: 'run smoke tests',
      description: 'boolean',
      value: true,
      hint: 'gate promote on the smoke suite',
    },
  ];
}

test('Submit returns the schema defaults as content', async () => {
  const onSubmit = vi.fn();
  const screen = await render(Elicit, {
    source: 'deploy-server',
    message: 'Confirm the deploy parameters.',
    fields: fields(),
    onSubmit,
  });
  await screen.getByRole('button', { name: 'Submit' }).click();
  expect(onSubmit).toHaveBeenCalledWith({
    environment: 'staging',
    version: 'refactor/sse-backoff',
    runSmokeTests: true,
  });
});

test('edits to fields flow into the submitted content', async () => {
  const onSubmit = vi.fn();
  const screen = await render(Elicit, {
    source: 'deploy-server',
    message: 'Confirm.',
    fields: fields(),
    onSubmit,
  });
  await screen.getByRole('radio', { name: /production/i }).click();
  await screen.getByRole('textbox', { name: /version/i }).fill('main');
  await screen.getByRole('switch', { name: /run smoke tests/i }).click();
  await screen.getByRole('button', { name: 'Submit' }).click();
  expect(onSubmit).toHaveBeenLastCalledWith({
    environment: 'production',
    version: 'main',
    runSmokeTests: false,
  });
});

test('Decline and Cancel fire their callbacks, not onSubmit', async () => {
  const onSubmit = vi.fn();
  const onDecline = vi.fn();
  const onCancel = vi.fn();
  const screen = await render(Elicit, {
    source: 'deploy-server',
    message: 'Confirm.',
    fields: fields(),
    onSubmit,
    onDecline,
    onCancel,
  });
  await screen.getByRole('button', { name: 'Decline' }).click();
  await screen.getByRole('button', { name: 'Cancel' }).click();
  expect(onDecline).toHaveBeenCalledTimes(1);
  expect(onCancel).toHaveBeenCalledTimes(1);
  expect(onSubmit).not.toHaveBeenCalled();
});

test('resolved state hides the form and shows the audit-trail row', async () => {
  const screen = await render(Elicit, {
    source: 'deploy-server',
    message: 'Confirm.',
    fields: fields(),
    state: 'submitted',
  });
  await expect
    .element(screen.getByText('submitted · action: accept · content returned'))
    .toBeVisible();
  // Resolved: the form + action buttons are removed from the a11y tree.
  expect(screen.getByRole('button', { name: 'Submit' }).elements()).toHaveLength(0);
  expect(screen.getByRole('textbox', { name: /version/i }).elements()).toHaveLength(0);
});
