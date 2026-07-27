/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { expect, test, vi } from 'vitest';
import Ask from './Ask.svelte';

test('yes_no mode emits yes / no from the two verbs', async () => {
  const onAnswer = vi.fn();
  render(Ask, {
    question: 'tsugite wants to run',
    command: 'git push origin main',
    questionType: 'yes_no',
    heading: 'Approval required',
    alert: true,
    affirmativeLabel: 'Approve & run',
    negativeLabel: 'Deny',
    onAnswer,
  });

  await page.getByRole('button', { name: 'Approve & run' }).click();
  expect(onAnswer).toHaveBeenLastCalledWith('yes');

  await page.getByRole('button', { name: 'Deny' }).click();
  expect(onAnswer).toHaveBeenLastCalledWith('no');
});

test('the "always" ghost button is absent until a label is supplied', async () => {
  render(Ask, { question: 'run it?', questionType: 'yes_no', onAnswer: vi.fn() });
  await expect.element(page.getByRole('button', { name: /always/i })).not.toBeInTheDocument();
});

test('the "always" ghost button fires its own callback', async () => {
  const onAlways = vi.fn();
  render(Ask, {
    question: 'run it?',
    questionType: 'yes_no',
    alwaysLabel: 'Always allow git push',
    onAlways,
    onAnswer: vi.fn(),
  });
  await page.getByRole('button', { name: 'Always allow git push' }).click();
  expect(onAlways).toHaveBeenCalledOnce();
});

test('choice mode: Send is gated on a selection, then emits the chosen option', async () => {
  const onAnswer = vi.fn();
  render(Ask, {
    question: 'Which embedding model?',
    questionType: 'choice',
    options: ['text-embedding-3-large', 'bge-large-en-v1.5', 'nomic-embed-text'],
    onAnswer,
  });

  const send = page.getByRole('button', { name: 'Send' });
  await expect.element(send).toBeDisabled();

  await page.getByRole('radio', { name: 'bge-large-en-v1.5' }).click();
  await expect.element(send).toBeEnabled();

  await send.click();
  expect(onAnswer).toHaveBeenCalledExactlyOnceWith('bge-large-en-v1.5');
});

test('approval mode: each option button answers with its own exact label', async () => {
  const onAnswer = vi.fn();
  render(Ask, {
    question: 'Fetch content from example.com?',
    questionType: 'approval',
    heading: 'Approval required',
    alert: true,
    options: ['Approve', 'Deny', 'Always allow'],
    onAnswer,
  });

  await page.getByRole('button', { name: 'Approve' }).click();
  expect(onAnswer).toHaveBeenLastCalledWith('Approve');

  await page.getByRole('button', { name: 'Always allow' }).click();
  expect(onAnswer).toHaveBeenLastCalledWith('Always allow');

  await page.getByRole('button', { name: 'Deny' }).click();
  expect(onAnswer).toHaveBeenLastCalledWith('Deny');

  expect(onAnswer).toHaveBeenCalledTimes(3);
});

test('approval mode: options are keyboard-operable (Enter activates the focused button)', async () => {
  const onAnswer = vi.fn();
  render(Ask, {
    question: 'Fetch content from example.com?',
    questionType: 'approval',
    options: ['Approve', 'Deny'],
    onAnswer,
  });

  const deny = page.getByRole('button', { name: 'Deny' });
  (deny.element() as HTMLButtonElement).focus();
  await userEvent.keyboard('{Enter}');
  expect(onAnswer).toHaveBeenCalledExactlyOnceWith('Deny');
});

test('approval mode: Escape does not answer (deny is an explicit choice, not a default)', async () => {
  const onAnswer = vi.fn();
  render(Ask, {
    question: 'Fetch content from example.com?',
    questionType: 'approval',
    options: ['Approve', 'Deny', 'Always allow'],
    onAnswer,
  });

  await userEvent.keyboard('{Escape}');
  expect(onAnswer).not.toHaveBeenCalled();
});

test('text mode: empty is not submittable; typed text emits via Send and Ctrl+Enter', async () => {
  const onAnswer = vi.fn();
  render(Ask, {
    question: 'Name the release',
    questionType: 'text',
    onAnswer,
  });

  const send = page.getByRole('button', { name: 'Send' });
  await expect.element(send).toBeDisabled();

  const box = page.getByRole('textbox');
  await box.fill('Backoff & Chill');
  await expect.element(send).toBeEnabled();
  await send.click();
  expect(onAnswer).toHaveBeenLastCalledWith('Backoff & Chill');

  // Ctrl+Enter is a submit shortcut from within the textarea.
  await box.click();
  await userEvent.keyboard('{Control>}{Enter}{/Control}');
  expect(onAnswer).toHaveBeenCalledTimes(2);
  expect(onAnswer).toHaveBeenLastCalledWith('Backoff & Chill');
});

test('resolved (answered) state is an inert audit trail: no controls, shows the record', async () => {
  const onAnswer = vi.fn();
  render(Ask, {
    question: 'tsugite wants to run',
    command: 'git push origin main',
    questionType: 'yes_no',
    heading: 'Approval required',
    resolution: { tone: 'approved', text: 'approved · ran at 14:32 · exit 0' },
    onAnswer,
  });

  await expect.element(page.getByText('approved · ran at 14:32 · exit 0')).toBeInTheDocument();
  // audit trail must not leave live controls behind
  await expect.element(page.getByRole('button')).not.toBeInTheDocument();
});

test('a11y: the block is a labelled group', async () => {
  render(Ask, {
    question: 'run it?',
    questionType: 'yes_no',
    heading: 'Approval required',
    onAnswer: vi.fn(),
  });
  await expect.element(page.getByRole('group', { name: 'Approval required' })).toBeInTheDocument();
});
