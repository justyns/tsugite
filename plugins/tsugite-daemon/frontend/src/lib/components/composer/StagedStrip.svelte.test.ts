/// <reference types="vitest/browser" />
import { page } from 'vitest/browser';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';

// A staged image thumbnail loads its bytes through the workspace-raw endpoint;
// a headerless auth stub + a fetch stub keep the test off real auth/network.
vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}) }));

import { TESTID } from '$lib/testids';
import StagedStrip from './StagedStrip.svelte';
import type { Attachment, ContextChip } from './types';

const IMG: Attachment = { id: 'a1', name: 'photo.png' };
const DOC: Attachment = { id: 'a2', name: 'notes.pdf', size: '12 KB' };
const CTX: ContextChip = {
  key: 'session:s1',
  label: 'Look up homebox',
  value: 'title: homebox',
  icon: 'chat',
};

function stubImageFetch() {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }), {
          status: 200,
          headers: { 'Content-Type': 'image/png' },
        }),
    ),
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

test('renders a context chip and opens its value in a preview modal', async () => {
  render(StagedStrip, { contextItems: [CTX] });
  await expect
    .element(page.getByTestId(TESTID.composerContextChip('session:s1')))
    .toBeInTheDocument();
  await page.getByRole('button', { name: 'Look up homebox' }).click();
  await expect.element(page.getByText('title: homebox')).toBeInTheDocument();
});

test('an image attachment renders as a thumbnail (not a filename chip)', async () => {
  stubImageFetch();
  render(StagedStrip, { attachments: [IMG] });
  await expect.element(page.getByTestId(TESTID.chatAttachmentImage)).toBeInTheDocument();
});

test('a non-image attachment shows as a file chip', async () => {
  render(StagedStrip, { attachments: [DOC] });
  await expect.element(page.getByText('notes.pdf · 12 KB')).toBeInTheDocument();
  expect(page.getByTestId(TESTID.chatAttachmentImage).elements()).toHaveLength(0);
});

test('remove buttons fire their callbacks with the right id/key', async () => {
  const onRemoveAttachment = vi.fn();
  const onRemoveContext = vi.fn();
  render(StagedStrip, {
    attachments: [DOC],
    contextItems: [CTX],
    onRemoveAttachment,
    onRemoveContext,
  });
  await page.getByRole('button', { name: 'Remove attachment notes.pdf' }).click();
  expect(onRemoveAttachment).toHaveBeenCalledWith('a2');
  await page.getByRole('button', { name: 'Remove Look up homebox context' }).click();
  expect(onRemoveContext).toHaveBeenCalledWith('session:s1');
});

test('collapses extras behind a "+N more" toggle', async () => {
  const many: ContextChip[] = Array.from({ length: 9 }, (_, i) => ({
    key: `k${i}`,
    label: `item ${i}`,
    value: 'v',
  }));
  render(StagedStrip, { contextItems: many });
  // 9 staged, 6 shown -> "+3 more"; the 9th (item 8) is hidden until expanded.
  await expect.element(page.getByTestId(TESTID.composerStagedMore)).toHaveTextContent('+3 more');
  expect(page.getByText('item 8').elements()).toHaveLength(0);
  await page.getByTestId(TESTID.composerStagedMore).click();
  await expect.element(page.getByText('item 8')).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.composerStagedMore)).toHaveTextContent('show less');
});
