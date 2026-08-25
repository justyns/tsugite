/// <reference types="vitest/browser" />
import { page } from 'vitest/browser';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';

// The blob helper reads the bearer header from the real client store; a headerless
// stub keeps the test off any auth state. navigate is spied to assert the chip's
// files-view open without a real router.
vi.mock('$lib/api/client', () => ({ authHeaders: () => ({}) }));
vi.mock('$lib/router.svelte', () => ({ navigate: vi.fn() }));

import { navigate } from '$lib/router.svelte';
import { TESTID } from '$lib/testids';
import Attachments from './Attachments.svelte';
import type { TurnAttachment } from './turns';

const IMG: TurnAttachment = { name: 'photo.png', type: 'image', path: 'uploads/photo.png' };
const DOC: TurnAttachment = { name: 'notes.pdf', type: 'document', path: 'uploads/notes.pdf' };

/** Global fetch returning a tiny image blob, standing in for GET /workspace/raw. */
function stubImageFetch() {
  const fetchMock = vi.fn(
    async (_input: RequestInfo | URL, _init?: RequestInit) =>
      new Response(new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }), {
        status: 200,
        headers: { 'Content-Type': 'image/png' },
      }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

test('renders an image attachment as a thumbnail read from the raw endpoint', async () => {
  const fetchMock = stubImageFetch();
  render(Attachments, { attachments: [IMG] });

  await expect.element(page.getByAltText('photo.png')).toBeInTheDocument();
  expect(String(fetchMock.mock.calls[0]![0])).toContain(
    '/api/workspace/raw?path=uploads%2Fphoto.png',
  );
});

test('renders a non-image attachment as a chip that opens the files view', async () => {
  stubImageFetch();
  render(Attachments, { attachments: [DOC] });

  const chip = page.getByTestId(TESTID.chatAttachmentChip);
  await expect.element(chip).toBeInTheDocument();
  await chip.click();
  expect(navigate).toHaveBeenCalledWith('files', { path: 'uploads/notes.pdf' });
});

test('clicking a thumbnail opens the full-image lightbox, and close dismisses it', async () => {
  stubImageFetch();
  render(Attachments, { attachments: [IMG] });

  // Wait for the blob to load (the button is disabled until then), then open it.
  await expect.element(page.getByAltText('photo.png')).toBeInTheDocument();
  await page.getByRole('button', { name: 'View photo.png' }).click();

  const dialog = page.getByRole('dialog', { name: 'photo.png' });
  await expect.element(dialog).toBeVisible();
  await expect.element(dialog.getByAltText('photo.png')).toBeInTheDocument();

  await page.getByRole('button', { name: 'Close' }).click();
  await expect.element(page.getByRole('dialog', { name: 'photo.png' })).not.toBeInTheDocument();
});

test('a failed load shows a broken-file placeholder instead of throwing', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response('missing', { status: 404 })),
  );
  render(Attachments, { attachments: [IMG] });

  await expect.element(page.getByTestId(TESTID.chatAttachmentImage)).toBeInTheDocument();
  await expect.element(page.getByAltText('photo.png')).not.toBeInTheDocument();
});

test('revokes the thumbnail object URL on unmount so a long chat never leaks', async () => {
  stubImageFetch();
  const createSpy = vi.spyOn(URL, 'createObjectURL');
  const revokeSpy = vi.spyOn(URL, 'revokeObjectURL');
  render(Attachments, { attachments: [IMG] });

  await expect.element(page.getByAltText('photo.png')).toBeInTheDocument();
  expect(createSpy).toHaveBeenCalledTimes(1);
  const url = createSpy.mock.results[0]!.value as string;

  cleanup();
  expect(revokeSpy).toHaveBeenCalledWith(url);
});
