/// <reference types="@vitest/browser/context" />
// Runs in the browser project (the `.svelte.test.ts` glob) because the re-encode
// needs a real canvas / createImageBitmap — unavailable in the node project.
import { expect, test } from 'vitest';
import { reencodeImage } from './imageEncode';

const CFG = { maxEdge: 1568, quality: 0.85 };

async function makePng(w: number, h: number): Promise<File> {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#3366cc';
  ctx.fillRect(0, 0, w, h);
  const blob = await new Promise<Blob>((res) => c.toBlob((b) => res(b!), 'image/png'));
  return new File([blob], 'photo.png', { type: 'image/png' });
}

test('downscales a large image to the max edge and re-encodes as JPEG', async () => {
  const out = await reencodeImage(await makePng(3000, 2000), CFG);
  expect(out.type).toBe('image/jpeg');
  expect(out.name).toMatch(/\.jpg$/);
  const bmp = await createImageBitmap(out);
  expect(Math.max(bmp.width, bmp.height)).toBe(1568);
  expect(bmp.height).toBe(Math.round(2000 * (1568 / 3000)));
});

test('keeps a small image at its own size but still emits JPEG', async () => {
  const out = await reencodeImage(await makePng(800, 600), CFG);
  expect(out.type).toBe('image/jpeg');
  const bmp = await createImageBitmap(out);
  expect(bmp.width).toBe(800);
  expect(bmp.height).toBe(600);
});

test('passes an SVG through untouched (a vector must not be rasterized)', async () => {
  const svg = new File(['<svg xmlns="http://www.w3.org/2000/svg"></svg>'], 'd.svg', {
    type: 'image/svg+xml',
  });
  expect(await reencodeImage(svg, CFG)).toBe(svg);
});

test('passes a GIF through untouched (animation must survive)', async () => {
  const gif = new File([new Uint8Array([0x47, 0x49, 0x46])], 'a.gif', { type: 'image/gif' });
  expect(await reencodeImage(gif, CFG)).toBe(gif);
});

test('passes a non-image file through untouched', async () => {
  const txt = new File(['hello'], 'notes.txt', { type: 'text/plain' });
  expect(await reencodeImage(txt, CFG)).toBe(txt);
});
