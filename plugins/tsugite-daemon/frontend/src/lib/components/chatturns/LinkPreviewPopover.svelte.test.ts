/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { cleanup, render } from 'vitest-browser-svelte';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import '../../../styles/tokens.css';
import '../../../styles/app.css';
import { api } from '$lib/api/client';
import { pluginsMeta } from '$lib/stores/pluginsMeta.svelte';
import type { LinkPreview } from '$lib/stores/linkPreviews.svelte';
import { TESTID } from '$lib/testids';
import Prose from './Prose.svelte';
import Think from './Think.svelte';

// The preview store is a module singleton keyed by URL. Each test links its own PR.
let n = 0;
const nextUrl = () => `https://forge.example.test/octo-org/demo/pulls/${++n}`;

function preview(url: string, overrides: Partial<LinkPreview> = {}): LinkPreview {
  return {
    label: 'demo#7',
    title: 'Add retry to the sync job',
    url,
    status: { text: 'CI failure', tone: 'err' },
    badges: [
      { text: 'open', tone: 'ok' },
      { text: 'CI failure', tone: 'err' },
    ],
    fields: [{ label: 'author', value: 'octocat' }],
    icon: null,
    ...overrides,
  };
}

beforeEach(() => {
  pluginsMeta.linkResolvers = [
    { plugin: 'forge', name: 'pulls', hosts: ['forge.example.test'], path: '/pulls/\\d+$' },
  ];
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

test('a resolved link renders as a chip showing its state', async () => {
  const url = nextUrl();
  vi.spyOn(api, 'get').mockResolvedValue(preview(url) as never);
  render(Prose, { content: `see ${url}` });

  const chip = page.getByTestId(TESTID.chatLinkChip);
  await expect.element(chip).toHaveTextContent('✕ demo#7');
  await expect.element(chip).toHaveAttribute('href', url);
});

test('hovering a chip opens its preview', async () => {
  const url = nextUrl();
  vi.spyOn(api, 'get').mockResolvedValue(preview(url) as never);
  render(Prose, { content: `see ${url}` });

  await page.getByTestId(TESTID.chatLinkChip).hover();

  const popover = page.getByTestId(TESTID.chatLinkPreviewPopover);
  await expect.element(popover).toHaveTextContent('Add retry to the sync job');
  await expect.element(popover).toHaveTextContent('octocat');
  await expect.element(popover.getByRole('link', { name: 'Open' })).toHaveAttribute('href', url);
});

test('the chip and its preview show the resolver icon', async () => {
  const url = nextUrl();
  const icon = '/api/plugins/forge/ui/icons/forgejo.svg';
  vi.spyOn(api, 'get').mockResolvedValue(preview(url, { icon }) as never);
  render(Prose, { content: `see ${url}` });

  const chip = page.getByTestId(TESTID.chatLinkChip);
  await chip.hover();

  const popover = page.getByTestId(TESTID.chatLinkPreviewPopover);
  await expect.element(popover).toBeVisible();
  for (const owner of [chip, popover]) {
    const mask = owner.element().querySelector('.ic-mask') as HTMLElement;
    expect(getComputedStyle(mask).maskImage).toContain('/api/plugins/forge/ui/icons/forgejo.svg');
    expect(mask.getBoundingClientRect().width).toBeGreaterThan(0);
  }
});

test('a link the daemon has no preview for stays a plain link', async () => {
  const url = nextUrl();
  const get = vi
    .spyOn(api, 'get')
    .mockRejectedValue(Object.assign(new Error('no preview'), { status: 404 }));
  const { container } = await render(Prose, { content: `see ${url}` });

  await expect.poll(() => get.mock.calls.length).toBe(1);
  await new Promise((resolve) => setTimeout(resolve, 50));
  const link = container.querySelector('a') as HTMLAnchorElement;
  expect(link.getAttribute('href')).toBe(url);
  expect(link.classList.contains('t-linkchip')).toBe(false);
});

test('a long chip label stays inside a phone-width pane', async () => {
  const url = nextUrl();
  vi.spyOn(api, 'get').mockResolvedValue(preview(url, { label: 'x'.repeat(200) }) as never);
  const { container } = await render(Prose, { content: `see ${url} now` });
  await expect.element(page.getByTestId(TESTID.chatLinkChip)).toBeVisible();

  container.style.width = '340px';
  container.style.overflowX = 'auto';
  expect(container.scrollWidth).toBeLessThanOrEqual(container.clientWidth);
});

test('the first tap on a chip opens the preview instead of following the link', async () => {
  const url = nextUrl();
  vi.spyOn(api, 'get').mockResolvedValue(preview(url) as never);
  render(Prose, { content: `see ${url}` });
  await expect.element(page.getByTestId(TESTID.chatLinkChip)).toBeVisible();
  const chip = page.getByTestId(TESTID.chatLinkChip).element() as HTMLAnchorElement;

  const tap = new PointerEvent('click', { pointerType: 'touch', bubbles: true, cancelable: true });
  chip.dispatchEvent(tap);

  expect(tap.defaultPrevented).toBe(true);
  await expect.element(page.getByTestId(TESTID.chatLinkPreviewPopover)).toBeVisible();
});

test('reasoning links render as chips too', async () => {
  const url = nextUrl();
  vi.spyOn(api, 'get').mockResolvedValue(preview(url) as never);
  render(Think, { content: `checked ${url}`, open: true });

  await page.getByTestId(TESTID.chatLinkChip).hover();

  await expect.element(page.getByTestId(TESTID.chatLinkPreviewPopover)).toBeVisible();
});

test('a bubble rendered before the resolver patterns load gets its chip when they arrive', async () => {
  const url = nextUrl();
  vi.spyOn(api, 'get').mockResolvedValue(preview(url) as never);
  const patterns = pluginsMeta.linkResolvers;
  pluginsMeta.linkResolvers = [];
  render(Prose, { content: `see ${url}` });

  pluginsMeta.linkResolvers = patterns;

  await expect.element(page.getByTestId(TESTID.chatLinkChip)).toBeVisible();
});
