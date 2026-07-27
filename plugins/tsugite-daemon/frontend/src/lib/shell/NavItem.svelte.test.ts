/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import NavItem from './NavItem.svelte';

afterEach(() => {
  // A modified click that falls through sets the fragment; reset it so it can't
  // leak into the next test.
  if (location.hash) history.replaceState(null, '', location.pathname);
});

test('the active row is marked with aria-current and links to its hash', async () => {
  await render(NavItem, { id: 'jobs', label: 'Jobs', icon: 'jobs', active: true });
  const link = page.getByRole('link', { name: 'Jobs' });
  await expect.element(link).toHaveAttribute('aria-current', 'page');
  await expect.element(link).toHaveAttribute('href', '#jobs');
});

test('an inactive row carries no aria-current', async () => {
  await render(NavItem, { id: 'jobs', label: 'Jobs', icon: 'jobs', active: false });
  await expect
    .element(page.getByRole('link', { name: 'Jobs' }))
    .not.toHaveAttribute('aria-current');
});

test('with no live count the badge slot renders nothing', async () => {
  const { container } = await render(NavItem, { id: 'files', label: 'Files', icon: 'files' });
  expect(container.querySelector('.bdg')).toBeNull();
});

test('a plain click opens the view in place instead of navigating', async () => {
  const onactivate = vi.fn();
  const { container } = await render(NavItem, {
    id: 'jobs',
    label: 'Jobs',
    icon: 'jobs',
    onactivate,
  });
  const link = container.querySelector<HTMLAnchorElement>('a')!;
  const event = new MouseEvent('click', { bubbles: true, cancelable: true, button: 0 });
  link.dispatchEvent(event);
  expect(onactivate).toHaveBeenCalledWith('jobs');
  expect(event.defaultPrevented).toBe(true);
});

test('a modified click falls through to the anchor deep-link', async () => {
  const onactivate = vi.fn();
  const { container } = await render(NavItem, {
    id: 'jobs',
    label: 'Jobs',
    icon: 'jobs',
    onactivate,
  });
  const link = container.querySelector<HTMLAnchorElement>('a')!;
  const event = new MouseEvent('click', {
    bubbles: true,
    cancelable: true,
    button: 0,
    ctrlKey: true,
  });
  link.dispatchEvent(event);
  expect(onactivate).not.toHaveBeenCalled();
  expect(event.defaultPrevented).toBe(false);
});
