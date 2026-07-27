/// <reference types="@vitest/browser/context" />
import { render } from 'vitest-browser-svelte';
import { expect, test } from 'vitest';
import Icon from '$lib/components/icon/Icon.svelte';

// Proves the vitest `browser` project (headless chromium via Playwright) can
// mount a real Svelte 5 component. Real component interaction tests are
// colocated as `<Name>.svelte.test.ts`; this stays dependency-free so it never
// fails for a reason other than "the browser harness itself is broken".
test('mounts a Svelte component in the browser harness', async () => {
  const { container } = await render(Icon, { props: { name: 'chat' } });
  expect(container.querySelector('svg')).not.toBeNull();
});
