/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import { type Snippet, createRawSnippet, mount, unmount } from 'svelte';
import Mux from './Mux.svelte';
import PluginSurface from '$lib/components/plugins/PluginSurface.svelte';
import { pluginsMeta, type PluginSurface as SurfaceDef } from '$lib/stores/pluginsMeta.svelte';
import { writeSurfaceDrag } from './drag';
import {
  type Layout,
  type PaneTabModel,
  type SplitDir,
  type SurfaceRef,
  closeTab,
  collectLeaves,
  defaultLayout,
  dockAsTab,
  focusPane,
  resizeSplit,
  selectTab,
  splitPane,
} from './layout';

afterEach(cleanup);

function seeded(ref: SurfaceRef = { kind: 'chat' }): Layout {
  const l = defaultLayout();
  return dockAsTab(l, l.root.id, ref);
}

// Wire Mux to the real reducers so an interaction exercises the whole loop:
// event -> Mux callback -> layout op -> rerender. Mirrors the chrome's wiring.
async function mountMux(initial: Layout, content?: Snippet<[PaneTabModel, () => void]>) {
  let layout = initial;
  let rerender!: (props: Record<string, unknown>) => Promise<void>;
  // Force the desktop split tree; the test viewport is narrow enough to trip the
  // <=700px single-pane mode otherwise.
  const apply = (next: Layout) => {
    layout = next;
    void rerender({ layout, narrow: false, content, ...handlers });
  };
  const handlers = {
    onSplit: (p: string, d: SplitDir, ref: SurfaceRef, pos: 'before' | 'after') =>
      apply(splitPane(layout, p, d, ref, pos)),
    onDock: (p: string, ref: SurfaceRef) => apply(dockAsTab(layout, p, ref)),
    onCloseTab: (p: string, t: string) => apply(closeTab(layout, p, t)),
    onSelectTab: (p: string, t: string) => apply(selectTab(layout, p, t)),
    onFocusPane: (p: string) => apply(focusPane(layout, p)),
    onResize: (s: string, i: number, d: number) => apply(resizeSplit(layout, s, i, d)),
  };
  const screen = await render(Mux, { layout, narrow: false, content, ...handlers });
  rerender = screen.rerender;
  const panes = () => [
    ...screen.container.querySelectorAll<HTMLElement>('[data-testid="mux-pane"]'),
  ];
  return {
    ...screen,
    panes,
    get layout() {
      return layout;
    },
  };
}

function dropSurface(el: HTMLElement, ref: SurfaceRef, fracX: number) {
  const r = el.getBoundingClientRect();
  const clientX = r.left + r.width * fracX;
  const clientY = r.top + r.height * 0.5;
  const dataTransfer = new DataTransfer();
  writeSurfaceDrag(dataTransfer, ref);
  for (const type of ['dragover', 'drop']) {
    el.dispatchEvent(
      new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer, clientX, clientY }),
    );
  }
}

test('dropping a surface on the left third splits the pane and places the new pane first', async () => {
  const mux = await mountMux(seeded({ kind: 'chat' }));
  expect(mux.panes()).toHaveLength(1);
  dropSurface(mux.panes()[0]!, { kind: 'terminal' }, 0.1);
  await expect.poll(() => mux.panes().length).toBe(2);
  // "before": the new terminal pane is on the left. The surface label capitalizes
  // the kind ("Terminal"/"Chat"), so match case-insensitively.
  expect(mux.panes()[0]!.textContent?.toLowerCase()).toContain('terminal');
  expect(mux.panes()[1]!.textContent?.toLowerCase()).toContain('chat');
});

test('dropping a surface on the right third splits and places the new pane last', async () => {
  const mux = await mountMux(seeded({ kind: 'chat' }));
  dropSurface(mux.panes()[0]!, { kind: 'terminal' }, 0.9);
  await expect.poll(() => mux.panes().length).toBe(2);
  expect(mux.panes()[0]!.textContent?.toLowerCase()).toContain('chat');
  expect(mux.panes()[1]!.textContent?.toLowerCase()).toContain('terminal');
});

test('dropping a surface in the centre docks it as a new tab, not a split', async () => {
  const mux = await mountMux(seeded({ kind: 'chat' }));
  dropSurface(mux.panes()[0]!, { kind: 'terminal' }, 0.5);
  // still one pane...
  await expect.element(page.getByRole('tab', { name: /terminal/i })).toBeInTheDocument();
  expect(mux.panes()).toHaveLength(1);
  // ...now holding both tabs
  await expect.element(page.getByRole('tab', { name: /chat/i })).toBeInTheDocument();
});

test('closing the last tab of a split pane collapses back to a single pane', async () => {
  const base = seeded({ kind: 'chat' });
  const split = splitPane(base, base.root.id, 'row', { kind: 'terminal' });
  const mux = await mountMux(split);
  expect(mux.panes()).toHaveLength(2);
  // Close via the accessible path: select the tab, then Delete (the pointer-only
  // x button is aria-hidden and has no role name).
  await userEvent.click(page.getByRole('tab', { name: /terminal/i }));
  await userEvent.keyboard('{Delete}');
  await expect.poll(() => mux.panes().length).toBe(1);
  expect(mux.panes()[0]!.textContent?.toLowerCase()).toContain('chat');
});

test('keyboard resize on a divider shifts the split ratio (aria-valuenow)', async () => {
  const base = seeded({ kind: 'chat' });
  const split = splitPane(base, base.root.id, 'row', { kind: 'terminal' });
  const mux = await mountMux(split);
  const sep = () => mux.container.querySelector<HTMLElement>('[role="separator"]');
  expect(sep()?.getAttribute('aria-valuenow')).toBe('50');
  sep()!.dispatchEvent(
    new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }),
  );
  await expect.poll(() => Number(sep()?.getAttribute('aria-valuenow'))).toBeGreaterThan(50);
});

test('an empty pane offers a real open-a-surface action, not the impossible drag hint', async () => {
  const layout = defaultLayout();
  const onNewTab = vi.fn();
  const { container } = await render(Mux, { layout, narrow: false, onNewTab });
  // The old copy told the user to "drag a row onto a pane", but the app ships no
  // drag source, so that instruction was impossible to act on.
  expect(container.textContent).not.toMatch(/drag a row onto a pane/i);
  // The empty pane instead exposes the open action, anchored to this pane so the
  // surface the user picks docks here.
  const open = page.getByRole('button', { name: /open a surface/i });
  await expect.element(open).toBeInTheDocument();
  await open.click();
  expect(onNewTab).toHaveBeenCalledWith(layout.root.id);
});

test('a surface can claim pane focus itself, the same as a click on the pane', async () => {
  // A surface that swallows its own pointer events (a plugin iframe) never lets
  // PaneView's pointerdown see the click, so the mux hands every rendered surface
  // the same focus call that wrapper makes.
  const focusers = new Map<string, (() => void) | undefined>();
  const content = createRawSnippet<[PaneTabModel, () => void]>((tab, focusPane) => {
    focusers.set(tab().kind, focusPane?.());
    return { render: () => '<div></div>' };
  });
  const base = seeded({ kind: 'chat' });
  const split = splitPane(base, base.root.id, 'row', { kind: 'terminal' });
  const mux = await mountMux(split, content);
  const chatPane = collectLeaves(mux.layout.root).find((l) =>
    l.tabs.some((t) => t.kind === 'chat'),
  )!;
  expect(mux.layout.focusedPaneId).not.toBe(chatPane.id);

  focusers.get('chat')?.();

  await expect.poll(() => mux.layout.focusedPaneId).toBe(chatPane.id);
});

test('a pane grows to fill its slot rather than collapsing to content height', async () => {
  // `.mux-pane` carries no flex-grow, so PaneView's wrapper must set it -
  // otherwise every pane sits at content height with dead space below it.
  const { container } = await render(Mux, { layout: seeded({ kind: 'chat' }), narrow: false });
  const pane = container.querySelector<HTMLElement>('[data-testid="mux-pane"] .mux-pane');
  expect(pane).not.toBeNull();
  expect(getComputedStyle(pane!).flexGrow).toBe('1');
});

test('narrow mode renders one pane plus a switcher instead of the split tree', async () => {
  const base = seeded({ kind: 'chat' });
  const split = splitPane(base, base.root.id, 'row', { kind: 'terminal' });
  const { container } = await render(Mux, { layout: split, narrow: true });
  expect(container.querySelectorAll('[data-testid="mux-pane"]')).toHaveLength(1);
  await expect.element(page.getByRole('group', { name: 'Switch pane' })).toBeInTheDocument();
});

test('narrow (phone) mode hides the split affordance - phone width never splits', async () => {
  const base = seeded({ kind: 'chat' });
  const split = splitPane(base, base.root.id, 'row', { kind: 'terminal' });
  const { container } = await render(Mux, { layout: split, narrow: true });
  expect(container.querySelectorAll('[data-testid="mux-pane"]')).toHaveLength(1);
  // Phones get one pane + a switcher, so the split button must not be reachable
  // (clicking it mutates the model into a hidden second pane).
  expect(container.querySelector('button[aria-label="Split pane"]')).toBeNull();
  // ...but tabs can still be closed - only splitting is withheld.
  expect(container.querySelector('.mux-tab .x')).not.toBeNull();
});

test('closing a pane via its tab strip restores focus into the workspace, not <body> (WCAG 2.4.3)', async () => {
  const base = seeded({ kind: 'chat' });
  const split = splitPane(base, base.root.id, 'row', { kind: 'terminal' });
  const mux = await mountMux(split);
  expect(mux.panes()).toHaveLength(2);
  // Close the terminal pane's only tab like a keyboard user: focus the tab,
  // press Delete. The last tab closing collapses the pane itself.
  await userEvent.click(page.getByRole('tab', { name: /terminal/i }));
  await userEvent.keyboard('{Delete}');
  await expect.poll(() => mux.panes().length).toBe(1);
  // Focus must land on the surviving focused pane, not fall through to <body>.
  await expect.poll(() => document.activeElement !== document.body).toBe(true);
  const focused = mux.container.querySelector<HTMLElement>(
    '[data-testid="mux-pane"][data-focused="true"]',
  );
  expect(focused).not.toBeNull();
  expect(focused!.contains(document.activeElement)).toBe(true);
});

test('closing the active tab (the pane survives) also keeps focus in the workspace', async () => {
  let l = seeded({ kind: 'chat', params: { id: 'a' } });
  l = dockAsTab(l, l.root.id, { kind: 'terminal', params: { id: 'b' } });
  const mux = await mountMux(l);
  // The pointer-only close control is a non-focusable span; focus sits on the
  // tab itself when a pointer user closes it. Prove focus recovers into the
  // workspace when that focused tab unmounts. Terminal is the second tab.
  const closeSpans = () => mux.container.querySelectorAll<HTMLElement>('.mux-tab .x');
  const termTab = mux.container.querySelectorAll<HTMLElement>('[data-tab-id]')[1]!;
  termTab.focus();
  closeSpans()[1]!.click();
  // Wait for the close to process (the terminal tab unmounts).
  await expect.poll(() => closeSpans().length).toBe(1);
  await expect.poll(() => document.activeElement !== document.body).toBe(true);
  const focused = mux.container.querySelector<HTMLElement>(
    '[data-testid="mux-pane"][data-focused="true"]',
  );
  expect(focused!.contains(document.activeElement)).toBe(true);
});

// ── focus recovery vs a surface whose content is a frame of its own ──
//
// A document editor's shape: a plugin page framing a document it does not own. A
// data: URL carries an opaque origin, so the inner frame below is as unreadable
// to the page holding it as the real one is, and the click that lands in it
// reaches neither that page nor the host.
const INNER_DOCUMENT = encodeURIComponent(
  '<body style="margin:0"><input id="t" style="width:100%;height:100%">' +
    '<script>' +
    "var t = document.getElementById('t');" +
    "addEventListener('pointerdown', function () { t.focus(); });" +
    "addEventListener('message', function (e) {" +
    "  if (e.data && e.data.type === 'harness:ask')" +
    "    parent.postMessage({ type: 'harness:typed', value: t.value }, '*');" +
    '});' +
    '<\/script></body>',
  // encodeURIComponent leaves apostrophes, which would close the string
  // literal this URL is built into on the page below.
).replace(/'/g, '%27');

const FRAMING_PLUGIN_PAGE = `
  document.body.style.margin = '0';
  const inner = document.createElement('iframe');
  inner.style.cssText = 'width:100%;height:100%;border:0;display:block';
  inner.src = 'data:text/html,${INNER_DOCUMENT}';
  document.body.append(inner);
  addEventListener('message', (e) => {
    if (e.source === inner.contentWindow) return parent.postMessage(e.data, '*');
    if (!e.data || typeof e.data !== 'object') return;
    if (e.data.type === 'tsugite:init') parent.postMessage({ type: 'tsugite:ready' }, '*');
    if (e.data.type === 'harness:ask') inner.contentWindow.postMessage(e.data, '*');
  });
`;

const blobUrls: string[] = [];

function seedFramingSurface(): void {
  const url = URL.createObjectURL(
    new Blob(
      [`<!doctype html><meta charset="utf-8"><body><script>${FRAMING_PLUGIN_PAGE}<\/script>`],
      {
        type: 'text/html',
      },
    ),
  );
  blobUrls.push(url);
  const surface: SurfaceDef = {
    plugin: 'demo',
    kind: 'plugin/demo/doc',
    label: 'Document',
    icon: 'files',
    entry: url,
    nav: false,
    params: [],
    events: [],
    mode: 'workspace',
  };
  pluginsMeta.surfaces = [surface];
  pluginsMeta.loaded = true;
}

afterEach(() => {
  pluginsMeta.surfaces = [];
  pluginsMeta.loaded = false;
  for (const url of blobUrls.splice(0)) URL.revokeObjectURL(url);
});

test('clicking into a plugin surface claims the pane without taking focus off the frame', async () => {
  // The claim the click raises runs a layout op, which re-runs Mux's recovery
  // effect. Recovering here would move focus onto the pane wrapper, and an editor
  // two frames down would go on drawing a caret while every keystroke landed
  // nowhere, so typing is what this asserts on.
  seedFramingSurface();
  const content = createRawSnippet<[PaneTabModel, () => void]>((tab, focusPane) => ({
    render: () => '<div style="height:100%"></div>',
    setup: (node: Element) => {
      if (tab().kind !== 'plugin/demo/doc') return;
      const app = mount(PluginSurface, {
        target: node,
        props: { kind: tab().kind, params: {}, focusPane: focusPane?.() },
      });
      return () => void unmount(app);
    },
  }));

  const base = seeded({ kind: 'chat' });
  const split = splitPane(base, base.root.id, 'row', { kind: 'plugin/demo/doc' });
  const mux = await mountMux(split, content);
  await expect
    .poll(() => document.querySelector('[data-phase]')?.getAttribute('data-phase'), {
      timeout: 8000,
    })
    .toBe('ready');

  // Start with the other pane holding focus, so the click has a claim to make.
  const [chatPane, docPane] = mux.panes();
  chatPane!.focus();
  await expect.poll(() => chatPane!.dataset.focused).toBe('true');

  const frame = document.querySelector('iframe')!;
  await userEvent.click(frame);
  await expect.poll(() => docPane!.dataset.focused).toBe('true');
  expect(document.activeElement).toBe(frame);

  await userEvent.keyboard('typed');
  let typed: string | undefined;
  window.addEventListener('message', (e) => {
    const data = e.data as { type?: string; value?: string };
    if (data?.type === 'harness:typed') typed = data.value;
  });
  frame.contentWindow!.postMessage({ type: 'harness:ask' }, '*');
  await expect.poll(() => typed).toBe('typed');
});
