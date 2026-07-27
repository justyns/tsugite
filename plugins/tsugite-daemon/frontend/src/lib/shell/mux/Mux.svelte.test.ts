/// <reference types="@vitest/browser/context" />
import { page, userEvent } from '@vitest/browser/context';
import { render, cleanup } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import Mux from './Mux.svelte';
import { writeSurfaceDrag } from './drag';
import {
  type Layout,
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
async function mountMux(initial: Layout) {
  let layout = initial;
  let rerender!: (props: Record<string, unknown>) => Promise<void>;
  // Force the desktop split tree; the test viewport is narrow enough to trip the
  // <=700px single-pane mode otherwise.
  const apply = (next: Layout) => {
    layout = next;
    void rerender({ layout, narrow: false, ...handlers });
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
  const screen = await render(Mux, { layout, narrow: false, ...handlers });
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
