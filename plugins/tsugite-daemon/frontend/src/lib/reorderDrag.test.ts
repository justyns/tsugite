import { describe, expect, it, vi } from 'vitest';
import { ReorderDrag } from './reorderDrag.svelte';

const ids = ['a', 'b', 'c', 'd'];
const indexOf = (id: string) => ids.indexOf(id);

/** A drag event over a 100x20 item whose top-left is the origin. */
function event(at: { x?: number; y?: number } = {}) {
  return {
    clientX: at.x ?? 0,
    clientY: at.y ?? 0,
    currentTarget: { getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 20 }) },
    preventDefault: vi.fn(),
    stopPropagation: vi.fn(),
  } as unknown as DragEvent & { preventDefault: ReturnType<typeof vi.fn> };
}

describe('ReorderDrag', () => {
  it('drops before an item the pointer is on the near half of', () => {
    const drag = new ReorderDrag(indexOf);
    drag.start('a');
    drag.over(event({ x: 10 }), 'c');
    expect(drag.dropAt).toBe(2);
  });

  it('drops after an item the pointer is past the midpoint of', () => {
    const drag = new ReorderDrag(indexOf);
    drag.start('a');
    drag.over(event({ x: 90 }), 'c');
    expect(drag.dropAt).toBe(3);
  });

  it('splits on the vertical midpoint when the axis is y', () => {
    const drag = new ReorderDrag(indexOf, { axis: 'y' });
    drag.start('a');
    drag.over(event({ y: 15 }), 'c');
    expect(drag.dropAt).toBe(3);
  });

  it('ignores a dragover when no drag of its own started', () => {
    const drag = new ReorderDrag(indexOf);
    const e = event({ x: 90 });
    drag.over(e, 'c');
    expect(drag.dropAt).toBeNull();
    expect(e.preventDefault).not.toHaveBeenCalled();
  });

  it('reports the move a drop performs', () => {
    const drag = new ReorderDrag(indexOf);
    drag.start('a');
    drag.over(event({ x: 90 }), 'c');
    expect(drag.drop(event())).toEqual({ id: 'a', insertAt: 3 });
  });

  it('reports nothing when the drop lands where the item already was', () => {
    const drag = new ReorderDrag(indexOf);
    drag.start('b');
    drag.over(event({ x: 10 }), 'b');
    expect(drag.drop(event())).toBeNull();
  });

  it('reports nothing when the drop lands just after the item, which is the same place', () => {
    const drag = new ReorderDrag(indexOf);
    drag.start('b');
    drag.over(event({ x: 90 }), 'b');
    expect(drag.drop(event())).toBeNull();
  });

  it('clears the drag whether or not the drop moved anything', () => {
    const drag = new ReorderDrag(indexOf);
    drag.start('b');
    drag.over(event({ x: 10 }), 'b');
    drag.drop(event());
    expect(drag.dragging).toBeNull();
    expect(drag.dropAt).toBeNull();
  });

  it('stops propagation only when asked, so an outer drop target can still act', () => {
    const plain = new ReorderDrag(indexOf);
    plain.start('a');
    const loose = event({ x: 90 });
    plain.over(loose, 'c');
    expect(loose.stopPropagation).not.toHaveBeenCalled();

    const nested = new ReorderDrag(indexOf, { stopPropagation: true });
    nested.start('a');
    const held = event({ x: 90 });
    nested.over(held, 'c');
    expect(held.stopPropagation).toHaveBeenCalled();
  });
});
