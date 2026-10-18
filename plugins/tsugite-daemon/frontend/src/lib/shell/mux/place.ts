import type { LayoutNode, LeafNode, SplitDir } from './layout';

/** Fractions (0-1) of the pane area. */
export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface PanePlacement extends Rect {
  leaf: LeafNode;
}

export interface DividerPlacement {
  splitId: string;
  index: number;
  dir: SplitDir;
  /** Divider position along the split's axis, as a fraction of the pane area. */
  at: number;
  /** The split's extent across the divider, as a fraction of the pane area. */
  cross: Rect;
  /** The split's size along its axis, as a fraction of the pane area. */
  span: number;
  /** First pane's share of the pair the divider sits between, as a whole percent. */
  pct: number;
}

export interface Placement {
  panes: PanePlacement[];
  dividers: DividerPlacement[];
}

/** Narrowest a pane may get before the pane area scrolls instead of squeezing it. */
export const MIN_PANE_PX = 260;

/** Lays the tree out as one flat list, so every pane can be rendered once under a
 *  stable key and moved by position when the tree changes shape around it. */
export function placeTree(root: LayoutNode): Placement {
  const out: Placement = { panes: [], dividers: [] };
  walk(root, { x: 0, y: 0, w: 1, h: 1 }, out);
  return out;
}

function walk(node: LayoutNode, rect: Rect, out: Placement): void {
  if (node.type === 'leaf') {
    out.panes.push({ leaf: node, ...rect });
    return;
  }
  const row = node.dir === 'row';
  const total = node.sizes.reduce((a, b) => a + b, 0);
  const span = row ? rect.w : rect.h;
  let offset = row ? rect.x : rect.y;
  node.children.forEach((child, i) => {
    const size = (span * node.sizes[i]!) / total;
    walk(child, row ? { ...rect, x: offset, w: size } : { ...rect, y: offset, h: size }, out);
    offset += size;
    if (i < node.children.length - 1) {
      const a = node.sizes[i]!;
      const b = node.sizes[i + 1]!;
      out.dividers.push({
        splitId: node.id,
        index: i,
        dir: node.dir,
        at: offset,
        cross: rect,
        span,
        pct: a + b > 0 ? Math.round((a / (a + b)) * 100) : 50,
      });
    }
  });
}
