/**
 * Icon registry. Each entry keeps its own viewBox and full inner markup (some
 * icons are multi-shape, and some shapes carry per-shape fill/stroke/stroke-width
 * overrides), so Icon.svelte can render them as a self-contained <svg> with no
 * global sprite mount.
 */

export interface IconDef {
  viewBox: string;
  body: string;
}

export const ICONS = {
  grid: {
    viewBox: '0 0 16 16',
    body: '<rect x="2.5" y="2.5" width="4.6" height="4.6"/><rect x="8.9" y="2.5" width="4.6" height="4.6"/><rect x="2.5" y="8.9" width="4.6" height="4.6"/><rect x="8.9" y="8.9" width="4.6" height="4.6"/>',
  },
  dot: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="8" r="4" fill="currentColor" stroke="none"/>',
  },
  ring: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="8" r="4.5"/>',
  },
  play: {
    viewBox: '0 0 16 16',
    body: '<path d="M5 3.5v9l7-4.5z" fill="currentColor" stroke="none"/>',
  },
  stop: {
    viewBox: '0 0 16 16',
    body: '<rect x="4.5" y="4.5" width="7" height="7" rx="1" fill="currentColor" stroke="none"/>',
  },
  pause: {
    viewBox: '0 0 16 16',
    body: '<path d="M6 4v8M10 4v8" stroke-width="2"/>',
  },
  check: {
    viewBox: '0 0 16 16',
    body: '<path d="M3.5 8.5l3 3 6-6.5"/>',
  },
  x: {
    viewBox: '0 0 16 16',
    body: '<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/>',
  },
  cancel: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="8" r="5"/><path d="M4.6 11.4l6.8-6.8"/>',
  },
  clock: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="8" r="5.5"/><path d="M8 5.2V8l2 1.4"/>',
  },
  alert: {
    viewBox: '0 0 16 16',
    body: '<path d="M8 2.6L14.2 13H1.8z"/><path d="M8 6.6v2.8"/><circle cx="8" cy="11.4" r=".4" fill="currentColor" stroke="none"/>',
  },
  q: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="8" r="5.5"/><path d="M6.3 6.4c.2-1 1-1.6 1.9-1.5.9 0 1.7.7 1.7 1.6 0 1.1-1.1 1.3-1.8 2v.7"/><circle cx="8" cy="11.3" r=".4" fill="currentColor" stroke="none"/>',
  },
  search: {
    viewBox: '0 0 16 16',
    body: '<circle cx="7" cy="7" r="4"/><path d="M10 10l3.4 3.4"/>',
  },
  'chev-d': {
    viewBox: '0 0 16 16',
    body: '<path d="M4 6l4 4 4-4"/>',
  },
  'chev-r': {
    viewBox: '0 0 16 16',
    body: '<path d="M6 4l4 4-4 4"/>',
  },
  pin: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="6" r="3.4"/><path d="M8 9.4V14"/>',
  },
  term: {
    viewBox: '0 0 16 16',
    body: '<path d="M3 4.5l3.5 3.5L3 11.5"/><path d="M8.5 11.5H13"/>',
  },
  send: {
    viewBox: '0 0 16 16',
    body: '<path d="M2.5 8L13.5 2.8 10.6 13.2 8 9.5z" fill="currentColor" stroke="none"/>',
  },
  retry: {
    viewBox: '0 0 16 16',
    body: '<path d="M13 8a5 5 0 1 1-1.5-3.6"/><path d="M13 2.8v2.4h-2.4" fill="none"/>',
  },
  copy: {
    viewBox: '0 0 16 16',
    body: '<rect x="5.5" y="5.5" width="7" height="7" rx="1.5"/><path d="M3.5 10.5v-6a1 1 0 0 1 1-1h6"/>',
  },
  compress: {
    viewBox: '0 0 16 16',
    body: '<path d="M4 3l4 3.4L12 3"/><path d="M4 13l4-3.4 4 3.4"/>',
  },
  chat: {
    viewBox: '0 0 16 16',
    body: '<path d="M2.5 3.5h11v7h-6l-3 2.8v-2.8h-2z"/>',
  },
  jobs: {
    viewBox: '0 0 16 16',
    body: '<rect x="2.5" y="3" width="3.2" height="10"/><rect x="6.9" y="3" width="3.2" height="6.4"/><rect x="11.3" y="3" width="3.2" height="8.4"/>',
  },
  agent: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="5.4" r="2.6"/><path d="M3.2 13.4c.6-2.6 2.5-3.8 4.8-3.8s4.2 1.2 4.8 3.8"/>',
  },
  skill: {
    viewBox: '0 0 16 16',
    body: '<path d="M8 2.5L13.5 8 8 13.5 2.5 8z"/>',
  },
  sched: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="8.5" r="5"/><path d="M8 6v2.5l1.8 1.2M5.5 2.2h5"/>',
  },
  usage: {
    viewBox: '0 0 16 16',
    body: '<path d="M2.5 13.5h11M4 13V9M8 13V5M12 13V7"/>',
  },
  files: {
    viewBox: '0 0 16 16',
    body: '<path d="M2.5 4.5h4l1.4 1.6h5.6v6.4h-11z"/>',
  },
  hook: {
    viewBox: '0 0 16 16',
    body: '<circle cx="11" cy="5" r="2.6"/><path d="M8.9 6.9L3 12.8M3 9.2v3.6h3.6"/>',
  },
  plus: {
    viewBox: '0 0 16 16',
    body: '<path d="M8 3.5v9M3.5 8h9"/>',
  },
  dots: {
    viewBox: '0 0 16 16',
    body: '<circle cx="3.4" cy="8" r=".9" fill="currentColor" stroke="none"/><circle cx="8" cy="8" r=".9" fill="currentColor" stroke="none"/><circle cx="12.6" cy="8" r=".9" fill="currentColor" stroke="none"/>',
  },
  edit: {
    viewBox: '0 0 16 16',
    body: '<path d="M3 13l.9-3L10.4 3.5l2.1 2.1L6 12.1z"/>',
  },
  fork: {
    viewBox: '0 0 16 16',
    body: '<circle cx="4.5" cy="4" r="1.7"/><circle cx="11.5" cy="4" r="1.7"/><circle cx="8" cy="12" r="1.7"/><path d="M4.5 5.7c0 2.5 3.5 2 3.5 4.6M11.5 5.7c0 2.5-3.5 2-3.5 4.6"/>',
  },
  file: {
    viewBox: '0 0 16 16',
    body: '<path d="M4 2.5h5l3 3v8H4z"/><path d="M9 2.5v3h3"/>',
  },
  lock: {
    viewBox: '0 0 16 16',
    body: '<rect x="4" y="7" width="8" height="6.5" rx="1"/><path d="M5.7 7V5.2a2.3 2.3 0 0 1 4.6 0V7"/>',
  },
  down: {
    viewBox: '0 0 16 16',
    body: '<path d="M8 3v9M4.5 8.5L8 12l3.5-3.5"/>',
  },
  out: {
    viewBox: '0 0 16 16',
    body: '<path d="M6.5 3.5h-3v9h9v-3"/><path d="M9.5 2.5h4v4M13 3L8.2 7.8"/>',
  },
  app: {
    viewBox: '0 0 16 16',
    body: '<rect x="2.5" y="2.5" width="11" height="11" rx="1.6"/><path d="M2.5 6.2h11"/><path d="M6 6.2v7.3"/>',
  },
  full: {
    viewBox: '0 0 16 16',
    body: '<path d="M6 2.5H2.5V6M10 2.5h3.5V6M6 13.5H2.5V10M10 13.5h3.5V10"/>',
  },
  git: {
    viewBox: '0 0 16 16',
    body: '<path d="M4 2.2a1.9 1.9 0 0 1 .75 3.65v4.3a1.9 1.9 0 1 1-1.5 0v-4.3A1.9 1.9 0 0 1 4 2.2zm8 2a1.9 1.9 0 0 1 .78 3.63c-.18 2.3-1.9 3.4-4.1 3.9l-.98.23-.5-1.44 1.1-.26c1.9-.44 2.8-1.2 2.96-2.44A1.9 1.9 0 0 1 12 4.2z"/>',
  },
  key: {
    viewBox: '0 0 16 16',
    body: '<circle cx="5.4" cy="5.4" r="3"/><path d="M7.5 7.5L13 13M11.2 11.2l1.3-1.3M9.4 9.4l1.3-1.3"/>',
  },
  link: {
    viewBox: '0 0 16 16',
    body: '<path d="M6.8 9.2a2.4 2.4 0 0 0 3.4 0l2-2a2.4 2.4 0 0 0-3.4-3.4l-1 1"/><path d="M9.2 6.8a2.4 2.4 0 0 0-3.4 0l-2 2a2.4 2.4 0 0 0 3.4 3.4l1-1"/>',
  },
  pip: {
    viewBox: '0 0 16 16',
    body: '<rect x="2.5" y="3" width="11" height="10" rx="1.4"/><rect x="8" y="8" width="4.2" height="3.4" rx="0.8" fill="currentColor" stroke="none"/>',
  },
  plug: {
    viewBox: '0 0 16 16',
    body: '<path d="M5.2 2.6v2.6M10.8 2.6v2.6"/><path d="M3.6 5.2h8.8v2.1a4.4 4.4 0 0 1-8.8 0z"/><path d="M8 11.7v2.6"/>',
  },
  sparkle: {
    viewBox: '0 0 16 16',
    body: '<path d="M8 2.2l1.3 3.5L12.8 7 9.3 8.3 8 11.8 6.7 8.3 3.2 7l3.5-1.3z" fill="currentColor" stroke="none"/><path d="M12.4 2.2l.5 1.4 1.4.5-1.4.5-.5 1.4-.5-1.4-1.4-.5 1.4-.5z" fill="currentColor" stroke="none"/>',
  },
  tool: {
    viewBox: '0 0 16 16',
    body: '<path d="M10.8 2.4a3 3 0 0 0-3.9 3.9L2.6 10.6a1.5 1.5 0 0 0 2.1 2.1l4.3-4.3a3 3 0 0 0 3.9-3.9L11 6.4l-1.7-.5-.5-1.7z"/>',
  },
  // Settings trigger.
  gear: {
    viewBox: '0 0 16 16',
    body: '<circle cx="8" cy="8" r="2.2"/><path d="M8 2.2v1.8M8 12v1.8M2.2 8H4M12 8h1.8M3.9 3.9l1.3 1.3M10.8 10.8l1.3 1.3M12.1 3.9l-1.3 1.3M5.2 10.8l-1.3 1.3"/>',
  },
  // Camera affordance for the composer's mobile photo-capture path.
  camera: {
    viewBox: '0 0 16 16',
    body: '<rect x="2" y="4.6" width="12" height="8" rx="1.6"/><circle cx="8" cy="8.8" r="2.3"/><path d="M5.6 4.6l1-1.6h2.8l1 1.6"/>',
  },
} satisfies Record<string, IconDef>;

export type IconName = keyof typeof ICONS;
