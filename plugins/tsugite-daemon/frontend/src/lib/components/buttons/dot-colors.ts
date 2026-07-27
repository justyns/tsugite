// The full semantic state-color set from tokens.css
// (--st-ok/-verify/-warn/-err/-info/-queue/-mute). Shared by Dot.svelte and
// its gallery so "every status color" stays a single source of truth.
export type DotColor = 'ok' | 'verify' | 'warn' | 'err' | 'info' | 'queue' | 'mute';

export const DOT_COLORS: readonly DotColor[] = [
  'ok',
  'verify',
  'warn',
  'err',
  'info',
  'queue',
  'mute',
];
