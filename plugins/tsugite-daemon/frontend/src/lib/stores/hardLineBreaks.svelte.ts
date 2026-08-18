/**
 * Whether a soft line break in the person's own message renders as a hard line
 * break, persisted to localStorage (`tsugite_hard_line_breaks`, default on).
 * Per-device like the other rendering preferences.
 */
import { booleanPref } from './booleanPref.svelte';

export const hardLineBreaks = booleanPref('tsugite_hard_line_breaks', true);
