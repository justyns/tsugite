/**
 * Whether this browser has skipped or finished the first-run welcome, persisted
 * per-device to localStorage (`tsugite_welcome_dismissed`, default off).
 */
import { booleanPref } from './booleanPref.svelte';

export const welcomeDismissed = booleanPref('tsugite_welcome_dismissed', false);
