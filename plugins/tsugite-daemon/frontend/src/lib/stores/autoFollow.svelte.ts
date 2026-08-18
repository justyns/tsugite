/**
 * Whether the conversation keeps itself pinned to the newest output as it
 * streams, persisted per-device to localStorage (`tsugite_auto_follow`,
 * default on).
 */
import { booleanPref } from './booleanPref.svelte';

export const autoFollow = booleanPref('tsugite_auto_follow', true);
