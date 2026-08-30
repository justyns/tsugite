/**
 * Whether a reasoning block in the conversation starts expanded, persisted
 * per-device to localStorage (`tsugite_expand_thinking`, default on).
 */
import { booleanPref } from './booleanPref.svelte';

export const expandThinking = booleanPref('tsugite_expand_thinking', true);
