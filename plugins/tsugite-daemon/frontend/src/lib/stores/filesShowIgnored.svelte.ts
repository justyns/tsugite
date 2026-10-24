/**
 * Whether the Files tree lists gitignored paths (dimmed), persisted to
 * localStorage (`tsugite_files_show_ignored`, default off). Per-device.
 */
import { booleanPref } from './booleanPref.svelte';

export const filesShowIgnored = booleanPref('tsugite_files_show_ignored', false);
