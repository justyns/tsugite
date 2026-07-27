/**
 * Pure helpers for the header model picker's grouped, metadata-rich rows.
 *
 * The picker groups the flat GET /api/models list under provider headers and
 * shows per-model context window + price + capability badges. Keeping the
 * grouping and number formatting here (out of the .svelte) makes them unit
 * testable and pins the option order the keyboard navigation walks: the flat
 * concatenation of the groups (see groupModelsByProvider).
 */

export interface PickerModel {
  id: string;
  provider?: string | null;
  context_window?: number | null;
  input_cost_per_million?: number | null;
  output_cost_per_million?: number | null;
  supports_vision?: boolean;
  supports_reasoning?: boolean;
}

export interface ModelGroup {
  provider: string;
  models: PickerModel[];
}

// Preferred providers sort to the top in this order; everything else follows
// alphabetically, with the null-provider "other" bucket always last.
const PROVIDER_ORDER = ['anthropic', 'openai', 'openrouter', 'ollama'];

function rank(provider: string): number {
  if (provider === 'other') return PROVIDER_ORDER.length + 1;
  const i = PROVIDER_ORDER.indexOf(provider);
  return i === -1 ? PROVIDER_ORDER.length : i;
}

/**
 * Group models under their provider, preserving each model's incoming order
 * within a group (the endpoint sorts by id) and ordering the groups by the
 * preferred list then alphabetically. A null/empty provider falls under a final
 * "other" group. Concatenating the groups' models in order reproduces the flat
 * option order the picker's keyboard navigation and selection index walk.
 */
export function groupModelsByProvider(models: PickerModel[]): ModelGroup[] {
  const byProvider = new Map<string, PickerModel[]>();
  for (const model of models) {
    const key = model.provider || 'other';
    const bucket = byProvider.get(key);
    if (bucket) bucket.push(model);
    else byProvider.set(key, [model]);
  }
  return [...byProvider.entries()]
    .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
    .map(([provider, group]) => ({ provider, models: group }));
}

function trimZeros(n: number): string {
  return n.toFixed(2).replace(/\.?0+$/, '');
}

/** Context window as a compact count: 200000 -> "200k", 1_000_000 -> "1M". */
export function formatContext(tokens?: number | null): string {
  if (!tokens || tokens <= 0) return '';
  if (tokens >= 1_000_000) return `${trimZeros(Math.round(tokens / 100_000) / 10)}M`;
  return `${Math.round(tokens / 1000)}k`;
}

function dollars(n: number): string {
  return `$${trimZeros(Math.round(n * 100) / 100)}`;
}

/**
 * Compact "input / output" price per 1M tokens, e.g. "$5 / $25". Empty when
 * either side is null (unpriced CLI / openai-compat models) so the row omits
 * price rather than showing a half figure.
 */
export function formatPrice(input?: number | null, output?: number | null): string {
  if (input == null || output == null) return '';
  return `${dollars(input)} / ${dollars(output)}`;
}
