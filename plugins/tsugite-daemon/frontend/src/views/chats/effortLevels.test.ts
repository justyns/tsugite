import { beforeEach, expect, test, vi } from 'vitest';

vi.mock('$lib/api/client', () => ({
  api: { get: vi.fn() },
  authHeaders: () => ({}),
}));

import { api } from '$lib/api/client';
import { fetchEffortLevels, type EffortLevels } from './effortLevels';

const LEVELS: EffortLevels = {
  model: 'anthropic:claude-sonnet-4-5',
  supported_effort_levels: ['low', 'medium', 'high'],
};

beforeEach(() => vi.clearAllMocks());

test('two callers asking at once share one request', async () => {
  let settle: (value: EffortLevels) => void = () => {};
  vi.mocked(api.get).mockReturnValue(new Promise<EffortLevels>((res) => (settle = res)));

  const chip = fetchEffortLevels('s1');
  const seg = fetchEffortLevels('s1');
  settle(LEVELS);

  expect(await chip).toBe(LEVELS);
  expect(await seg).toBe(LEVELS);
  expect(vi.mocked(api.get)).toHaveBeenCalledTimes(1);
});

test('a caller after the request settles gets a fresh one', async () => {
  vi.mocked(api.get).mockResolvedValue(LEVELS);

  await fetchEffortLevels('s1');
  await fetchEffortLevels('s1');

  expect(vi.mocked(api.get)).toHaveBeenCalledTimes(2);
});
