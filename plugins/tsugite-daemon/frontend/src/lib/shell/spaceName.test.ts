import { expect, test } from 'vitest';
import { nextSpaceName } from './spaceName';

test('numbers from 2, past the seeded first space', () => {
  expect(nextSpaceName(['Main'])).toBe('Space 2');
  expect(nextSpaceName(['Main', 'Space 2'])).toBe('Space 3');
});

test('fills the lowest gap left by a closed space', () => {
  expect(nextSpaceName(['Main', 'Space 3'])).toBe('Space 2');
});

test('ignores names that are not numbered spaces', () => {
  expect(nextSpaceName(['notes', 'review'])).toBe('Space 2');
});

test('compares trimmed, so a padded name still counts as taken', () => {
  expect(nextSpaceName(['  Space 2  '])).toBe('Space 3');
});
