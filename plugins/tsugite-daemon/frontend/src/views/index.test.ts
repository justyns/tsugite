import { expect, test } from 'vitest';
import { views, viewById } from './index';

test('an unknown view id resolves to the landing view', () => {
  expect(viewById('not-a-view').id).toBe('chats');
  expect(views[0]?.id).toBe('chats');
});

test('the five rows the phone rail keeps', () => {
  // The rail hides everything past the fifth row at <=640px; those five are the
  // ones a phone gets without opening the palette.
  expect(views.slice(0, 5).map((v) => v.id)).toEqual([
    'chats',
    'terminals',
    'files',
    'jobs',
    'schedules',
  ]);
});
