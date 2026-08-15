import { expect, test } from 'vitest';
import { chatRouteParams } from './chatLink';

test('a chat deep link carries the session and its agent', () => {
  expect(chatRouteParams('s1', 'smokeagent')).toEqual({ sessionId: 's1', agent: 'smokeagent' });
});

test('an unset agent is left out rather than sent as an empty param', () => {
  expect(chatRouteParams('s1', '')).toEqual({ sessionId: 's1' });
  expect(chatRouteParams('s1')).toEqual({ sessionId: 's1' });
});
