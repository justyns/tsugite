import { afterEach, expect, test, vi } from 'vitest';
import { api } from '$lib/api/client';
import {
  captureServerContext,
  fetchServerChoices,
  fetchServerProviders,
  searchServerProvider,
} from './serverProviders';

afterEach(() => vi.restoreAllMocks());

test('fetchServerProviders normalizes has_choices and passes through a shipped icon', async () => {
  vi.spyOn(api, 'get').mockResolvedValue({
    providers: [
      { key: 'terminal', label: 'Terminal output', icon: 'term', has_choices: true },
      { key: 'webpage', label: 'Web page', icon: 'link', has_choices: false },
    ],
  });
  const res = await fetchServerProviders();
  expect(res).toEqual([
    {
      key: 'terminal',
      label: 'Terminal output',
      icon: 'term',
      hasChoices: true,
      picker: false,
      inMenu: true,
      autocompletePrefix: null,
    },
    {
      key: 'webpage',
      label: 'Web page',
      icon: 'link',
      hasChoices: false,
      picker: false,
      inMenu: true,
      autocompletePrefix: null,
    },
  ]);
});

test('fetchServerProviders normalizes the picker flag for a large-option provider', async () => {
  vi.spyOn(api, 'get').mockResolvedValue({
    providers: [
      { key: 'file', label: 'Workspace file', icon: 'file', has_choices: true, picker: true },
    ],
  });
  const [p] = await fetchServerProviders();
  expect(p).toEqual({
    key: 'file',
    label: 'Workspace file',
    icon: 'file',
    hasChoices: true,
    picker: true,
    inMenu: true,
    autocompletePrefix: null,
  });
});

test('fetchServerProviders surfaces an autocomplete source kept out of the menu', async () => {
  vi.spyOn(api, 'get').mockResolvedValue({
    providers: [
      {
        key: 'jira',
        label: 'Jira',
        icon: 'link',
        has_choices: false,
        in_menu: false,
        autocomplete_prefix: 'jira',
      },
    ],
  });
  const [p] = await fetchServerProviders();
  expect(p?.inMenu).toBe(false);
  expect(p?.autocompletePrefix).toBe('jira');
});

test('fetchServerProviders falls back to a generic glyph for an icon this build lacks', async () => {
  vi.spyOn(api, 'get').mockResolvedValue({
    providers: [{ key: 'weird', label: 'Weird', icon: 'no-such-icon', has_choices: false }],
  });
  const [p] = await fetchServerProviders();
  expect(p?.icon).toBe('sparkle');
});

test('fetchServerProviders swallows an unreachable daemon into an empty list', async () => {
  vi.spyOn(api, 'get').mockRejectedValue(new Error('connection refused'));
  expect(await fetchServerProviders()).toEqual([]);
});

test('fetchServerChoices scopes the request to the session and returns the options', async () => {
  const get = vi
    .spyOn(api, 'get')
    .mockResolvedValue({ choices: [{ value: 't1', label: 'npm test' }] });
  const res = await fetchServerChoices('terminal', 's1');
  expect(res).toEqual([{ value: 't1', label: 'npm test' }]);
  expect(get.mock.calls[0]?.[0]).toBe('/api/context-providers/terminal/choices?session_id=s1');
});

test('fetchServerChoices swallows an error into no options', async () => {
  vi.spyOn(api, 'get').mockRejectedValue(new Error('boom'));
  expect(await fetchServerChoices('terminal', 's1')).toEqual([]);
});

test('searchServerProvider scopes to the session + query and returns the results', async () => {
  const get = vi
    .spyOn(api, 'get')
    .mockResolvedValue({ results: [{ value: 'PROJ-1', label: 'auth flow' }] });
  const res = await searchServerProvider('jira', 's1', 'auth');
  expect(res).toEqual([{ value: 'PROJ-1', label: 'auth flow' }]);
  expect(get.mock.calls[0]?.[0]).toBe('/api/context-providers/jira/search?session_id=s1&q=auth');
});

test('searchServerProvider swallows an error into no results', async () => {
  vi.spyOn(api, 'get').mockRejectedValue(new Error('boom'));
  expect(await searchServerProvider('jira', 's1', 'auth')).toEqual([]);
});

test('captureServerContext posts the session + arg and returns the items', async () => {
  const post = vi
    .spyOn(api, 'post')
    .mockResolvedValue({ items: [{ key: 'terminal:t1', label: 'Terminal', value: 'output' }] });
  const items = await captureServerContext('terminal', 's1', 't1');
  expect(items).toEqual([{ key: 'terminal:t1', label: 'Terminal', value: 'output' }]);
  expect(post.mock.calls[0]?.[0]).toBe('/api/context-providers/terminal/capture');
  expect(post.mock.calls[0]?.[1]).toEqual({ session_id: 's1', arg: 't1' });
});

test('captureServerContext sends arg:null for a no-choices provider', async () => {
  const post = vi.spyOn(api, 'post').mockResolvedValue({ items: [] });
  await captureServerContext('webpage', 's1', null);
  expect(post.mock.calls[0]?.[1]).toEqual({ session_id: 's1', arg: null });
});

test('captureServerContext propagates a provider error so the caller can surface it', async () => {
  vi.spyOn(api, 'post').mockRejectedValue(new Error('no such terminal'));
  await expect(captureServerContext('terminal', 's1', 'gone')).rejects.toThrow('no such terminal');
});
