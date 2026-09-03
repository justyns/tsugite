import { describe, expect, test } from 'vitest';
import { ArtifactsStore, parseArtifactOpen, ARTIFACT_EVENT } from './artifacts.svelte';

/** The frame `open_artifact` puts on the wire (tests/test_artifact_tools.py pins
 *  the Python half against this exact shape). */
function frame(extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    session_id: 'sess-1',
    event_type: ARTIFACT_EVENT,
    artifact_id: 'agent',
    path: 'notes.md',
    content: null,
    content_type: 'markdown',
    mode: 'rendered',
    title: 'notes.md',
    placement: 'right',
    opened_by: 'agent',
    ...extra,
  };
}

describe('parseArtifactOpen', () => {
  test('reads the daemon frame', () => {
    expect(parseArtifactOpen(frame())).toEqual({
      id: 'agent',
      sessionId: 'sess-1',
      path: 'notes.md',
      content: null,
      contentType: 'markdown',
      mode: 'rendered',
      title: 'notes.md',
      placement: 'right',
      openedByAgent: true,
    });
  });

  test('ignores any other session_event type', () => {
    expect(parseArtifactOpen(frame({ event_type: 'file_write' }))).toBeNull();
    expect(parseArtifactOpen({})).toBeNull();
  });

  test('ignores a frame with no artifact id', () => {
    expect(parseArtifactOpen(frame({ artifact_id: '' }))).toBeNull();
    expect(parseArtifactOpen(frame({ artifact_id: undefined }))).toBeNull();
  });

  test('ignores a frame that names neither a path nor content', () => {
    expect(parseArtifactOpen(frame({ path: null, content: null }))).toBeNull();
  });

  test('falls back to safe values for an unknown content type, mode or placement', () => {
    const out = parseArtifactOpen(
      frame({ content_type: 'pdf', mode: 'edit', placement: 'floating' }),
    )!;
    expect(out.contentType).toBe('text');
    expect(out.mode).toBe('source');
    expect(out.placement).toBe('right');
  });

  test('plain text never claims a rendered mode', () => {
    expect(parseArtifactOpen(frame({ content_type: 'text', mode: 'rendered' }))!.mode).toBe(
      'source',
    );
  });

  test('carries ephemeral content and its missing path', () => {
    const out = parseArtifactOpen(
      frame({ path: null, content: '# hi', content_type: 'markdown' }),
    )!;
    expect(out.path).toBeNull();
    expect(out.content).toBe('# hi');
  });

  test('titles fall back to the file name, then to a generic label', () => {
    expect(parseArtifactOpen(frame({ title: '' }))!.title).toBe('notes.md');
    expect(parseArtifactOpen(frame({ title: '', path: 'a/b/c.html' }))!.title).toBe('c.html');
    expect(parseArtifactOpen(frame({ title: '', path: null, content: 'x' }))!.title).toBe(
      'Artifact',
    );
  });
});

describe('ArtifactsStore', () => {
  test('an open records the artifact and reports it to the caller', () => {
    const store = new ArtifactsStore();
    const opened = store.applySessionEvent(frame());

    expect(opened?.id).toBe('agent');
    expect(store.get('agent')?.title).toBe('notes.md');
    expect(store.get('agent')?.rev).toBe(1);
  });

  test('a non-artifact frame is ignored and changes nothing', () => {
    const store = new ArtifactsStore();
    expect(store.applySessionEvent({ session_id: 's', event_type: 'file_write' })).toBeNull();
    expect(Object.keys(store.items)).toEqual([]);
  });

  test('re-opening the same slot replaces it in place and bumps its revision', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame());
    store.applySessionEvent(
      frame({ path: 'reports/cov.html', content_type: 'html', title: 'Cov' }),
    );

    expect(Object.keys(store.items)).toEqual(['agent']);
    const cur = store.get('agent')!;
    expect(cur.path).toBe('reports/cov.html');
    expect(cur.contentType).toBe('html');
    expect(cur.title).toBe('Cov');
    expect(cur.rev).toBe(2);
  });

  test('a different slot is kept alongside the first', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame());
    store.applySessionEvent(frame({ artifact_id: 'x2', title: 'Second' }));

    expect(Object.keys(store.items)).toEqual(['agent', 'x2']);
    expect(store.get('x2')?.rev).toBe(1);
  });

  test('close drops one slot and leaves the others', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame());
    store.applySessionEvent(frame({ artifact_id: 'x2' }));

    store.close('agent');

    expect(Object.keys(store.items)).toEqual(['x2']);
    expect(store.get('agent')).toBeUndefined();
  });

  test('closing an unknown slot is a no-op', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame());
    store.close('nope');
    expect(Object.keys(store.items)).toEqual(['agent']);
  });
});
