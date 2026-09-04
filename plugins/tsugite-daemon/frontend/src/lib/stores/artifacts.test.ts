import { describe, expect, test } from 'vitest';
import {
  ArtifactsStore,
  parseArtifactOpen,
  ARTIFACT_EVENT,
  artifactSurfaceParams,
} from './artifacts.svelte';

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
    expect(store.get('agent', 'sess-1')?.title).toBe('notes.md');
    expect(store.get('agent', 'sess-1')?.rev).toBe(1);
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

    expect(Object.keys(store.items)).toEqual(['sess-1:agent']);
    const cur = store.get('agent', 'sess-1')!;
    expect(cur.path).toBe('reports/cov.html');
    expect(cur.contentType).toBe('html');
    expect(cur.title).toBe('Cov');
    expect(cur.rev).toBe(2);
  });

  test('a different slot is kept alongside the first', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame());
    store.applySessionEvent(frame({ artifact_id: 'x2', title: 'Second' }));

    expect(Object.keys(store.items).sort()).toEqual(['sess-1:agent', 'sess-1:x2']);
    expect(store.get('x2', 'sess-1')?.rev).toBe(1);
  });

  test('agent opens with the same slot are scoped by session', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame({ session_id: 'sess-a', path: 'ops/alpha.md', title: 'Alpha' }));
    store.applySessionEvent(frame({ session_id: 'sess-b', path: 'ops/beta.md', title: 'Beta' }));

    expect(Object.keys(store.items).sort()).toEqual(['sess-a:agent', 'sess-b:agent']);
    expect(store.get('agent', 'sess-a')?.title).toBe('Alpha');
    expect(store.get('agent', 'sess-a')?.path).toBe('ops/alpha.md');
    expect(store.get('agent', 'sess-b')?.title).toBe('Beta');
    expect(store.agentForSession('sess-a')?.title).toBe('Alpha');
    expect(store.agentForSession('sess-b')?.title).toBe('Beta');
  });

  test('repeat opens reuse only that chat pane slot', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame({ session_id: 'sess-a', path: 'ops/alpha.md', title: 'Alpha' }));
    store.applySessionEvent(frame({ session_id: 'sess-b', path: 'ops/beta.md', title: 'Beta' }));
    store.applySessionEvent(
      frame({ session_id: 'sess-a', path: 'reports/cov.html', title: 'Cov' }),
    );

    expect(Object.keys(store.items).sort()).toEqual(['sess-a:agent', 'sess-b:agent']);
    expect(store.get('agent', 'sess-a')?.title).toBe('Cov');
    expect(store.get('agent', 'sess-a')?.rev).toBe(2);
    expect(store.get('agent', 'sess-b')?.title).toBe('Beta');
    expect(store.get('agent', 'sess-b')?.rev).toBe(1);
  });

  test('user-opened artifacts remain window-level by slot', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame({ session_id: 'sess-a', title: 'A', opened_by: 'user' }));
    store.applySessionEvent(frame({ session_id: 'sess-b', title: 'B', opened_by: 'user' }));

    expect(Object.keys(store.items)).toEqual(['agent']);
    expect(store.get('agent')?.title).toBe('B');
    expect(store.get('agent')?.rev).toBe(2);
    expect(store.agentArtifacts()).toEqual([]);
  });

  test('a scoped lookup with no record never falls back to the bare slot', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame({ session_id: null, title: 'Unscoped' }));

    expect(store.items.agent.title).toBe('Unscoped');
    expect(store.get('agent', 'sess-a')).toBeUndefined();
  });

  test('surface params distinguish agent panes by session but not user-opened panes', () => {
    const store = new ArtifactsStore();
    const agent = store.applySessionEvent(frame({ session_id: 'sess-a' }))!;
    const user = store.applySessionEvent(frame({ artifact_id: 'manual', opened_by: 'user' }))!;

    expect(artifactSurfaceParams(agent)).toEqual({ id: 'agent', sessionId: 'sess-a' });
    expect(artifactSurfaceParams(user)).toEqual({ id: 'manual' });
  });

  test('close drops one slot and leaves the others', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame());
    store.applySessionEvent(frame({ artifact_id: 'x2' }));

    store.close('agent', 'sess-1');

    expect(Object.keys(store.items)).toEqual(['sess-1:x2']);
    expect(store.get('agent', 'sess-1')).toBeUndefined();
  });

  test('closing an unknown slot is a no-op', () => {
    const store = new ArtifactsStore();
    store.applySessionEvent(frame());
    store.close('nope');
    expect(Object.keys(store.items)).toEqual(['sess-1:agent']);
  });
});
