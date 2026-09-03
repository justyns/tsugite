/**
 * Agent artifact panes: documents the agent asked the UI to show beside the chat.
 *
 * The daemon's `open_artifact` tool broadcasts one `session_event` frame with
 * `event_type: "artifact_open"`; the shell routes it here like every other
 * session event. This store owns what is open, keyed by the daemon's
 * `artifact_id` slot; the mux layout (`openBeside`) owns where.
 * `applySessionEvent` returns the artifact it recorded so the shell can hand it
 * straight to the layout.
 *
 * The tool sends the same id ("agent") for every open unless the agent asked for
 * a second pane, so a re-open replaces the record in place and bumps `rev`. A
 * mounted surface watches `rev`, which is what makes a second open of the very
 * same path still reload.
 *
 * Nothing here is persisted. An artifact pane is a live-session affordance; on
 * reload a path-backed artifact re-reads from the workspace API and an ephemeral
 * one is gone (the surface says so).
 */

/** `event_type` of the daemon frame. Frozen; matches tsugite/tools/artifacts.py. */
export const ARTIFACT_EVENT = 'artifact_open';

export type ArtifactContentType = 'markdown' | 'html' | 'text';
export type ArtifactMode = 'rendered' | 'source';
export type ArtifactPlacement = 'right' | 'below';

const CONTENT_TYPES: ArtifactContentType[] = ['markdown', 'html', 'text'];
const PLACEMENTS: ArtifactPlacement[] = ['right', 'below'];

export interface ArtifactOpen {
  /** Pane slot. Repeated opens of the same slot replace each other. */
  id: string;
  sessionId: string | null;
  /** Workspace-relative path, or null for generated content. */
  path: string | null;
  /** Inline generated content, or null when the surface should read `path`. */
  content: string | null;
  contentType: ArtifactContentType;
  mode: ArtifactMode;
  title: string;
  placement: ArtifactPlacement;
  /** Drives the "opened by the agent" badge on the pane. */
  openedByAgent: boolean;
}

export interface AgentArtifact extends ArtifactOpen {
  /** Bumped on every re-open of this slot, so a mounted surface reloads. */
  rev: number;
}

function str(value: unknown): string | null {
  return typeof value === 'string' && value !== '' ? value : null;
}

/**
 * Validate one `session_event` frame into an artifact open, or null when it is
 * not one (or is missing what a pane needs). Unknown enum values fall back to
 * the least-privileged option rather than being trusted through: an unknown
 * content type renders as text, and text always shows as source.
 */
export function parseArtifactOpen(data: Record<string, unknown>): ArtifactOpen | null {
  if (data.event_type !== ARTIFACT_EVENT) return null;
  const id = str(data.artifact_id);
  if (!id) return null;

  const path = str(data.path);
  const content = typeof data.content === 'string' ? data.content : null;
  if (path === null && content === null) return null;

  const declared = str(data.content_type) as ArtifactContentType | null;
  const contentType = declared && CONTENT_TYPES.includes(declared) ? declared : 'text';
  const mode: ArtifactMode =
    contentType !== 'text' && data.mode === 'rendered' ? 'rendered' : 'source';
  const placement = str(data.placement) as ArtifactPlacement | null;

  return {
    id,
    sessionId: str(data.session_id),
    path,
    content,
    contentType,
    mode,
    title: str(data.title) ?? path?.split('/').pop() ?? 'Artifact',
    placement: placement && PLACEMENTS.includes(placement) ? placement : 'right',
    openedByAgent: data.opened_by === 'agent',
  };
}

export class ArtifactsStore {
  /** Open artifacts by slot id. */
  items = $state<Record<string, AgentArtifact>>({});

  applySessionEvent(data: Record<string, unknown>): AgentArtifact | null {
    const open = parseArtifactOpen(data);
    if (!open) return null;
    const next = { ...open, rev: (this.items[open.id]?.rev ?? 0) + 1 };
    this.items[open.id] = next;
    return next;
  }

  get(id: string): AgentArtifact | undefined {
    return this.items[id];
  }

  close(id: string): void {
    delete this.items[id];
  }
}

export const artifacts = new ArtifactsStore();
