/**
 * Resolving which agent a chat surface should act as for a given session.
 *
 * A deep link (the jobs board, a notification, ...) points the surface at a
 * session id and may carry an `agent` hint that is stale or plain wrong: the
 * jobs board passed the job's WORKER-agent name (e.g. job_worker), a builtin
 * agent file that is never a chat adapter. Trusting that hint pinned every
 * agent-scoped call (session list, effort levels, the send itself) to a
 * non-adapter and 404'd - surfaced to the user as "agent does not exist" when
 * they replied. So the surface heals: the session's OWN agent (read from its
 * record) always wins over the hint.
 */

export function resolveChatAgent(opts: {
  /** The session's true agent, from its record; null until resolved / unknown. */
  sessionAgent: string | null;
  /** The deep-link hint (params.agent); may be wrong. */
  paramAgent?: string;
  /** Last resort before anything is known (the roster's first agent). */
  fallbackAgent?: string;
}): string {
  return opts.sessionAgent ?? opts.paramAgent ?? opts.fallbackAgent ?? '';
}

/** A background job artifact (a worker or verifier session) carries the spawning
 *  job's id in its metadata. Its transcript is inspect-only: chatting into a
 *  worker/verifier mid- or post-job is a footgun - the conversation continues in
 *  the parent chat, never by injecting turns into the artifact. The parent host
 *  session carries `job_host` instead, so it is not caught here. */
export function isJobArtifact(metadata: Record<string, unknown> | null | undefined): boolean {
  return !!metadata && metadata['job_id'] != null;
}
