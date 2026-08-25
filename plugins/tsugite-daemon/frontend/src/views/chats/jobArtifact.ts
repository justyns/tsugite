/** A background job artifact (a worker or verifier session) carries the spawning
 *  job's id in its metadata. Its transcript is inspect-only: chatting into a
 *  worker/verifier mid- or post-job is a footgun - the conversation continues in
 *  the parent chat, never by injecting turns into the artifact. The parent host
 *  session carries `job_host` instead, so it is not caught here. */
export function isJobArtifact(metadata: Record<string, unknown> | null | undefined): boolean {
  return !!metadata && metadata['job_id'] != null;
}
