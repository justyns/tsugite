/**
 * Jobs store: the Jobs board's backing data. Lists GET /api/jobs (newest first),
 * folds live `job_update` broadcasts (each carries a full Job payload) by id, and
 * owns the cancel / mark-done / retry mutations plus the executor list the
 * new-job composer needs. Board grouping runs through the pure jobsFilter
 * helpers; the board's own filter lives in the route, not here.
 *
 * Two distinct envelopes carry a Job: the global broadcast
 * `{type:'job_update', data:<payload>}` (this store) and the per-session history
 * item `{type:'job_status', ...payload}` (the in-chat tile, consumed elsewhere).
 * This store only handles the broadcast shape. Exported as a class instance.
 */
import { api } from '$lib/api/client';
import { groupCounts, type JobGroup, type JobLike } from './jobsFilter';

export interface JobAttempt {
  index: number;
  kind: string;
  worker_session_id: string | null;
  verifier_session_id: string | null;
  verifier_pass: boolean | null;
  model: string | null;
}

export interface JobAcResult {
  ac_index: number;
  ac_text: string;
  pass: boolean;
  reason: string;
  attempt: number;
}

export interface Job extends JobLike {
  job_id: string;
  parent_session_id: string | null;
  worker_session_id: string | null;
  verifier_session_id: string | null;
  state: string;
  prompt: string;
  verify_attempts: number;
  max_attempts: number;
  notify_when: string | null;
  error: string | null;
  error_detail: string | null;
  pending_question: string | null;
  attempts: JobAttempt[];
  acceptance_criteria: string[];
  ac_results: JobAcResult[];
  result: Record<string, unknown> | null;
  agent: string;
  model: string | null;
  effort: string | null;
  model_ladder: string[] | null;
  ladder_index: number | null;
  verifier_model: string | null;
  repo: string | null;
  created_at: string;
  updated_at: string;
  resolved_at: string | null;
  spawned_by: string | null;
  executor: string;
  worker_terminal_id: string | null;
}

export interface JobRetryOpts {
  hint?: string;
  model?: string;
  verifierModel?: string;
  resetCounter?: boolean;
  freshWorkspace?: boolean;
}

export interface TerminalRef {
  id: string;
  [key: string]: unknown;
}

export class JobsStore {
  jobs = $state<Job[]>([]);
  executors = $state<string[]>(['agent']);
  loading = $state(false);
  error = $state<string | null>(null);

  get counts(): Record<JobGroup, number> {
    return groupCounts(this.jobs);
  }

  async load(opts: { state?: string; limit?: number } = {}): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const params = new URLSearchParams();
      if (opts.state) params.set('state', opts.state);
      if (opts.limit != null) params.set('limit', String(opts.limit));
      const qs = params.toString();
      const res = await api.get<{ jobs: Job[] }>(`/api/jobs${qs ? `?${qs}` : ''}`);
      this.jobs = res.jobs;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
    } finally {
      this.loading = false;
    }
  }

  async loadExecutors(): Promise<void> {
    try {
      const res = await api.get<{ executors: string[] }>('/api/executors');
      this.executors = res.executors;
    } catch {
      // Non-fatal: the composer just falls back to the built-in "agent".
      this.executors = ['agent'];
    }
  }

  /** Fold a `job_update` broadcast (a full Job payload) in by id: replace in
   *  place when known, else prepend as the newest. */
  applyJobUpdate(data: Record<string, unknown>): void {
    const job = data as unknown as Job;
    if (!job.job_id) return;
    const idx = this.jobs.findIndex((j) => j.job_id === job.job_id);
    if (idx === -1) {
      this.jobs = [job, ...this.jobs];
    } else {
      const next = this.jobs.slice();
      next[idx] = job;
      this.jobs = next;
    }
  }

  async cancel(jobId: string, reason?: string): Promise<void> {
    await api.post(
      `/api/jobs/${encodeURIComponent(jobId)}/cancel`,
      reason ? { reason } : undefined,
    );
  }

  async markDone(jobId: string, reason?: string): Promise<void> {
    await api.post(
      `/api/jobs/${encodeURIComponent(jobId)}/mark-done`,
      reason ? { reason } : undefined,
    );
  }

  /** Retry a stuck/errored job. The backend requires a hint OR a model. */
  async retry(jobId: string, opts: JobRetryOpts): Promise<void> {
    const body: Record<string, unknown> = {};
    if (opts.hint) body.hint = opts.hint;
    if (opts.model) body.model = opts.model;
    if (opts.verifierModel) body.verifier_model = opts.verifierModel;
    if (opts.resetCounter) body.reset_counter = true;
    if (opts.freshWorkspace) body.fresh_workspace = true;
    await api.post(`/api/jobs/${encodeURIComponent(jobId)}/retry`, body);
  }

  /** Resolve the live terminal for a job's worker: prefer the direct
   *  worker_terminal_id (cc-executor jobs), else probe the parent-session link.
   *  Returns the terminal id, or null when the job has no terminal. */
  async terminalIdForJob(job: Job): Promise<string | null> {
    if (job.worker_terminal_id) return job.worker_terminal_id;
    if (!job.worker_session_id) return null;
    const params = new URLSearchParams({ parent_session_id: job.worker_session_id });
    try {
      const res = await api.get<{ terminals: TerminalRef[] }>(
        `/api/terminals/?${params.toString()}`,
      );
      return res.terminals[0]?.id ?? null;
    } catch {
      return null;
    }
  }
}

export const jobs = new JobsStore();
