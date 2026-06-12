import { get, post } from '../api.js';
import { toast, formatRelativeTime, jobCriteriaStates } from '../utils.js';

const STATE_META = {
  queued:    { color: 'var(--overlay0)', word: 'queued' },
  running:   { color: 'var(--green)',    word: 'running' },
  verifying: { color: 'var(--yellow)',   word: 'verifying' },
  done:      { color: 'var(--green)',    word: 'done' },
  stuck:     { color: 'var(--peach)',    word: 'stuck' },
  errored:   { color: 'var(--red)',      word: 'error' },
  cancelled: { color: 'var(--overlay0)', word: 'cancelled' },
};

// Group definitions shared by the board columns, the summary pills, and the
// pill-driven filter. Each entry's `key` is the pill id; `states` is the set
// of Job.state values that belong to the group. The 'all' pill is implicit
// (no filter, total count).
const GROUPS = [
  { key: 'running', label: 'active',    color: 'var(--green)',    states: ['running', 'verifying'] },
  { key: 'queued',  label: 'queued',    color: 'var(--overlay0)', states: ['queued'] },
  { key: 'needs',   label: 'needs you', color: 'var(--peach)',    states: ['stuck', 'errored'] },
  { key: 'done',    label: 'resolved',  color: 'var(--green)',    states: ['done', 'cancelled'] },
];
const GROUPS_BY_KEY = Object.fromEntries(GROUPS.map((g) => [g.key, g]));
const STUCK_STATES = GROUPS_BY_KEY.needs.states;

function shellQuote(text) {
  // Match the worker prompt's expected escaping - quote with " and escape
  // embedded quotes. Simple enough for the preview to read accurately.
  return '"' + String(text || '').replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
}

function relativeTime(iso) {
  // Compact variant of the shared formatter for the dense jobs table.
  return formatRelativeTime(iso).replace(' ago', '') || '-';
}

function defaultLine(job) {
  // Map a Job's current state into the single activity-line the mini card
  // shows under the title.
  if (job.state === 'running') {
    return { kind: 'tool', text: 'working' + (job.verify_attempts ? ' · att ' + (job.verify_attempts + 1) : '') };
  }
  if (job.state === 'verifying') {
    const ac = (job.acceptance_criteria || []).length;
    return { kind: 'verify', text: 'verifier · checking' + (ac ? ' ' + ac + ' criteria' : '') };
  }
  if (job.state === 'queued') return { kind: 'queue', text: 'queued' };
  if (job.state === 'stuck') return { kind: 'issue', text: job.error || 'needs you' };
  if (job.state === 'errored') return { kind: 'error', text: job.error || 'worker crashed' };
  if (job.state === 'done') {
    // Use the real per-criterion verdicts, not a fabricated all-pass count - a
    // mark-done override can land DONE with some ACs still failing, and the
    // activity line must agree with the AC chips.
    const states = jobCriteriaStates(job);
    if (!states.length) return { kind: 'verdict', text: 'done' };
    const pass = states.filter((s) => s.status === 'pass').length;
    return { kind: 'verdict', text: `verifier pass · ${pass}/${states.length}` };
  }
  if (job.state === 'cancelled') return { kind: 'cancel', text: 'cancelled' };
  return { kind: 'queue', text: job.state };
}

function acStatesFor(job) {
  // Real per-criterion verdicts from job.ac_results, shared with the
  // conversation tile so the two surfaces can't disagree.
  return jobCriteriaStates(job).map((c) => c.status);
}

function jobTitle(job) {
  return (job.prompt || '').split('\n')[0] || '(no prompt)';
}

function jobConv(job) {
  return job.parent_session_id ? job.parent_session_id.slice(0, 8) : '-';
}

function jobAttempts(job) {
  const attempts = (job.attempts || []).length;
  if (attempts) return attempts;
  if (job.verify_attempts) return job.verify_attempts;
  return job.state === 'queued' ? 0 : 1;
}

function emptyNewJobForm() {
  return {
    prompt: '',
    agent: '',
    acs: [],
    maxAttempts: 3,
  };
}

function parseFilterText(text) {
  // Lightweight token parser: `state:running #287 foo` -> { states, terms }.
  // Anything unrecognised drops into the free-text matcher. Matches the
  // contract the filter input placeholder advertises.
  const out = { states: [], agents: [], terms: [] };
  for (const tok of (text || '').trim().split(/\s+/).filter(Boolean)) {
    if (tok.startsWith('state:')) {
      out.states.push(tok.slice(6).toLowerCase());
    } else if (tok.startsWith('agent:')) {
      out.agents.push(tok.slice(6).toLowerCase());
    } else if (tok.startsWith('#')) {
      out.terms.push(tok.slice(1).toLowerCase());
    } else {
      out.terms.push(tok.toLowerCase());
    }
  }
  return out;
}

export default () => ({
  jobs: [],
  loading: true,
  error: null,
  layout: localStorage.getItem('tsugite-jobs-layout') || 'board',
  activeFilter: 'all',
  filterText: '',
  newForm: emptyNewJobForm(),
  newError: null,
  submitting: false,

  init() {
    this.load();
    this._unwatch = this.$watch('$store.app.lastEvent', (e) => {
      if (!e) return;
      // SSE job_update events carry the COMPLETE job payload (Job.to_payload),
      // so patch it in place instead of refetching the whole list per event.
      const t = e.type || e.event_type;
      if (t === 'job_update' || t === 'job_status') {
        const payload = e.data;
        if (payload && payload.job_id) this.upsertJob(payload);
        else this.load();
      } else if (t === 'reconnect') {
        this.load();
      }
    });
  },

  destroy() {
    if (this._unwatch) this._unwatch();
  },

  async load() {
    this.error = null;
    try {
      const data = await get('/api/jobs');
      this.jobs = data.jobs || [];
      this.syncBadge();
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  },

  upsertJob(payload) {
    const i = this.jobs.findIndex((j) => j.job_id === payload.job_id);
    if (i >= 0) this.jobs.splice(i, 1, payload);
    else this.jobs.unshift(payload);
    this.syncBadge();
  },

  syncBadge() {
    this.$store.app.jobsNeedsYou = this.needsYouCount;
  },

  setLayout(layout) {
    this.layout = layout;
    localStorage.setItem('tsugite-jobs-layout', layout);
  },

  setActiveFilter(key) {
    this.activeFilter = key;
  },

  stateMeta(state) {
    return STATE_META[state] || { color: 'var(--overlay0)', word: state || 'unknown' };
  },

  // ---- derived filtering ------------------------------------------------
  filterJobsByPill(rows) {
    if (this.activeFilter === 'all') return rows;
    const group = GROUPS_BY_KEY[this.activeFilter];
    return group ? rows.filter((j) => group.states.includes(j.state)) : rows;
  },

  filterJobsByText(rows) {
    const parsed = parseFilterText(this.filterText);
    if (!parsed.states.length && !parsed.agents.length && !parsed.terms.length) return rows;
    return rows.filter((j) => {
      if (parsed.states.length && !parsed.states.includes(j.state)) return false;
      if (parsed.agents.length && !parsed.agents.includes((j.agent || '').toLowerCase())) return false;
      if (parsed.terms.length) {
        const haystack = [
          j.job_id || '',
          j.prompt || '',
          j.agent || '',
          j.parent_session_id || '',
        ].join(' ').toLowerCase();
        if (!parsed.terms.every((t) => haystack.includes(t))) return false;
      }
      return true;
    });
  },

  get filteredJobs() {
    return this.filterJobsByText(this.filterJobsByPill(this.jobs));
  },

  get boardColumns() {
    return GROUPS.map((group) => ({
      ...group,
      items: this.filteredJobs.filter((j) => group.states.includes(j.state)),
    }));
  },

  get summaryPills() {
    const countIn = (states) => this.jobs.filter((j) => states.includes(j.state)).length;
    return [
      { k: 'all', label: 'all', color: null, count: this.jobs.length },
      ...GROUPS.map((g) => ({ k: g.key, label: g.label, color: g.color, count: countIn(g.states) })),
    ];
  },

  get hasAnyJobs() {
    return this.jobs.length > 0;
  },

  get needsYouCount() {
    return this.jobs.filter((j) => STUCK_STATES.includes(j.state)).length;
  },

  // ---- per-job render helpers -------------------------------------------
  jobTitle,
  jobConv,
  jobAttempts,
  defaultLine,
  acStatesFor,
  relativeTime,
  acMark(s) { return s === 'pass' ? '✓' : s === 'fail' ? '✗' : s === 'active' ? '◔' : '○'; },
  acPassCount(states) { return states.filter((s) => s === 'pass').length; },

  rowActionsFor(job) {
    if (job.state === 'stuck' || job.state === 'errored') {
      return [{ key: 'retry', glyph: '↻', primary: true, title: 'retry / restart' }];
    }
    if (job.state === 'done' || job.state === 'cancelled') {
      return [{ key: 'open', glyph: '⤢', title: 'open parent session' }];
    }
    return [{ key: 'cancel', glyph: '✗', title: 'cancel' }];
  },

  openJobOriginSession(job) {
    if (!job.parent_session_id) return;
    this.$store.app.viewSessionId = job.parent_session_id;
    this.$store.app.view = 'conversations';
    location.hash = 'conversations?session=' + encodeURIComponent(job.parent_session_id);
  },

  async invokeRowAction(job, key) {
    if (key === 'cancel') {
      if (!confirm(`Cancel job ${job.job_id}?`)) return;
      try {
        await post(`/api/jobs/${job.job_id}/cancel`, { reason: 'cancelled by user' });
        await this.load();
      } catch (e) {
        toast('Cancel failed: ' + e.message, 'error');
      }
      return;
    }
    if (key === 'open') {
      this.openJobOriginSession(job);
      return;
    }
    if (key === 'retry') {
      const hint = prompt('Retry hint (sent to the worker):');
      if (!hint || !hint.trim()) return;
      try {
        await post(`/api/jobs/${job.job_id}/retry`, { hint: hint.trim() });
        await this.load();
      } catch (e) {
        toast('Retry failed: ' + e.message, 'error');
      }
    }
  },

  // ---- new-job modal ----------------------------------------------------
  openNewJob() {
    this.newForm = emptyNewJobForm();
    if (!this.newForm.agent) {
      const sel = this.$store.app.selectedAgent;
      const first = (this.$store.app.agents || [])[0];
      this.newForm.agent = sel || (first && first.name) || '';
    }
    this.newError = null;
    this.$store.tsu.open('new-job');
  },

  closeNewJob() {
    this.$store.tsu.close('new-job');
  },

  addAc() {
    this.newForm.acs.push({ text: '' });
  },

  removeAc(i) {
    this.newForm.acs.splice(i, 1);
  },

  stepAttempts(delta) {
    const next = Math.max(1, Math.min(9, this.newForm.maxAttempts + delta));
    this.newForm.maxAttempts = next;
  },

  get commandPreviewParts() {
    const f = this.newForm;
    const parts = [{ cls: 'cmd', text: '/job' }];
    if (f.agent) {
      parts.push({ text: ' ' }, { cls: 'flag', text: '--agent' }, { text: ' ' + f.agent });
    }
    parts.push({ text: ' ' }, { cls: 'flag', text: '--max-attempts' }, { text: ' ' + f.maxAttempts });
    for (const ac of (f.acs || []).filter((a) => a.text && a.text.trim())) {
      parts.push({ text: ' ' }, { cls: 'flag', text: '--ac' }, { text: ' ' }, { cls: 'str', text: shellQuote(ac.text) });
    }
    if (f.prompt && f.prompt.trim()) {
      parts.push({ text: ' ' }, { cls: 'str', text: shellQuote(f.prompt) });
    }
    return parts;
  },

  get canSubmit() {
    return !this.submitting && !!(this.newForm.prompt && this.newForm.prompt.trim()) && !!this.newForm.agent;
  },

  async submitNewJob() {
    if (!this.canSubmit) return;
    this.submitting = true;
    this.newError = null;
    const f = this.newForm;
    const ac_list = (f.acs || []).map((a) => a.text && a.text.trim()).filter(Boolean);
    const body = {
      user_id: this.$store.app.userId,
      prompt: f.prompt.trim(),
      max_attempts: f.maxAttempts,
    };
    if (ac_list.length) body.acceptance_criteria = ac_list.join('|');
    // No session_id: the new-job modal is never opened from inside a conversation,
    // so the backend provisions a fresh host session for the Job rather than
    // guessing which chat to attach it to (the old "wrong session" behaviour).
    try {
      // cmd_job failures raise CommandError server-side, which the endpoint maps
      // to a 400 - post() throws with the message, landing in the catch below.
      const data = await post(`/api/agents/${encodeURIComponent(f.agent)}/commands/job`, body);
      this.$store.tsu.close('new-job');
      toast(data.result || 'job spawned', 'success');
      await this.load();
    } catch (e) {
      this.newError = e.message;
      toast(e.message, 'error');
    } finally {
      this.submitting = false;
    }
  },
});
