import { get, post } from '../api.js';
import { toast } from '../utils.js';

const STATE_META = {
  queued:    { color: 'var(--overlay0)', word: 'queued' },
  running:   { color: 'var(--green)',    word: 'running' },
  verifying: { color: 'var(--yellow)',   word: 'verifying' },
  done:      { color: 'var(--green)',    word: 'done' },
  stuck:     { color: 'var(--peach)',    word: 'stuck' },
  errored:   { color: 'var(--red)',      word: 'error' },
  cancelled: { color: 'var(--overlay0)', word: 'cancelled' },
};

const COLUMNS = [
  { key: 'active', label: 'active',    color: 'var(--green)',    states: ['running', 'verifying'] },
  { key: 'queued', label: 'queued',    color: 'var(--overlay0)', states: ['queued'] },
  { key: 'needs',  label: 'needs you', color: 'var(--peach)',    states: ['stuck', 'errored'] },
  { key: 'done',   label: 'resolved',  color: 'var(--green)',    states: ['done', 'cancelled'] },
];

const AC_KINDS = ['test', 'ui', 'cmd', 'llm'];

function shellQuote(text) {
  // Match the worker prompt's expected escaping — quote with " and escape
  // embedded quotes. Simple enough for the preview to read accurately.
  return '"' + String(text || '').replace(/\\/g, '\\\\').replace(/"/g, '\\"') + '"';
}

function relativeTime(iso) {
  if (!iso) return '—';
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return '—';
  const delta = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (delta < 60) return delta + 's';
  if (delta < 3600) return Math.floor(delta / 60) + 'm';
  if (delta < 86400) return Math.floor(delta / 3600) + 'h';
  return Math.floor(delta / 86400) + 'd';
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
    const acs = job.acceptance_criteria || [];
    return { kind: 'verdict', text: acs.length ? `verifier pass · ${acs.length}/${acs.length}` : 'done' };
  }
  if (job.state === 'cancelled') return { kind: 'cancel', text: 'cancelled' };
  return { kind: 'queue', text: job.state };
}

function acStatesFor(job) {
  // Each acceptance_criterion is just a string; the API doesn't carry per-criterion
  // verdicts yet. Render terminal-state assumptions so the mini chips read truthfully.
  const acs = job.acceptance_criteria || [];
  if (!acs.length) return [];
  if (job.state === 'done') return acs.map(() => 'pass');
  if (job.state === 'stuck' || job.state === 'errored') return acs.map(() => 'pending');
  if (job.state === 'verifying') return acs.map((_, i) => (i === 0 ? 'active' : 'pending'));
  return acs.map(() => 'pending');
}

function jobTitle(job) {
  return (job.prompt || '').split('\n')[0] || '(no prompt)';
}

function jobConv(job) {
  return job.parent_session_id ? job.parent_session_id.slice(0, 8) : '—';
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
    verify: true,
    background: true,
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
  showNew: false,
  newForm: emptyNewJobForm(),
  newError: null,
  submitting: false,
  COLUMNS,
  STATE_META,
  AC_KINDS,

  init() {
    this.load();
    this._unwatch = this.$watch('$store.app.lastEvent', (e) => {
      if (!e) return;
      // SSE job_update events fire whenever the orchestrator mutates a Job;
      // refresh so the board/table reflect the new state without a manual poll.
      const t = e.type || e.event_type;
      if (t === 'job_update' || t === 'job_status' || t === 'reconnect') {
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

  syncBadge() {
    const stuck = this.jobs.filter((j) => j.state === 'stuck' || j.state === 'errored').length;
    this.$store.app.jobsNeedsYou = stuck;
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
    if (this.activeFilter === 'running') return rows.filter((j) => j.state === 'running' || j.state === 'verifying');
    if (this.activeFilter === 'queued') return rows.filter((j) => j.state === 'queued');
    if (this.activeFilter === 'needs') return rows.filter((j) => j.state === 'stuck' || j.state === 'errored');
    if (this.activeFilter === 'done') return rows.filter((j) => j.state === 'done' || j.state === 'cancelled');
    return rows;
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
    return COLUMNS.map((col) => ({
      ...col,
      items: this.filteredJobs.filter((j) => col.states.includes(j.state)),
    }));
  },

  get summaryPills() {
    const n = (pred) => this.jobs.filter(pred).length;
    return [
      { k: 'all', label: 'all', color: null, count: this.jobs.length },
      { k: 'running', label: 'active', color: 'var(--green)', count: n((j) => j.state === 'running' || j.state === 'verifying') },
      { k: 'queued', label: 'queued', color: 'var(--overlay0)', count: n((j) => j.state === 'queued') },
      { k: 'needs', label: 'needs you', color: 'var(--peach)', count: n((j) => j.state === 'stuck' || j.state === 'errored') },
      { k: 'done', label: 'resolved', color: 'var(--green)', count: n((j) => j.state === 'done' || j.state === 'cancelled') },
    ];
  },

  get hasAnyJobs() {
    return this.jobs.length > 0;
  },

  get needsYouCount() {
    return this.jobs.filter((j) => j.state === 'stuck' || j.state === 'errored').length;
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
    this.showNew = true;
  },

  closeNewJob() {
    this.showNew = false;
  },

  addAc() {
    this.newForm.acs.push({ kind: 'test', text: '' });
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
    if (f.verify) {
      parts.push({ text: ' ' }, { cls: 'flag', text: '--verify' });
    }
    if (!f.background) {
      parts.push({ text: ' ' }, { cls: 'flag', text: '--fg' });
    }
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
    };
    if (ac_list.length) body.acceptance_criteria = ac_list.join('|');
    // Foreground vs background isn't an exposed --fg flag on the backend cmd today;
    // jobs always run as background workers. We persist the toggle for parity with
    // the design's command preview but don't send it on the wire.
    const activeSession = this.$store.app.viewSessionId;
    if (activeSession) body.session_id = activeSession;
    try {
      const data = await post(`/api/agents/${encodeURIComponent(f.agent)}/commands/job`, body);
      this.showNew = false;
      toast(data.result || 'job spawned', 'success');
      await this.load();
    } catch (e) {
      this.newError = e.message;
    } finally {
      this.submitting = false;
    }
  },
});
