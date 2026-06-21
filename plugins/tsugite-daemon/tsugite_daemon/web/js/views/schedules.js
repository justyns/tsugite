import { get, post, patch, del } from '../api.js';
import { formatDate, stateBadgeClass } from '../utils.js';

const emptyForm = () => ({ id: '', agent: '', prompt: '', schedule_type: 'cron', cron_expr: '', run_at: '', timezone: 'UTC', model: '', agent_file: '', max_turns: '', execution_type: 'agent', command: '', script_timeout: 60 });

export default () => ({
  schedules: [],
  showForm: false,
  loading: true,
  error: null,
  editingId: null,
  form: emptyForm(),
  expandedId: null,
  runSessions: {},

  init() {
    this.load();
  },

  async load() {
    this.error = null;
    this.runSessions = {};
    try {
      const data = await get('/api/schedules');
      this.schedules = data.schedules || [];
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  },

  async toggle(s) {
    try {
      await post(`/api/schedules/${s.id}/${s.enabled ? 'disable' : 'enable'}`);
      await this.load();
    } catch (e) {
      this.error = e.message;
    }
  },

  async runNow(s) {
    try {
      await post(`/api/schedules/${s.id}/run`);
      delete this.runSessions[s.id];
    } catch (e) {
      this.error = e.message;
    }
  },

  async remove(s) {
    if (!confirm(`Delete schedule "${s.id}"?`)) return;
    try {
      await del(`/api/schedules/${s.id}`);
      await this.load();
    } catch (e) {
      this.error = e.message;
    }
  },

  edit(s) {
    this.editingId = s.id;
    this.form = {
      id: s.id,
      agent: s.agent,
      prompt: s.prompt,
      schedule_type: s.schedule_type,
      cron_expr: s.cron_expr || '',
      run_at: s.run_at || '',
      timezone: s.timezone || 'UTC',
      model: s.model || '',
      agent_file: s.agent_file || '',
      max_turns: s.max_turns || '',
      execution_type: s.execution_type || 'agent',
      command: s.command || '',
      script_timeout: s.script_timeout || 60,
    };
    this.showForm = true;
  },

  cancelForm() {
    this.showForm = false;
    this.editingId = null;
    this.form = emptyForm();
  },

  async save() {
    this.error = null;
    if (this.editingId) {
      const body = {};
      body.execution_type = this.form.execution_type;
      if (this.form.execution_type === 'script') {
        if (this.form.command) body.command = this.form.command;
        if (this.form.script_timeout) body.script_timeout = parseInt(this.form.script_timeout, 10);
      } else {
        if (this.form.prompt) body.prompt = this.form.prompt;
        if (this.form.model) body.model = this.form.model;
        if (this.form.agent_file) body.agent_file = this.form.agent_file;
        if (this.form.max_turns) body.max_turns = parseInt(this.form.max_turns, 10);
      }
      if (this.form.schedule_type === 'cron' && this.form.cron_expr) body.cron_expr = this.form.cron_expr;
      if (this.form.schedule_type === 'once' && this.form.run_at) body.run_at = this.form.run_at;
      if (this.form.timezone) body.timezone = this.form.timezone;
      try {
        await patch(`/api/schedules/${this.editingId}`, body);
        this.cancelForm();
        await this.load();
      } catch (e) {
        this.error = e.message;
      }
    } else {
      const body = { ...this.form };
      if (body.schedule_type === 'cron') {
        delete body.run_at;
      } else {
        delete body.cron_expr;
      }
      if (body.execution_type === 'script') {
        delete body.prompt;
        delete body.model;
        delete body.agent_file;
        delete body.max_turns;
        if (body.script_timeout) body.script_timeout = parseInt(body.script_timeout, 10);
        else delete body.script_timeout;
      } else {
        delete body.command;
        delete body.script_timeout;
        if (!body.model) delete body.model;
        if (!body.agent_file) delete body.agent_file;
        if (body.max_turns) body.max_turns = parseInt(body.max_turns, 10);
        else delete body.max_turns;
      }
      try {
        await post('/api/schedules', body);
        this.cancelForm();
        await this.load();
      } catch (e) {
        this.error = e.message;
      }
    }
  },

  get tableRows() {
    const rows = [];
    for (const s of this.schedules) {
      rows.push({ type: 'schedule', data: s, key: s.id });
      if (this.expandedId === s.id) {
        const runs = this.runSessions[s.id];
        if (runs && runs.length > 0) {
          for (const run of runs) {
            rows.push({ type: 'run', data: run, key: 'run-' + run.id });
          }
        } else if (runs) {
          rows.push({ type: 'empty', data: null, key: 'empty-' + s.id });
        }
      }
    }
    return rows;
  },

  async toggleRuns(s) {
    if (this.expandedId === s.id) {
      this.expandedId = null;
      return;
    }
    this.expandedId = s.id;
    if (!this.runSessions[s.id]) {
      try {
        const data = await get(`/api/schedules/${s.id}/sessions`);
        this.runSessions = { ...this.runSessions, [s.id]: data.sessions || [] };
      } catch {
        this.runSessions = { ...this.runSessions, [s.id]: [] };
      }
    }
  },

  runBadge(status) {
    return stateBadgeClass(status);
  },

  viewRunHistory(run) {
    this.$store.app.viewSessionId = run.id;
    this.$store.app.view = 'conversations';
  },

  get formTitle() {
    return this.editingId ? `Edit: ${this.editingId}` : 'New Schedule';
  },

  statusBadge(s) {
    if (!s.enabled) return 'badge-muted';
    if (s.last_error) return 'badge-error';
    return 'badge-ok';
  },

  statusLabel(s) {
    if (!s.enabled) return 'disabled';
    if (s.last_error) return 'error';
    return 'active';
  },

  formatDate(iso) {
    return formatDate(iso) || 'never';
  },
});
