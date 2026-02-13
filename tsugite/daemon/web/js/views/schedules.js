import { get, post, patch, del } from '../api.js';
import { formatDate } from '../utils.js';

const emptyForm = () => ({ id: '', agent: '', prompt: '', schedule_type: 'cron', cron_expr: '', run_at: '', timezone: 'UTC' });

export default () => ({
  schedules: [],
  showForm: false,
  loading: true,
  error: null,
  editingId: null,
  form: emptyForm(),

  init() {
    this.load();
  },

  async load() {
    this.error = null;
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
      if (this.form.prompt) body.prompt = this.form.prompt;
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
      try {
        await post('/api/schedules', body);
        this.cancelForm();
        await this.load();
      } catch (e) {
        this.error = e.message;
      }
    }
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
