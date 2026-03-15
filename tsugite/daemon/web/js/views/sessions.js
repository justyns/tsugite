import { get, post } from '../api.js';
import { formatDate, stateBadgeClass } from '../utils.js';

export default () => ({
  sessions: [],
  selectedSession: null,
  events: [],
  showForm: false,
  loading: true,
  error: null,
  form: { agent: '', prompt: '', model: '' },
  _debounceTimer: null,

  init() {
    this.load();
    this.$watch('$store.app.lastEvent', (ev) => {
      if (!ev) return;
      if (ev.type === 'session_update') {
        this._debouncedLoad();
      }
    });
  },

  destroy() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
  },

  _debouncedLoad() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    this._debounceTimer = setTimeout(() => this.load(), 200);
  },

  async load() {
    try {
      const sessData = await get('/api/sessions');
      this.sessions = (sessData.sessions || []).map(s => ({ ...s, state: s.state || s.status }));
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  },

  async selectSession(s) {
    this.selectedSession = s;
    this.events = [];
    try {
      const [data, evData] = await Promise.all([
        get(`/api/sessions/${s.id}`),
        get(`/api/sessions/${s.id}/events`),
      ]);
      data.state = data.state || data.status;
      this.selectedSession = data;
      this.events = evData.events || [];
    } catch (e) {
      this.error = e.message;
    }
  },

  async startSession() {
    this.error = null;
    const body = {
      agent: this.form.agent,
      prompt: this.form.prompt,
    };
    if (this.form.model) body.model = this.form.model;
    try {
      await post('/api/sessions', body);
      this.showForm = false;
      this.form = { agent: '', prompt: '', model: '' };
      await this.load();
    } catch (e) {
      this.error = e.message;
    }
  },

  async cancelSession(s) {
    if (!confirm(`Cancel session "${s.id}"?`)) return;
    try {
      await post(`/api/sessions/${s.id}/cancel`);
      await this.load();
    } catch (e) {
      this.error = e.message;
    }
  },

  async restartSession(s) {
    try {
      await post(`/api/sessions/${s.id}/restart`);
      await this.load();
    } catch (e) {
      this.error = e.message;
    }
  },

  get groupedSessions() {
    const groups = { background: [], spawned: [], schedule: [], interactive: [] };
    for (const s of this.sessions) {
      const src = s.source || 'background';
      if (groups[src]) groups[src].push(s);
      else groups.background.push(s);
    }
    return groups;
  },

  get sourceOrder() {
    return ['background', 'spawned', 'schedule', 'interactive'].filter(
      src => (this.groupedSessions[src] || []).length > 0
    );
  },

  sourceLabel(src) {
    return { background: 'Background', spawned: 'Spawned', schedule: 'Schedule Runs', interactive: 'Interactive' }[src] || src;
  },

  stateBadge(state) {
    return stateBadgeClass(state);
  },

  canCancel(s) {
    return s.state === 'running';
  },

  canRestart(s) {
    return s.state === 'failed' || s.state === 'cancelled';
  },

  formatDate(iso) {
    return formatDate(iso) || '—';
  },
});
