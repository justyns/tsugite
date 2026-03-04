import { get, post } from '../api.js';
import { formatDate } from '../utils.js';

export default () => ({
  sessions: [],
  reviews: [],
  selectedSession: null,
  events: [],
  showForm: false,
  loading: true,
  error: null,
  reviewComment: '',
  form: { agent: '', prompt: '', model: '' },
  _refreshTimer: null,

  init() {
    this.load();
    this._startPolling();
    this.$watch('$store.app.view', (view) => {
      if (view === 'sessions') this._startPolling();
      else this._stopPolling();
    });
  },

  destroy() {
    this._stopPolling();
  },

  _startPolling() {
    if (this._refreshTimer) return;
    this._refreshTimer = setInterval(() => this.load(), 5000);
  },

  _stopPolling() {
    if (this._refreshTimer) {
      clearInterval(this._refreshTimer);
      this._refreshTimer = null;
    }
  },

  async load() {
    try {
      const [sessData, revData] = await Promise.all([
        get('/api/sessions'),
        get('/api/reviews?status=pending'),
      ]);
      this.sessions = sessData.sessions || [];
      this.reviews = revData.reviews || [];
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
      const data = await get(`/api/sessions/${s.id}`);
      this.selectedSession = data;
      const evData = await get(`/api/sessions/${s.id}/events`);
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

  async resolveReview(r, decision) {
    try {
      await post(`/api/reviews/${r.id}/resolve`, {
        decision,
        comment: this.reviewComment,
      });
      this.reviewComment = '';
      await this.load();
      if (this.selectedSession) await this.selectSession(this.selectedSession);
    } catch (e) {
      this.error = e.message;
    }
  },

  stateBadge(state) {
    const map = {
      pending: 'badge-muted',
      running: 'badge-accent',
      waiting_for_review: 'badge-warning',
      completed: 'badge-ok',
      failed: 'badge-error',
      cancelled: 'badge-muted',
      interrupted: 'badge-error',
    };
    return map[state] || '';
  },

  canCancel(s) {
    return s.state === 'running' || s.state === 'waiting_for_review';
  },

  canRestart(s) {
    return s.state === 'interrupted' || s.state === 'failed' || s.state === 'cancelled';
  },

  formatDate(iso) {
    return formatDate(iso) || '—';
  },
});
