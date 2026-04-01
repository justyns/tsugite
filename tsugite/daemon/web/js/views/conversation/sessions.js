import { get, post, patch } from '../../api.js';

export const sessionsMixin = {
  _debounceTimer: null,
  allSessions: [],
  selectedSessionId: null,
  isActiveSession: true,
  selectedSessionMeta: null,
  sidebarOpen: false,
  showCompletedScheduled: false,
  sessionFilter: '',
  editingSessionId: null,
  editingTitle: '',

  _debouncedLoadSessions() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    this._debounceTimer = setTimeout(() => this.loadSessions(), 200);
  },

  async loadSessions() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) { this.loading = false; return; }
    try {
      const data = await get(`/api/agents/${agent}/sessions`);
      this.allSessions = (data.sessions || []).map(s => ({ ...s, state: s.state || s.status }));
    } catch { this.allSessions = []; }
    this.loading = false;
  },

  _isMyInteractive(s) {
    return s.source === 'interactive' && (s.user_id === this.userId || s.conversation_id === this.userId);
  },

  autoSelectInteractive() {
    const g = this.groupedSessions;
    const first = g.interactive[0] || g.scheduled[0] || g.background[0];
    if (first) this.selectSession(first);
  },

  selectSession(session) {
    this.sidebarOpen = false;
    const convId = session.conversation_id || session.id;
    this.selectedSessionId = convId;
    this.selectedSessionMeta = session;
    this.messages = [];
    this.resetHistory();

    this.isActiveSession = this._isMyInteractive(session);

    if (this.isActiveSession) {
      this.loadHistory();
      this.loadStatus();
    } else {
      this.loadDetailHistory();
    }
  },

  backToSessions() {
    this.selectedSessionId = null;
    this.selectedSessionMeta = null;
    this.messages = [];
    this.resetHistory();
    this.isActiveSession = true;
  },

  async newSession() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    try {
      const data = await post(`/api/agents/${agent}/sessions/new`, { user_id: this.userId });
      await this.loadSessions();
      const session = this.allSessions.find(s => s.id === data.id);
      if (session) this.selectSession(session);
    } catch (e) {
      this.messages.push({ type: 'error', text: `Failed to create session: ${e.message}` });
    }
  },

  async completeSession(session) {
    const id = session.id || session.conversation_id;
    if (!id || !confirm('Mark this session as completed?')) return;
    try {
      await patch(`/api/sessions/${id}`, { status: 'completed' });
      await this.loadSessions();
    } catch (e) {
      this.messages.push({ type: 'error', text: `Mark complete failed: ${e.message}` });
    }
  },

  async cancelSession(session) {
    const id = session.id;
    if (!id || !confirm(`Cancel session "${id}"?`)) return;
    try {
      await post(`/api/sessions/${id}/cancel`);
      await this.loadSessions();
    } catch (e) {
      this.messages.push({ type: 'error', text: `Cancel failed: ${e.message}` });
    }
  },

  async restartSession(session) {
    const id = session.id;
    if (!id) return;
    try {
      await post(`/api/sessions/${id}/restart`);
      await this.loadSessions();
    } catch (e) {
      this.messages.push({ type: 'error', text: `Restart failed: ${e.message}` });
    }
  },

  canCancel(s) {
    return s?.state === 'running' && !s?.is_default;
  },

  canRestart(s) {
    return (s?.state === 'failed' || s?.state === 'cancelled') && !s?.is_default;
  },

  canComplete(s) {
    return (s?.state === 'active' || s?.state === 'running') && !s?.is_default;
  },

  sessionLabel(s) {
    if (s.title) return s.title;
    if (this._isMyInteractive(s)) {
      return s.label || s.agent || 'Interactive';
    }
    if (s.id && s.id.startsWith('sched_')) {
      const parts = s.id.replace(/^sched_/, '').split('_');
      const dateIdx = parts.findIndex(p => /^\d{8}$/.test(p));
      const name = dateIdx > 0 ? parts.slice(0, dateIdx).join('-') : parts[0];
      return name || s.id;
    }
    return s.label || s.conversation_id || s.id || 'unknown';
  },

  startEditTitle(s, event) {
    event.stopPropagation();
    this.editingSessionId = s.id;
    this.editingTitle = s.title || this.sessionLabel(s);
    this.$nextTick(() => {
      const input = this.$el.querySelector('.session-title-input');
      if (input) { input.focus(); input.select(); }
    });
  },

  async saveTitle(s) {
    const title = this.editingTitle.trim();
    this.editingSessionId = null;
    if (!title || title === (s.title || this.sessionLabel(s))) return;
    try {
      await patch(`/api/sessions/${s.id}`, { title });
      s.title = title;
    } catch (e) {
      console.error('Failed to rename session', e);
    }
  },

  statusDotColor(state) {
    if (state === 'active' || state === 'running') return 'var(--ok)';
    if (state === 'error' || state === 'failed') return 'var(--error)';
    return 'var(--muted)';
  },
};
