import { get, post, patch } from '../../api.js';

export const sessionsMixin = {
  _debounceTimer: null,
  allSessions: [],
  selectedSessionId: null,
  isActiveSession: true,
  selectedSessionMeta: null,
  sidebarOpen: false,
  showRecentHidden: false,
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
    const first = g.active[0] || g.recent[0];
    if (first) this.selectSession(first);
  },

  selectSession(session) {
    this.sidebarOpen = false;
    this._saveDraftNow();
    const convId = session.conversation_id || session.id;
    this.selectedSessionId = convId;
    this.selectedSessionMeta = session;
    this.messages = [];
    this.statusInfo = {};
    this.resetHistory();

    const hash = `conversations?session=${encodeURIComponent(convId)}`;
    if (location.hash.slice(1) !== hash) location.hash = hash;

    this.isActiveSession = this._isMyInteractive(session);

    this._sessionProgress = null;
    this.loadStatus();
    this.loadHistory();
    this.loadSessionEffort();
    this._restoreDraft();
    if (!this.isActiveSession && session.state === 'running') {
      this._sessionProgress = { type: 'progress', steps: [], statusText: 'Running...', turnCount: 0, toolCount: 0 };
      this.messages.push(this._sessionProgress);
    }
  },

  backToSessions() {
    this._saveDraftNow();
    this.selectedSessionId = null;
    this.selectedSessionMeta = null;
    this._sessionProgress = null;
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

  isSessionInputVisible() {
    return this.isActiveSession || this.selectedSessionMeta?.state === 'running' || this.selectedSessionMeta?.state === 'active';
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
    if (!s) return 'unknown';
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
    if (state === 'running') return 'var(--ok)';
    if (state === 'active') return 'var(--ctp-yellow, var(--ok))';
    if (state === 'error' || state === 'failed') return 'var(--error)';
    return 'var(--muted)';
  },

  sourceIcon(source) {
    return { interactive: 'I', web: 'W', discord: 'D', cli: '>', schedule: 'S',
             background: 'B', spawned: 'P' }[source] || '?';
  },

  metadataChips(s) {
    const meta = s.metadata || {};
    const chips = [];
    if (meta.type) chips.push({ label: meta.type, cls: 'chip-type' });
    if (meta.status_text) chips.push({ label: meta.status_text, cls: 'chip-status' });
    if (meta.task) chips.push({ label: 'Task', href: meta.task, cls: 'chip-link' });
    if (meta.pr) chips.push({ label: 'PR', href: meta.pr, cls: 'chip-link' });
    return chips;
  },

  lastMessagePreview(s) {
    return (s.result || s.prompt || '').slice(0, 60) || '';
  },

  _matchesFilters(s) {
    if (this.sessionFilter) {
      const text = [s.title, s.label, s.id, s.conversation_id, s.source, s.state].filter(Boolean).join(' ');
      if (!text.toLowerCase().includes(this.sessionFilter.toLowerCase())) return false;
    }
    return true;
  },
};
