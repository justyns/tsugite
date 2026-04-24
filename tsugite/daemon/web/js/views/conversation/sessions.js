import { get, post, patch } from '../../api.js';
import { REPLAY_SKIP_EVENTS, progressFromPayload } from './event_types.js';

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
      const liveIds = new Set();
      for (const s of this.allSessions) {
        if (s.state !== 'running' && s.state !== 'active') continue;
        liveIds.add(s.id);
        if (s.progress && !this.progressCache[s.id]) {
          this.progressCache[s.id] = progressFromPayload(s.progress);
        }
      }
      for (const id of Object.keys(this.progressCache)) {
        if (!liveIds.has(id)) delete this.progressCache[id];
      }
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

  async selectSession(session) {
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
    const historyPromise = this.loadHistory();
    this.loadSessionEffort();
    this._restoreDraft();
    if (!this.isActiveSession && session.state === 'running') {
      await historyPromise;
      if (this.selectedSessionId !== convId) return;
      this._rehydrateProgressFromEvents(convId);
    }
  },

  async _rehydrateProgressFromEvents(sessionId) {
    const progress = { type: 'progress', steps: [], statusText: 'Starting...', turnCount: 0, toolCount: 0 };
    this._sessionProgress = progress;
    const idx = this.messages.push(progress) - 1;
    try {
      const data = await get(`/api/sessions/${sessionId}/events`);
      // Bail if reload/backToSessions/another selectSession swapped our bubble out while the fetch was in flight.
      if (this.selectedSessionId !== sessionId || this.messages[idx] !== progress) return;
      for (const ev of data.events || []) {
        if (REPLAY_SKIP_EVENTS.has(ev.type)) continue;
        this._handleProgressEvent(idx, ev);
      }
      this.scrollMessages();
    } catch { /* ignore */ }
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

  sessionProgressLabel(s) {
    if (!s) return '';
    const running = s.state === 'running' || s.state === 'active';
    if (!running) return '';
    const cached = this.progressCache[s.id] || progressFromPayload(s.progress);
    if (!cached) return 'Starting...';
    const parts = [];
    if (cached.turnCount) parts.push(`Turn ${cached.turnCount}`);
    if (cached.toolCount) parts.push(`${cached.toolCount} tool${cached.toolCount > 1 ? 's' : ''}`);
    if (cached.statusText) parts.push(cached.statusText);
    return parts.join(' · ') || 'Starting...';
  },

  isSessionProgressFresh(s) {
    void this._freshnessTick;  // establish reactive dep so pulse turns off when events stop
    const cached = this.progressCache[s?.id];
    if (!cached || !cached.lastEventTime) return false;
    const age = Date.now() - Date.parse(cached.lastEventTime);
    return age >= 0 && age < 10000;
  },

  _matchesFilters(s) {
    if (this.sessionFilter) {
      const text = [s.title, s.label, s.id, s.conversation_id, s.source, s.state].filter(Boolean).join(' ');
      if (!text.toLowerCase().includes(this.sessionFilter.toLowerCase())) return false;
    }
    return true;
  },
};
