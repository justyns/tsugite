import { get, post, patch } from '../api.js';
import { escapeHtml, renderMarkdown, scrollToBottom, formatDate, formatRelativeTime, stateBadgeClass, copyText, toast } from '../utils.js';
import { sessionsMixin } from './conversation/sessions.js';
import { historyMixin } from './conversation/history.js';
import { attachmentsMixin } from './conversation/attachments.js';
import { streamingMixin } from './conversation/streaming.js';
import { inputMixin } from './conversation/input.js';
import { SESSION_END_EVENTS, TURN_END_EVENTS, progressStatusFor } from './conversation/event_types.js';

// Maps prompt-inspector category names to Catppuccin CSS variables. Using vars
// (not raw hex) means the stacked bar re-colors when the user switches themes.
const PI_COLORS = {
  instructions: 'var(--pink)',
  tools: 'var(--lavender)',
  attachments: 'var(--teal)',
  skills: 'var(--green)',
  task: 'var(--yellow)',
  system: 'var(--mauve)',
  user_messages: 'var(--blue)',
  system_prompt: 'var(--pink)',
  metadata: 'var(--sapphire)',
};

export default () => ({
  ...sessionsMixin,
  ...historyMixin,
  ...attachmentsMixin,
  ...streamingMixin,
  ...inputMixin,

  loading: true,
  // session_id -> per-session UI state. Single source of truth so every per-session
  // field is added/cleared in one place instead of spread across parallel maps.
  sessionsState: {},
  showAttachments: false,
  attachments: [],
  inputMenuOpen: false,
  showSkills: false,
  inspectingSnapshot: null,
  piExpanded: null,
  effortLevels: [],
  availableModels: null,
  // Getters must stay here — spread loses get descriptors
  get userId() {
    return this.$store.app.userId;
  },

  _sessionState(sid) {
    if (!sid) return null;
    return (this.sessionsState[sid] ||= {
      messages: [],
      sending: false,
      reader: null,
      historyLoading: false,
      prefetchedEvents: null,
      prefetching: false,
      progress: null,
      compacting: false,
      compactingCounts: null,
      compactingPhase: null,
      loadedSkills: [],
      statusInfo: {},
      effort: '',
      model: '',
      compactionSummary: null,
      compactedIntoEvent: null,
      liveProgress: null,
    });
  },

  get messages() {
    const s = this._sessionState(this.selectedSessionId);
    return s ? s.messages : [];
  },
  set messages(v) {
    const s = this._sessionState(this.selectedSessionId);
    if (s) s.messages = v;
  },
  get sending() {
    return !!this.sessionsState[this.selectedSessionId]?.sending;
  },
  get historyLoading() {
    return !!this.sessionsState[this.selectedSessionId]?.historyLoading;
  },
  get compacting() {
    return !!this.sessionsState[this.selectedSessionId]?.compacting;
  },
  get compactingCounts() {
    return this.sessionsState[this.selectedSessionId]?.compactingCounts || null;
  },
  get compactingPhase() {
    return this.sessionsState[this.selectedSessionId]?.compactingPhase || null;
  },
  get loadedSkills() {
    return this.sessionsState[this.selectedSessionId]?.loadedSkills || [];
  },
  get statusInfo() {
    return this.sessionsState[this.selectedSessionId]?.statusInfo || {};
  },
  get sessionEffort() {
    return this.sessionsState[this.selectedSessionId]?.effort || '';
  },
  get sessionModel() {
    return this.sessionsState[this.selectedSessionId]?.model || '';
  },
  get compactionSummary() {
    return this.sessionsState[this.selectedSessionId]?.compactionSummary || null;
  },
  get compactedIntoEvent() {
    return this.sessionsState[this.selectedSessionId]?.compactedIntoEvent || null;
  },
  get _sessionProgress() {
    return this.sessionsState[this.selectedSessionId]?.liveProgress || null;
  },
  set _sessionProgress(v) {
    const s = this._sessionState(this.selectedSessionId);
    if (s) s.liveProgress = v;
  },

  get filteredCommands() {
    const text = this.messageText;
    if (!text.startsWith('/')) return [];
    const query = text.slice(1).split(/\s/)[0].toLowerCase();
    if (text.includes(' ')) return [];
    return this.availableCommands.filter(c => c.name.startsWith(query));
  },

  get groupedSessions() {
    const pinned = [];
    const active = [];
    const recent = [];

    for (const s of this.allSessions) {
      if (!this._matchesFilters(s)) continue;
      // Superseded sessions stay in allSessions for chase lookups but should
      // never appear in the sidebar.
      if (s.superseded_by) continue;
      if (s.pinned) {
        pinned.push(s);
        continue;
      }
      const state = s.state;
      if (state === 'active' || state === 'running') {
        active.push(s);
      } else {
        recent.push(s);
      }
    }

    const byDate = (a, b) => (b.last_active || b.created_at || '').localeCompare(a.last_active || a.created_at || '');
    pinned.sort((a, b) => (a.pin_position ?? 0) - (b.pin_position ?? 0));
    active.sort(byDate);
    recent.sort(byDate);

    const maxRecent = 10;
    const visibleRecent = this.showRecentHidden ? recent : recent.slice(0, maxRecent);

    return {
      pinned,
      active,
      recent: visibleRecent,
      recentTotal: recent.length,
      recentHidden: recent.length - visibleRecent.length,
    };
  },

  get statusParts() {
    const info = this.statusInfo;
    const parts = [];
    if (info.message_count != null) parts.push({ label: 'Messages', value: String(info.message_count) });
    if (info.attachments?.length) parts.push({ label: 'Attachments', value: `${info.attachments.length} file${info.attachments.length > 1 ? 's' : ''}`, isLink: 'attachments' });
    if (this.loadedSkills.length) parts.push({ label: 'Skills', value: `${this.loadedSkills.length} loaded`, isLink: 'skills' });
    return parts;
  },

  // Lifecycle
  init() {
    this._loadCommands();
    this._mobileQuery = window.matchMedia('(max-width: 640px)');
    const maybeReload = () => {
      if (this.$store.app.view === 'conversations' && this.$store.app.selectedAgent) this.reload();
    };
    this.$watch('$store.app.selectedAgent', maybeReload);
    this.$watch('$store.app.view', (view) => {
      if (view === 'conversations' && this.$store.app.selectedAgent) this.reload();
    });
    if (this.$store.app.view === 'conversations' && this.$store.app.selectedAgent) {
      this.reload();
    }
    this.$watch('$store.app.lastEvent', (ev) => {
      if (!ev) return;
      const d = ev.data || {};

      if (ev.type === 'history_update' && d.agent === this.$store.app.selectedAgent) {
        // If the event carries a session_id, only reload when it matches the
        // viewed session — avoids waking every tab on the agent when one
        // session's turn completes. Falls back to the legacy agent-scope path
        // for events that predate the session_id plumbing.
        if (d.session_id && d.session_id !== this.selectedSessionId) {
          // no-op: turn completed in some other session this tab isn't viewing
        } else if (this.isActiveSession && !this.sending) {
          // Per-chat streaming reader is authoritative mid-turn; reloading
          // races its final_result push. Reconciliation happens via the next
          // reload once the session's `sending` flag clears.
          this._debouncedLoadHistory();
        }
      }
      if (ev.type === 'session_update') {
        if (d.action === 'titled' && d.title) {
          const s = this.allSessions.find(x => x.id === d.id);
          if (s) { s.title = d.title; return; }
        }
        if (d.action === 'metadata_updated' && d.id && d.metadata) {
          const s = this.allSessions.find(x => x.id === d.id);
          if (s) {
            s.metadata = { ...d.metadata };
            if (this.selectedSessionId === (s.conversation_id || s.id)) {
              this.selectedSessionMeta = { ...this.selectedSessionMeta, metadata: s.metadata };
            }
            return;
          }
        }
        if (d.action === 'compacted' && d.successor_id && this.selectedSessionId === d.id) {
          // The session the user is looking at just compacted; follow the chain forward
          // so the next message doesn't post to a now-completed predecessor.
          this.loadSessions().then(() => this.selectSessionById(d.successor_id));
          return;
        }
        this._debouncedLoadSessions();
      }
      if (ev.type === 'session_event') {
        this._updateProgressCache(d);
        this._handleSessionEvent(d);
      }
      if (ev.type === 'compaction_started' && d.agent === this.$store.app.selectedAgent && d.session_id) {
        const s = this._sessionState(d.session_id);
        s.compacting = true;
        s.compactingCounts = null;
        s.compactingPhase = null;
      }
      if (ev.type === 'compaction_progress' && d.agent === this.$store.app.selectedAgent && d.session_id) {
        const s = this._sessionState(d.session_id);
        if (d.phase === 'starting') {
          s.compactingCounts = { replaced_count: d.replaced_count || 0, retained_count: d.retained_count || 0 };
        } else {
          s.compactingPhase = d;
        }
      }
      if (ev.type === 'compaction_finished' && d.agent === this.$store.app.selectedAgent && d.session_id) {
        const s = this._sessionState(d.session_id);
        s.compacting = false;
        s.compactingCounts = null;
        s.compactingPhase = null;
        if (this.isActiveSession && d.session_id === this.selectedSessionId) this.reload();
      }
      if (ev.type === 'reconnect' && this.$store.app.selectedAgent) {
        this.reload();
      }
    });
    this.$watch('$store.app.pendingWorkspaceFiles', (files) => {
      if (!files || !files.length) return;
      for (const f of files) {
        this.pendingFiles.push({ name: f.name, file: null, previewUrl: null, fromWorkspace: true, uploadInfo: f });
      }
      this.$store.app.pendingWorkspaceFiles = [];
    });
    // Refresh relative-time labels ("5m ago") in the sidebar.
    this._relTimeTimer = setInterval(() => { this._relTimeTick++; }, 60000);
    // Mark the currently-selected session as viewed when the user refocuses the tab.
    this._onVisibilityChange = () => {
      if (document.visibilityState !== 'visible' || !this.selectedSessionId) return;
      const s = this.allSessions.find(x => (x.conversation_id || x.id) === this.selectedSessionId);
      if (s && s.unread) this._markSessionViewed(s);
    };
    document.addEventListener('visibilitychange', this._onVisibilityChange);
  },

  _relTimeTick: 0,

  destroy() {
    if (this._draftTimer) clearTimeout(this._draftTimer);
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    if (this._scrollTimer) clearTimeout(this._scrollTimer);
    if (this._historyDebounceTimer) clearTimeout(this._historyDebounceTimer);
    if (this._relTimeTimer) clearInterval(this._relTimeTimer);
    if (this._onVisibilityChange) document.removeEventListener('visibilitychange', this._onVisibilityChange);
    Object.values(this.sessionsState).forEach(s => { if (s.reader) s.reader.cancel().catch(() => {}); });
    this.pendingFiles.forEach(f => { if (f.previewUrl) URL.revokeObjectURL(f.previewUrl); });
  },

  async reload() {
    this.resetHistory();
    this.sessionsState = {};
    this.selectedSessionMeta = null;
    await this.loadSessions();
    await this.loadEffortLevels();

    const targetId = this.$store.app.viewSessionId;
    if (targetId) {
      this.$store.app.viewSessionId = null;
      const match = this.allSessions.find(s => s.conversation_id === targetId || s.id === targetId);
      if (match) {
        this.selectSession(match);
      } else {
        this.selectSession({ conversation_id: targetId, agent: this.$store.app.selectedAgent });
      }
    } else {
      // PWA cold-restart at start_url drops the URL hash; restore the user's
      // last selection from localStorage before falling back to autoSelect.
      // selectSession already chases superseded_by, so a stale persisted ID for
      // a now-compacted session lands on the live successor automatically.
      const persisted = this._loadPersistedSessionId();
      const restored = persisted ? this.findSession(persisted) : null;
      if (restored) {
        this.selectSession(restored);
      } else {
        this.autoSelectInteractive();
      }
    }
    this._prefetchTopSessions();
  },

  _prefetchTopSessions() {
    const g = this.groupedSessions;
    const seen = new Set();
    const picks = [];
    const take = (s) => {
      const sid = s?.conversation_id || s?.id;
      if (!sid || seen.has(sid) || sid === this.selectedSessionId) return;
      seen.add(sid);
      picks.push(sid);
    };
    g.pinned.forEach(take);
    g.active.forEach(take);
    g.recent.forEach(take);
    picks.slice(0, 5).forEach(sid => this._prefetchHistory(sid));
  },

  async _prefetchHistory(sessionId) {
    if (!sessionId) return;
    const state = this._sessionState(sessionId);
    if (state.prefetchedEvents || state.messages.length || state.prefetching) return;
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    state.prefetching = true;
    try {
      const url = `/api/agents/${agent}/history?user_id=${encodeURIComponent(this.userId)}&limit=100&session_id=${encodeURIComponent(sessionId)}`;
      const data = await get(url);
      state.prefetchedEvents = data.events || [];
    } catch { /* non-fatal */ }
    state.prefetching = false;
  },

  async loadEffortLevels() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) { this.effortLevels = []; return; }
    try {
      const data = await get(`/api/agents/${agent}/effort-levels`);
      this.effortLevels = Array.isArray(data.supported_effort_levels) ? data.supported_effort_levels : [];
    } catch {
      this.effortLevels = [];
    }
  },

  async loadSessionEffort() {
    const sid = this.selectedSessionId;
    if (!sid) return;
    const state = this._sessionState(sid);
    try {
      const data = await get(`/api/sessions/${sid}/settings`);
      state.effort = data.reasoning_effort || '';
      state.model = data.model || '';
    } catch {
      state.effort = '';
      state.model = '';
    }
  },

  async setSessionEffort(value) {
    const sid = this.selectedSessionId;
    if (!sid) return;
    this._sessionState(sid).effort = value || '';
    try {
      await patch(`/api/sessions/${sid}/settings`, { reasoning_effort: value || null });
    } catch (e) {
      console.warn('Failed to save reasoning_effort', e);
    }
  },

  async setSessionAgent(name) {
    if (!name || name === this.$store.app.selectedAgent) return;
    this.$store.app.selectedAgent = name;
    if (this.selectedSessionMeta) this.selectedSessionMeta.agent = name;
    if (!this.selectedSessionId) return;
    try {
      await patch(`/api/sessions/${this.selectedSessionId}/settings`, { agent: name });
    } catch (e) {
      console.warn('Failed to save agent override', e);
      toast(`agent change failed: ${e.message || 'unknown'}`, 'error');
    }
  },

  async setSessionModel(value) {
    const sid = this.selectedSessionId;
    if (!sid) return;
    const next = value || '';
    if (next === this.sessionModel) return;
    this._sessionState(sid).model = next;
    try {
      await patch(`/api/sessions/${sid}/settings`, { model: value || null });
    } catch (e) {
      console.warn('Failed to save model override', e);
      toast(`model change failed: ${e.message || 'unknown'}`, 'error');
    }
  },

  async loadModels() {
    if (this.availableModels !== null) return this.availableModels;
    try {
      const data = await get('/api/models');
      this.availableModels = (data.models || []).map(m => ({
        id: m.id || m.name,
        description: m.description || m.provider || '',
        context: m.context_window || m.context || null,
      }));
    } catch {
      this.availableModels = [];
    }
    return this.availableModels;
  },

  // Shared helper used by both history and streaming mixins
  _hookStepHtml(name, phase, exitCode) {
    const cls = exitCode === 0 ? 'ok' : 'err';
    const label = `${name || 'hook'} (${phase})`;
    return `<code>${escapeHtml(label)}</code> <span class="hook-exit ${cls}">exit ${exitCode}</span>`;
  },

  // Status
  async loadStatus() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    const sid = this.selectedSessionId;
    if (!sid) return;
    const state = this._sessionState(sid);
    try {
      const data = await get(`/api/agents/${agent}/status?user_id=${encodeURIComponent(this.userId)}&session_id=${encodeURIComponent(sid)}`);
      state.statusInfo = data;
      if (data.compacting !== undefined) state.compacting = !!data.compacting;
      if (data.busy && data.pending_message && !state.sending &&
          !this.messages.some(m => m.type === 'user' && m.text === data.pending_message)) {
        this.messages.push({ type: 'user', text: data.pending_message });
        this.messages.push({ type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 });
        this.scrollMessages();
      }
    } catch { /* ignore */ }
  },

  updateStatusFromEvent(event, sessionId) {
    const sid = sessionId || this.selectedSessionId;
    if (!sid) return;
    this._sessionState(sid).statusInfo = {
      model: event.model,
      tokens: event.tokens,
      context_limit: event.context_limit,
      threshold: event.threshold,
      message_count: event.message_count,
      attachments: event.attachments,
    };
  },

  async openPromptInspector() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    try {
      let url = `/api/agents/${agent}/prompt-snapshot?user_id=${encodeURIComponent(this.userId)}`;
      if (this.selectedSessionId) url += `&session_id=${encodeURIComponent(this.selectedSessionId)}`;
      const data = await get(url);
      if (data.prompt_snapshot) this.inspectingSnapshot = data.prompt_snapshot;
    } catch { /* ignore */ }
  },

  async openAttachments() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    this.showAttachments = true;
    this.attachments = [];
    try {
      const data = await get(`/api/agents/${agent}/attachments`);
      this.attachments = data.attachments || [];
    } catch { /* ignore */ }
  },

  async removeLoadedSkill(name) {
    const agent = this.$store.app.selectedAgent;
    const sid = this.selectedSessionId;
    if (!agent || !sid) return;
    try {
      await post(`/api/agents/${agent}/unload-skill`, { user_id: this.userId, name });
      const state = this._sessionState(sid);
      state.loadedSkills = state.loadedSkills.filter(s => s.name !== name);
    } catch (e) {
      this.messages.push({ type: 'error', text: `Remove skill failed: ${e.message}` });
    }
  },

  get compactingLabel() {
    if (!this.compacting) return '';
    const c = this.compactingCounts;
    const phase = this.compactingPhase;
    let base;
    if (c && c.replaced_count) {
      const n = c.replaced_count;
      base = `summarizing ${n} turn${n === 1 ? '' : 's'}`;
      if (c.retained_count) base += `, keeping ${c.retained_count}`;
    } else {
      base = 'compacting';
    }
    if (phase && phase.phase === 'summarizing' && phase.chunk_total > 1) {
      base += ` (chunk ${phase.chunk_index}/${phase.chunk_total})`;
    } else if (phase && phase.phase === 'combining') {
      base = 'combining summary';
    } else if (phase && phase.phase === 'chunking') {
      base += ' (preparing chunks)';
    }
    return base + '…';
  },

  async compactSession(retryMsg = null) {
    const agent = this.$store.app.selectedAgent;
    const sid = this.selectedSessionId;
    if (!agent || !sid) return;
    const state = this._sessionState(sid);
    if (state.sending || state.compacting) return;
    state.compacting = true;
    try {
      await post(`/api/agents/${agent}/compact`, { user_id: this.userId, session_id: sid });
      await this.loadHistory();
      if (retryMsg) {
        this.messageText = retryMsg;
        await this.sendMessage();
      }
    } catch (e) {
      this.messages.push({ type: 'error', text: `Compact failed: ${e.message}` });
    } finally {
      state.compacting = false;
    }
  },

  async submitAskUser(msgIndex, response) {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    const msg = this.messages[msgIndex];
    if (!msg || msg.answered) return;
    msg.answered = true;
    msg.answer = response;
    try {
      await post(`/api/agents/${agent}/respond`, { response, user_id: this.userId, session_id: this.selectedSessionId });
    } catch (e) {
      this.messages.push({ type: 'error', text: `Failed to submit answer: ${e.message}` });
    }
  },

  // Rendering helpers
  renderHtml(text) { return renderMarkdown(text); },
  escape(s) { return escapeHtml(s); },
  formatDate(iso) { return formatDate(iso) || '-'; },
  formatRelativeTime(iso) {
    void this._relTimeTick;  // reactive dep so Alpine re-renders when the tick advances
    return formatRelativeTime(iso) || '';
  },
  stateBadge(state) { return stateBadgeClass(state); },

  // User-initiated copy of a message's raw markdown
  copyMessage(text) { copyText(text); },

  // Delegated click handler on #messages — catches .copy-code buttons
  // injected by renderMarkdown (Alpine can't bind handlers to x-html content).
  handleMessagesClick(e) {
    const btn = e.target.closest('.copy-code');
    if (!btn) return;
    const pre = btn.closest('pre');
    const code = pre && pre.querySelector('code');
    if (!code) return;
    // textContent preserves exact whitespace; innerText collapses it.
    copyText(code.textContent);
  },

  isAtBottom: true,

  onMessagesScroll() {
    const el = this.$refs.messages;
    if (!el) return;
    this.isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  },

  scrollMessages(force = false) {
    this.$nextTick(() => {
      const el = this.$refs.messages;
      if (!el) return;
      if (force || this.isAtBottom) scrollToBottom(el);
    });
  },

  _updateProgressCache(d) {
    const id = d.session_id;
    if (!id) return;
    const state = this._sessionState(id);
    const evType = d.event_type;
    if (SESSION_END_EVENTS.has(evType)) {
      state.progress = null;
      return;
    }
    const now = d.timestamp || new Date().toISOString();
    if (TURN_END_EVENTS.has(evType)) {
      // Keep lastEventTime so sessionProgressLabel knows we're between turns
      // (otherwise it falls back to "Starting...").
      const entry = state.progress;
      if (entry && !entry.turnCount && !entry.toolCount && !entry.statusText) {
        entry.lastEventTime = now;
      } else {
        state.progress = { turnCount: 0, toolCount: 0, statusText: '', lastEventTime: now };
      }
      return;
    }
    let entry = state.progress;
    if (!entry) {
      entry = { turnCount: 0, toolCount: 0, statusText: 'Starting...', lastEventTime: now };
      state.progress = entry;
    }
    let { turnCount, toolCount, statusText } = entry;
    if (evType === 'turn_start' && typeof d.turn === 'number' && d.turn > turnCount) turnCount = d.turn;
    else if (evType === 'tool_result' && (d.tool || 'unknown') !== 'unknown') toolCount += 1;
    const label = progressStatusFor(evType, d);
    if (label) statusText = label;
    const derivedChanged = turnCount !== entry.turnCount || toolCount !== entry.toolCount || statusText !== entry.statusText;
    if (derivedChanged) {
      state.progress = { turnCount, toolCount, statusText, lastEventTime: now };
    } else if (entry.lastEventTime !== now) {
      entry.lastEventTime = now;
    }
  },

  _handleSessionEvent(d) {
    if (!this.selectedSessionId) return;
    if (d.session_id !== this.selectedSessionId) return;
    // sendMessage's per-chat streaming response already populates the live
    // progress bubble while it's running; bail to avoid double-rendering.
    // Once it finishes (or after a reload), this session's sending flag is
    // cleared and the global SSE feed becomes the source of in-flight progress.
    if (this.sessionsState[d.session_id]?.sending) return;

    const evType = d.event_type;
    // Defense-in-depth: even if a future adapter broadcasts turn-end events,
    // loadHistory (fired by the paired history_update) reconciles them from
    // JSONL. Pushing here would race the per-chat reader's clear of the
    // session's `sending` flag and produce duplicate bubbles.
    if (TURN_END_EVENTS.has(evType)) return;

    if (!this._sessionProgress) {
      this._sessionProgress = { type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 };
      this.messages.push(this._sessionProgress);
    }

    if (SESSION_END_EVENTS.has(evType)) {
      this._sessionProgress.type = 'progress-done';
      if (evType === 'session_error') {
        this._sessionProgress.failed = true;
        this._sessionProgress.errorText = d.error || 'Session failed';
      }
      this._sessionProgress = null;
      this._debouncedLoadSessions();
      if (evType === 'session_complete' && d.result_preview) {
        this.messages.push({ type: 'agent', text: d.result_preview });
      }
      this._scrollThrottled();
      return;
    }

    const progressIdx = this.messages.indexOf(this._sessionProgress);
    if (progressIdx >= 0) {
      const { session_id, event_type, ...rest } = d;
      this._handleProgressEvent(progressIdx, { type: evType, ...rest });
      this._scrollThrottled();
    }
  },

  // ---------- Console helpers ----------

  isLive(s) {
    if (!s) return false;
    return s.state === 'running' || s.state === 'active' || s.state === 'thinking' || s.state === 'starting';
  },

  statusDotClass(state) {
    switch (state) {
      case 'running':
      case 'active': return 'running';
      case 'thinking': return 'thinking';
      case 'starting': return 'starting';
      case 'scheduled': return 'scheduled';
      case 'completed':
      case 'done': return 'done';
      case 'failed':
      case 'error':
      case 'cancelled': return 'failed';
      default: return 'idle';
    }
  },

  // Alpine 3's :class array form does not unpack nested objects — it stringifies
  // them to `[object Object]`. Build a flat string so the `pulse` class actually
  // lands on the dot.
  dotClassNames(s) {
    const base = this.statusDotClass(s.state);
    return this.isSessionProgressFresh(s) ? `${base} pulse` : base;
  },

  recentBuckets() {
    void this._relTimeTick;
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const yest = new Date(today);
    yest.setDate(yest.getDate() - 1);
    const weekStart = new Date(today);
    weekStart.setDate(weekStart.getDate() - 7);

    const buckets = [
      { label: 'today', items: [] },
      { label: 'yesterday', items: [] },
      { label: 'this week', items: [] },
      { label: 'older', items: [] },
    ];
    for (const s of (this.groupedSessions.recent || [])) {
      const ts = Date.parse(s.last_active || s.created_at || '');
      if (!ts) { buckets[3].items.push(s); continue; }
      if (ts >= today.getTime()) buckets[0].items.push(s);
      else if (ts >= yest.getTime()) buckets[1].items.push(s);
      else if (ts >= weekStart.getTime()) buckets[2].items.push(s);
      else buckets[3].items.push(s);
    }
    return buckets;
  },

  turnTimeShort(msg) {
    const ts = msg && (msg.timestamp || msg.ts || msg.created_at);
    if (!ts) return '';
    const d = new Date(ts);
    if (isNaN(d.getTime())) return '';
    const h = String(d.getHours()).padStart(2, '0');
    const m = String(d.getMinutes()).padStart(2, '0');
    return `${h}:${m}`;
  },

  turnIndex(i) {
    return '#' + String(i + 1).padStart(2, '0');
  },

  // Compact tokens — "12.3k" / "1.2M" / "850".
  fmtTok(n) {
    if (n == null) return '0';
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return String(n);
  },

  categoryColor(name) { return PI_COLORS[name] || 'var(--lavender)'; },

  composerMetaLabel() {
    const turn = this.statusInfo?.message_count ?? this.messages.length;
    const attached = this.pendingFiles?.length || 0;
    const tokens = this.statusInfo?.tokens;
    const limit = this.statusInfo?.context_limit;
    const ctxPct = (tokens && limit) ? Math.round((tokens / limit) * 100) : null;
    const parts = [];
    if (turn) parts.push(`turn ${turn}`);
    if (attached) parts.push(`${attached} attached`);
    if (ctxPct != null) parts.push(`${ctxPct}% ctx`);
    return parts.length ? parts.join(' · ') : 'shift+enter for newline';
  },
});
