import { get, post, patch } from '../api.js';
import { escapeHtml, renderMarkdown, scrollToBottom, formatDate, formatRelativeTime, stateBadgeClass, copyText, toast, jobCriteriaStates } from '../utils.js';
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

// Canonical low->high ordering of reasoning-effort levels, used to clamp a
// selected effort down to the nearest supported level when the model changes.
const EFFORT_ORDER = ['low', 'medium', 'high', 'xhigh', 'max'];

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
  // OLD -> NEW redirect map. Compaction rotates the session id, but the
  // streaming.js closure captured sendSessionId = OLD; walking the chain in
  // _sessionState routes every closure-captured write to the live successor's
  // state object so in-flight progress shows up on the session the user now sees.
  _supersededMap: {},
  attachments: [],
  inputMenuOpen: false,
  // Retry-with-hint and mark-done dialogs replace the browser prompt()/confirm()
  // formerly used in jobRetryWithHint/jobMarkDone. Open via _openRetryDialog /
  // _openMarkDoneDialog; submit posts to the same /api/jobs/{id}/... endpoints.
  retryDialog: {
    jobId: '',
    prompt: '',
    lastVerdict: '',
    hint: '',
    model: '',
    currentModel: '',
    resetCounter: true,
    workspace: 'keep',
    suggestions: [
      "the failing tests are pre-existing - verify against a clean baseline first",
      "split the change into smaller commits and re-run tests between each",
      "check the verifier output more carefully and address each failure",
    ],
  },
  markDoneDialog: {
    jobId: '',
    prompt: '',
    criteria: [],
    reason: '',
  },
  inspectingSnapshot: null,
  piExpanded: null,
  effortLevels: [],
  availableModels: null,
  // Getters must stay here — spread loses get descriptors
  get userId() {
    return this.$store.app.userId;
  },

  // Which "kind" of thing is selected in the sidebar - drives the main pane.
  // chat (the default) vs terminal (the /run sub-session, including the
  // empty-state placeholder). Used by index.html to pick between the chat
  // thread view and the full-session terminal view.
  get selectedKind() {
    const t = this.$store.terminals;
    if (t?.selectedId || t?.showEmpty) return 'terminal';
    return 'chat';
  },

  // Clear any chat selection then hand off to the terminals store. Called
  // from the sidebar terminal-section row click handler so the two selection
  // surfaces stay mutually exclusive.
  selectTerminal(terminalId) {
    if (this.selectedSessionId) this.backToSessions();
    this.$store.terminals?.selectTerminal(terminalId);
  },

  _resolveSessionId(sid) {
    if (!sid) return sid;
    // Path compression in _registerSupersession keeps chains at depth 1, so this
    // is normally a single lookup. The hop cap is a paranoid stop in case a
    // pathological SSE stream introduces a cycle.
    const map = this._supersededMap;
    for (let hops = 0; map[sid] && hops < 32; hops++) sid = map[sid];
    return sid;
  },

  _sessionState(sid) {
    if (!sid) return null;
    sid = this._resolveSessionId(sid);
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
      compactionSourceId: null,
      liveProgress: null,
      awaitingInput: false,
    });
  },

  _registerSupersession(oldId, newId) {
    if (!oldId || !newId || oldId === newId) return;
    const map = this._supersededMap;
    map[oldId] = newId;
    // Path-compress any existing entries that pointed at oldId so subsequent
    // _resolveSessionId calls land on newId in a single hop instead of walking
    // a growing chain after multiple compactions of the same session.
    for (const k in map) if (map[k] === oldId) map[k] = newId;
    const oldState = this.sessionsState[oldId];
    if (!oldState) return;
    const newState = this.sessionsState[newId];
    if (!newState) {
      this.sessionsState[newId] = oldState;
    } else {
      // NEW may already have its own state (loadSessions pre-creates one). Carry
      // OLD's mid-stream messages to the tail since they happened most recently;
      // OR-merge flags so a flipped-on `sending` from the in-flight closure isn't
      // dropped; only overwrite skills/statusInfo if NEW hasn't seen its own yet.
      newState.messages.push(...oldState.messages);
      newState.sending = newState.sending || oldState.sending;
      newState.reader = newState.reader || oldState.reader;
      newState.liveProgress = newState.liveProgress || oldState.liveProgress;
      if (oldState.loadedSkills?.length && !newState.loadedSkills.length) {
        newState.loadedSkills = oldState.loadedSkills;
      }
      const newEmpty = newState.statusInfo && !Object.keys(newState.statusInfo).length;
      const oldFilled = oldState.statusInfo && Object.keys(oldState.statusInfo).length;
      if (newEmpty && oldFilled) newState.statusInfo = oldState.statusInfo;
    }
    delete this.sessionsState[oldId];
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
  get compactionSourceId() {
    return this.sessionsState[this.selectedSessionId]?.compactionSourceId || null;
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
    // /run is wired client-side (input.js) and isn't in the server-side
    // adapter command registry, so synthesise it into the suggestion list.
    const RUN_VIRTUAL = { name: 'run', description: 'open a PTY-backed terminal session', plugin: '' };
    const all = this.availableCommands.some(c => c.name === 'run')
      ? this.availableCommands
      : [...this.availableCommands, RUN_VIRTUAL];
    return all.filter(c => c.name.startsWith(query));
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
      // Hide Job worker / verifier sessions from the sidebar - they're
      // represented by the tile in the parent chat. Users navigate to them by
      // clicking the tile (which calls selectSessionById on the id), not via
      // the sidebar. Without this filter, every job spawns 2 extra sidebar
      // rows that clutter the view.
      if (s.metadata && s.metadata.job_id) continue;
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
    // e2e hook: `this` here is the TRUE reactive context. Alpine.$data()
    // hands tests a wrapper whose nested writes can miss the dependency
    // graph, so Playwright helpers drive the view through this reference.
    window.__tsugiteConv = this;
    this._loadCommands();
    this._mobileQuery = window.matchMedia('(max-width: 640px)');
    const maybeReload = () => {
      if (this.$store.app.view === 'conversations' && this.$store.app.selectedAgent) this.reload();
    };
    this.$watch('$store.app.selectedAgent', maybeReload);
    this.$watch('$store.app.view', (view) => {
      if (view === 'conversations' && this.$store.app.selectedAgent) this.reload();
    });
    // Session-only back/forward keeps view==='conversations', so the view
    // watcher above never fires and reload() never re-consumes viewSessionId.
    // Handle it here. The equality guard prevents re-entry when a sidebar click
    // writes the hash, re-setting viewSessionId to the now-selected id.
    this.$watch('$store.app.viewSessionId', (targetId) => {
      if (!targetId || this.$store.app.view !== 'conversations') return;
      if (targetId === this.selectedSessionId) return;
      this.openViewSession(targetId, { follow: false });
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
        if (d.action === 'busy' && d.id) {
          // Patch in place - a turn-start must not trigger a full session-list
          // refetch (turn completion already does, via action 'updated').
          const s = this.allSessions.find(x => x.id === d.id);
          if (s) s.busy = !!d.busy;
          if (this.selectedSessionMeta && (this.selectedSessionMeta.conversation_id || this.selectedSessionMeta.id) === d.id) {
            this.selectedSessionMeta = { ...this.selectedSessionMeta, busy: !!d.busy };
          }
          return;
        }
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
        if (d.action === 'compacted' && d.successor_id) {
          this._registerSupersession(d.id, d.successor_id);
          if (this.selectedSessionId === d.id) {
            // The session the user is looking at just compacted; follow the chain forward
            // so the next message doesn't post to a now-completed predecessor.
            this.loadSessions().then(() => this.selectSessionById(d.successor_id));
            return;
          }
        }
        this._debouncedLoadSessions();
      }
      if (ev.type === 'session_event') {
        this._updateProgressCache(d);
        this._handleSessionEvent(d);
      }
      if (ev.type === 'job_update') {
        this._handleJobUpdate(d);
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
    // Watchdog for stranded progress bubbles: a live "Working..." that gets no
    // events while the server says not-busy (missed terminal event after a
    // disconnect/sleep) is finalized instead of spinning forever.
    this._staleProgressTimer = setInterval(() => { this._reconcileStaleProgress(); }, 8000);
    // On tab refocus, mark the selected session viewed AND refetch its history:
    // the SSE reconnect on resume suppresses its 'reconnect' catch-up event (fresh
    // closure) and missed events aren't replayed, so the live stream can't be trusted
    // to have delivered messages that arrived while backgrounded (e.g. a reply from
    // another device).
    this._onVisibilityChange = () => {
      if (document.visibilityState !== 'visible' || !this.selectedSessionId) return;
      const s = this.allSessions.find(x => (x.conversation_id || x.id) === this.selectedSessionId);
      if (s && s.unread) this._markSessionViewed(s);
      // Only catch up while following the tail: loadHistory force-scrolls to the
      // bottom, so refetching when the user has scrolled up into history would yank
      // them down. !sending mirrors history_update (don't race a turn); the 200ms
      // debounce coalesces rapid tab-flips into a single refetch.
      if (this.isActiveSession && !this.sending && this.isAtBottom) this._debouncedLoadHistory();
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
    if (this._staleProgressTimer) clearInterval(this._staleProgressTimer);
    if (this._onVisibilityChange) document.removeEventListener('visibilitychange', this._onVisibilityChange);
    if (this._followObserver) this._followObserver.disconnect();
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
      this.openViewSession(targetId);
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
    const sid = this.selectedSessionId;
    const qs = sid ? `?session_id=${encodeURIComponent(sid)}` : '';
    try {
      const data = await get(`/api/agents/${agent}/effort-levels${qs}`);
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
      // The new model may support a different set of effort levels, so refresh
      // the dropdown and clamp a now-unsupported selection to a valid level.
      await this.loadEffortLevels();
      await this._clampSessionEffort();
    } catch (e) {
      console.warn('Failed to save model override', e);
      toast(`model change failed: ${e.message || 'unknown'}`, 'error');
    }
  },

  async _clampSessionEffort() {
    const current = this.sessionEffort;
    if (!current || this.effortLevels.includes(current)) return;
    // Prefer the highest supported level at or below the current one, so we
    // never silently raise effort; fall back to the lowest supported otherwise.
    const rank = EFFORT_ORDER.indexOf(current);
    const supported = EFFORT_ORDER.filter((lvl) => this.effortLevels.includes(lvl));
    const atOrBelow = supported.filter((lvl) => EFFORT_ORDER.indexOf(lvl) <= rank);
    const target = (atOrBelow.length ? atOrBelow[atOrBelow.length - 1] : supported[0]) || '';
    await this.setSessionEffort(target);
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
  // Belt-and-suspenders for a live progress bubble whose terminal event never
  // arrived (SSE gap: laptop sleep, wifi blip, daemon restart). If the bubble
  // has been quiet past the threshold and the server reports not-busy, settle
  // it and reload the real history.
  async _reconcileStaleProgress() {
    const sid = this.selectedSessionId;
    const agent = this.$store.app.selectedAgent;
    if (!sid || !agent || !this._sessionProgress) return;
    const state = this._sessionState(sid);
    if (state.sending) return;
    const last = state.progress?.lastEventTime;
    // Only act when we KNOW events flowed and then stopped; a bubble with no
    // event timestamp yet is a fresh/rehydrating turn the other reconcile
    // paths own (and acting on it would misfire right after a send starts).
    if (!last || Date.now() - Date.parse(last) < 15000) return;
    try {
      const data = await get(`/api/agents/${agent}/status?user_id=${encodeURIComponent(this.userId)}&session_id=${encodeURIComponent(sid)}`);
      if (data.busy) return;
      // Re-check after the await: an event may have landed or the view swapped.
      if (this.selectedSessionId !== sid || !this._sessionProgress) return;
      this._sessionProgress.type = 'progress-done';
      this._sessionProgress = null;
      if (state.progress) state.progress.statusText = '';
      const s = this.allSessions.find(x => x.id === sid);
      if (s) s.busy = false;
      this._debouncedLoadHistory();
    } catch { /* transient; next tick retries */ }
  },

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
        // Track this as the live progress bubble so the next session_event reuses it
        // instead of pushing a second one that spins "Working..." forever.
        this._sessionProgress = { type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 };
        this.messages.push(this._sessionProgress);
        this.scrollMessages();
      }
    } catch { /* ignore */ }
  },

  updateStatusFromEvent(event, sessionId) {
    const sid = sessionId || this.selectedSessionId;
    if (!sid) return;
    // Merge, don't replace: a partial session_info (interrupted-turn/resume
    // paths) must not wipe previously-good tokens/context_limit - that blanks
    // the context bar until the next full /status fetch.
    const prev = this._sessionState(sid).statusInfo || {};
    this._sessionState(sid).statusInfo = {
      model: event.model ?? prev.model,
      tokens: event.tokens ?? prev.tokens,
      context_limit: event.context_limit ?? prev.context_limit,
      threshold: event.threshold ?? prev.threshold,
      message_count: event.message_count ?? prev.message_count,
      attachments: event.attachments ?? prev.attachments,
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
    this.$store.tsu.open('attachments');
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
    return this.sessionCompactingLabel(this.selectedSessionMeta);
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
    // Clear the sidebar "waiting" flag immediately; the agent's resume event
    // (final_result) isn't broadcast, so don't wait for _updateProgressCache.
    const st = this.sessionsState[this.selectedSessionId];
    if (st) st.awaitingInput = false;
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

  copySessionId() { copyText(this.selectedSessionId); },

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
      this._ensureFollowObserver(el);
      if (force || this.isAtBottom) scrollToBottom(el);
    });
  },

  // While the user is at the bottom, keep the view pinned as content streams in
  // and as markdown/code blocks render late: the one-shot scroll above can fire
  // before late layout settles, so it undershoots the true bottom. The observer
  // re-pins on any content mutation (rAF-throttled), and only while isAtBottom,
  // so a user who has scrolled up is never yanked back down.
  _ensureFollowObserver(el) {
    if (this._followObserver) return;
    let scheduled = false;
    this._followObserver = new MutationObserver(() => {
      if (!this.isAtBottom || scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        const m = this.$refs.messages;
        if (m && this.isAtBottom) scrollToBottom(m);
      });
    });
    this._followObserver.observe(el, { childList: true, subtree: true, characterData: true });
  },

  _updateProgressCache(d) {
    const id = d.session_id;
    if (!id) return;
    const state = this._sessionState(id);
    const evType = d.event_type;
    // A session blocked on ask_user is waiting on the user. The agent emits no
    // further events until they reply, so the next event clears the flag. This
    // runs for every broadcast session_event, so it flags sessions the user
    // isn't currently viewing too.
    state.awaitingInput = evType === 'ask_user';
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
    // The mute window covers stragglers delivered just after `sending` clears.
    const st = this.sessionsState[d.session_id];
    if (st?.sending) return;
    if (st?.passiveMuteUntil && Date.now() < st.passiveMuteUntil && !TURN_END_EVENTS.has(d.event_type) && !SESSION_END_EVENTS.has(d.event_type)) return;

    const evType = d.event_type;
    // Finalize the in-flight bubble directly on turn end. The paired
    // history_update reload reconciles the CONTENT, but it's conditional
    // (isActiveSession, sending) and can be dropped entirely - the spinner
    // must not depend on it. Content is never pushed here, so if the live
    // reader already finalized, _sessionProgress is null and this is a no-op.
    if (TURN_END_EVENTS.has(evType)) {
      if (this._sessionProgress) {
        this._sessionProgress.type = 'progress-done';
        if (evType === 'error') {
          this._sessionProgress.failed = true;
          this._sessionProgress.errorText = d.error || 'Turn failed';
        }
        this._sessionProgress = null;
        this._scrollThrottled();
      }
      return;
    }

    if (SESSION_END_EVENTS.has(evType)) {
      // Only finalize an in-flight progress bubble; don't materialize an empty one
      // for a session that ended with no progress shown in this tab (which would
      // render a phantom "code ✓/✗" bubble with zero steps).
      if (this._sessionProgress) {
        this._sessionProgress.type = 'progress-done';
        if (evType === 'session_error') {
          this._sessionProgress.failed = true;
          this._sessionProgress.errorText = d.error || 'Session failed';
        }
        this._sessionProgress = null;
      }
      this._debouncedLoadSessions();
      if (evType === 'session_complete' && d.result_preview) {
        this.messages.push({ type: 'agent', text: d.result_preview });
      }
      this._scrollThrottled();
      return;
    }

    if (!this._sessionProgress) {
      this._sessionProgress = { type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 };
      this.messages.push(this._sessionProgress);
    }

    // Guard: only forward while the live bubble is still rendered (selectSession
    // / reload may have swapped the messages array).
    if (this.messages.includes(this._sessionProgress)) {
      const { session_id, event_type, ...rest } = d;
      this._handleProgressEvent({ type: evType, ...rest }, this.selectedSessionId);
      this._scrollThrottled();
    }
  },

  _handleJobUpdate(d) {
    if (!d.parent_session_id) return;
    // Route the update to the OWNING session's message list, not just the one
    // currently on screen. A job's running→verifying→done updates arrive while
    // the user may be viewing the worker/verifier session (e.g. via "open
    // worker"); dropping them there left the tile stuck on its last-seen state
    // when they returned, because revisits don't reload history.
    const owner = this._resolveSessionId(d.parent_session_id);
    const state = this.sessionsState[owner];
    if (!state) return;  // parent not loaded in memory; the tile renders fresh from history on next visit
    const existing = state.messages.find(m => m.type === 'job_status' && m.job_id === d.job_id);
    // Drop undefined fields so an emit without `error` (e.g. a RUNNING tick after
    // a STUCK terminal) doesn't wipe a previously-set error from the tile.
    // acceptance_criteria/result are consumed when present - the backend doesn't
    // include them in _emit_job_event today, but the tile is forward-compatible.
    // ac_results is broadcast top-level during VERIFYING so mid-verify criteria reach the tile.
    const fields = {};
    for (const k of ['state', 'prompt', 'worker_session_id', 'worker_terminal_id', 'verifier_session_id', 'verify_attempts', 'error', 'attempts', 'acceptance_criteria', 'result', 'ac_results']) {
      if (d[k] !== undefined) fields[k] = d[k];
    }
    if (existing) {
      Object.assign(existing, fields);
    } else if (owner === this.selectedSessionId) {
      // Only spawn a brand-new tile in the session being viewed (a job just
      // created in this chat). Background sessions already carry their tile.
      state.messages.push({ type: 'job_status', job_id: d.job_id, ...fields });
      this._scrollThrottled();
    }
  },

  // ---------- Job tile helpers (template-side) ----------

  // Default-open the tile for live/attention states; collapse done/cancelled/queued.
  // Sticky once the user toggles, via msg._open.
  jobTileOpen(msg) {
    if (typeof msg._open === 'boolean') return msg._open;
    return msg.state === 'running' || msg.state === 'verifying'
      || msg.state === 'stuck' || msg.state === 'errored';
  },

  jobStateLabel(state) {
    return state === 'errored' ? 'error' : (state || 'unknown');
  },

  jobShortId(jobId) {
    if (!jobId) return '';
    // Strip the "job-" prefix and keep the hex suffix; matches the design's
    // "job_8e3b" style without baking in the underscore separator.
    const cleaned = String(jobId).replace(/^job[-_]/, '');
    return cleaned.length > 8 ? cleaned.slice(0, 8) : cleaned;
  },

  jobMetaLine(msg) {
    const bits = [];
    const attempts = (msg.attempts || []).length;
    if (attempts > 1) bits.push(`${attempts} attempts`);
    else if (msg.verify_attempts) bits.push(`attempt ${msg.verify_attempts + 1}`);
    return bits.join(' · ');
  },

  // Per-criterion AC chip list. Reads `acceptance_criteria` (list of strings)
  // from the job event payload. Each chip's status is derived from
  // `result.ac_results` when available - matched first by `ac_index`, then by
  // `ac_text` as a fallback. Otherwise pending (or `active` for the first AC
  // while the verifier is mid-flight).
  jobCriteria(msg) {
    return jobCriteriaStates(msg);
  },

  jobResultSummary(msg) {
    if (!msg.result) return '';
    if (typeof msg.result === 'string') return msg.result;
    return msg.result.summary || '';
  },

  jobHasFooter(msg) {
    if (!msg) return false;
    if (msg.state === 'running' || msg.state === 'verifying') return true;
    if (msg.state === 'stuck' || msg.state === 'errored') return true;
    return this.jobHasTerminal(msg);
  },

  // A worker terminal exists when the backend supplied its id directly (a cc
  // executor spawns no worker Session, so it carries only worker_terminal_id)
  // or when a worker session ran (agent executor - jobTerminalView probes
  // /api/terminals for its PTY, if the worker produced one).
  jobHasTerminal(msg) {
    return !!(msg && (msg.worker_terminal_id || msg.worker_session_id));
  },

  // ---------- Job tile actions (dialogs) ----------

  async jobCancel(jobId) {
    // Cancel is reversible enough - no confirm dialog per the design.
    try {
      await post(`/api/jobs/${jobId}/cancel`, {});
    } catch (e) {
      toast(`Cancel failed: ${e.message}`, 'error');
    }
  },

  jobMarkDone(jobId) {
    const msg = this._findJobMessage(jobId);
    const criteria = msg ? this.jobCriteria(msg) : [];
    const failed = criteria.filter(c => c.status === 'fail');
    const passed = criteria.filter(c => c.status === 'pass');
    this.markDoneDialog.jobId = jobId;
    this.markDoneDialog.prompt = msg?.prompt || '';
    this.markDoneDialog.criteria = [...failed, ...passed];  // failed first, design's recap order
    this.markDoneDialog.reason = '';
    this.$store.tsu.open('mark-done');
  },

  closeMarkDoneDialog() {
    this.$store.tsu.close('mark-done');
  },

  async submitMarkDoneDialog() {
    const { jobId, reason } = this.markDoneDialog;
    this.$store.tsu.close('mark-done');
    try {
      await post(`/api/jobs/${jobId}/mark-done`, { reason: reason.trim() || 'marked done by user' });
    } catch (e) {
      toast(`Mark done failed: ${e.message}`, 'error');
    }
  },

  jobRetryWithHint(jobId) {
    const msg = this._findJobMessage(jobId);
    // Pull the last verifier verdict from result.ac_results (first failing reason)
    // when available; fall back to the job's error string.
    let lastVerdict = '';
    if (msg?.result?.ac_results && Array.isArray(msg.result.ac_results)) {
      const firstFail = msg.result.ac_results.find(r => r && r.pass === false);
      if (firstFail) lastVerdict = firstFail.reason || firstFail.ac_text || '';
    }
    if (!lastVerdict && msg?.error) lastVerdict = String(msg.error).split('\n')[0].slice(0, 200);
    this.retryDialog.jobId = jobId;
    this.retryDialog.prompt = msg?.prompt || '';
    this.retryDialog.lastVerdict = lastVerdict;
    this.retryDialog.hint = '';
    this.retryDialog.model = '';
    this.retryDialog.currentModel = msg?.model || '';
    this.retryDialog.resetCounter = true;
    this.retryDialog.workspace = 'keep';
    this.$store.tsu.open('retry-hint');
    this.$nextTick(() => {
      try { this.$refs.retryHintText?.focus(); } catch { /* non-fatal */ }
    });
  },

  closeRetryDialog() {
    this.$store.tsu.close('retry-hint');
  },

  async submitRetryDialog() {
    const hint = (this.retryDialog.hint || '').trim();
    const model = (this.retryDialog.model || '').trim();
    // Hint OR model: retrying purely to switch models (usage-limit death)
    // shouldn't force inventing a hint.
    if (!hint && !model) return;
    const jobId = this.retryDialog.jobId;
    const body = {
      reset_counter: !!this.retryDialog.resetCounter,
      fresh_workspace: this.retryDialog.workspace === 'fresh',
    };
    if (hint) body.hint = hint;
    if (model && model !== this.retryDialog.currentModel) body.model = model;
    this.$store.tsu.close('retry-hint');
    try {
      await post(`/api/jobs/${jobId}/retry`, body);
    } catch (e) {
      toast(`Retry failed: ${e.message}`, 'error');
    }
  },

  _findJobMessage(jobId) {
    if (!jobId) return null;
    return this.messages.find(m => m.type === 'job_status' && m.job_id === jobId) || null;
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
  // The one client-side verdict for "is a turn in flight": the server's
  // authoritative busy flag, or fresh live progress (covers the moments before
  // the busy broadcast lands). Every consumer (dot, label, history dropTrailing)
  // must use this instead of re-deriving its own version.
  sessionTurnInFlight(s) {
    return !!(s && (s.busy || this.isSessionProgressFresh(s)));
  },

  dotClassNames(s) {
    // A session waiting on the user gets a distinct (peach), attention-pulsing dot
    // so it reads differently from the green running/working state.
    if (this._isAwaitingInput(s)) return 'awaiting pulse';
    // Server-side busy is authoritative: render the running state even when
    // this client has no fresh progress (reconnect / PWA resume).
    const base = s.busy ? this.statusDotClass('running') : this.statusDotClass(s.state);
    return this.sessionTurnInFlight(s) ? `${base} pulse` : base;
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
