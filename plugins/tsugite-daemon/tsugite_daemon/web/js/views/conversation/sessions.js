import { get, post, patch, del } from '../../api.js';
import { formatDate } from '../../utils.js';
import { REPLAY_SKIP_EVENTS, SESSION_END_EVENTS, progressFromPayload } from './event_types.js';

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
  editingTopicSessionId: null,
  editingTopic: '',
  topicError: '',
  draggingPinId: null,
  dragOverPinId: null,

  _debouncedLoadSessions() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    this._debounceTimer = setTimeout(() => this.loadSessions(), 200);
  },

  _selectedSessionStorageKey() {
    const agent = this.$store.app.selectedAgent;
    return agent ? `tsugite_selected_session_${agent}` : null;
  },

  _loadPersistedSessionId() {
    const key = this._selectedSessionStorageKey();
    if (!key) return null;
    try { return localStorage.getItem(key); } catch { return null; }
  },

  _persistSelectedSession(convId) {
    const key = this._selectedSessionStorageKey();
    if (!key) return;
    try {
      if (convId) localStorage.setItem(key, convId);
      else localStorage.removeItem(key);
    } catch { /* quota or disabled */ }
  },

  async loadSessions() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) { this.loading = false; return; }
    const cacheKey = `tsugite_sessions_${agent}`;
    // Stale-while-revalidate: paint the cached list immediately on cold load /
    // PWA resume so the sidebar isn't blank while we wait for the network.
    if (this.allSessions.length === 0) {
      try {
        const cached = JSON.parse(localStorage.getItem(cacheKey) || 'null');
        if (Array.isArray(cached) && cached.length) {
          this.allSessions = cached;
          this.loading = false;
        }
      } catch { /* corrupt cache — ignore and fetch fresh */ }
    }
    try {
      // include_superseded=true so allSessions has the chain info; we filter
      // them out of groupedSessions for the sidebar but keep them indexable
      // here so selectSession can chase superseded_by to the live successor.
      const data = await get(`/api/agents/${agent}/sessions?include_superseded=true`);
      this.allSessions = (data.sessions || []).map(s => ({ ...s, state: s.state || s.status }));
      const liveIds = new Set();
      for (const s of this.allSessions) {
        if (s.state !== 'running' && s.state !== 'active') continue;
        liveIds.add(s.id);
        const state = this._sessionState(s.id);
        // Server clears status_text on turn-end; reconcile so a missed SSE can't keep a stale cache alive.
        if (s.progress?.status_text === '') {
          state.progress = null;
        } else if (s.progress && !state.progress) {
          state.progress = progressFromPayload(s.progress);
        }
      }
      for (const [id, st] of Object.entries(this.sessionsState)) {
        if (st.progress && !liveIds.has(id)) st.progress = null;
      }
      try { localStorage.setItem(cacheKey, JSON.stringify(this.allSessions)); } catch { /* quota or disabled */ }
    } catch {
      // Network failed — keep whatever is currently displayed (cache or last successful fetch)
      // instead of dropping back to []; that's the empty-flash bug on PWA resume.
    }
    this.loading = false;
  },

  _isMyInteractive(s) {
    return s.source === 'interactive' && (s.user_id === this.userId || s.conversation_id === this.userId);
  },

  findSession(id) {
    return this.allSessions.find(s => s.id === id);
  },

  selectSessionById(id, opts = {}) {
    const s = this.findSession(id);
    if (s) this.selectSession(s, opts);
  },

  // Resolve a viewSessionId (a conversation id) to a loaded session and select
  // it, falling back to a synthetic stub when the target isn't loaded yet.
  // Matches on conversation_id OR id (unlike findSession's id-only match), so
  // reload() and the viewSessionId watcher resolve the hash the same way.
  openViewSession(targetId, opts = {}) {
    const match = this.allSessions.find((s) => s.conversation_id === targetId || s.id === targetId);
    this.selectSession(match || { conversation_id: targetId, agent: this.$store.app.selectedAgent }, opts);
  },

  autoSelectInteractive() {
    // Pinned-and-mine wins over a recently-fired schedule run; if no pinned
    // interactive session exists, fall back to any of mine, then to whatever's
    // active, then recent.
    const g = this.groupedSessions;
    const mine = (s) => this._isMyInteractive(s);
    const first = g.pinned.find(mine) || g.active.find(mine) || g.active[0] || g.recent[0];
    if (first) this.selectSession(first);
  },

  async selectSession(session, opts = {}) {
    // Selecting a chat clears any terminal selection so the two sidebar
    // surfaces stay mutually exclusive in the main pane.
    this.$store.terminals?.deselectTerminal();
    // Compaction marks the old session completed and stamps superseded_by on it.
    // The localStorage sidebar cache may still show the old one as pinned on
    // cold load, so chase the chain to land on the live continuation. Pass
    // {follow: false} from the back-link affordance on a new session's
    // compaction summary bubble — that explicitly wants to land on the
    // predecessor, not auto-forward back.
    const follow = opts.follow !== false;
    if (follow) {
      const visited = new Set();
      while (session?.superseded_by && !visited.has(session.id)) {
        visited.add(session.id);
        const next = this.allSessions.find(s => s.id === session.superseded_by);
        if (!next) break;
        session = next;
      }
    }
    this.sidebarOpen = false;
    this._saveDraftNow();
    const convId = session.conversation_id || session.id;
    this.selectedSessionId = convId;
    this.selectedSessionMeta = session;
    this._persistSelectedSession(convId);
    const state = this._sessionState(convId);
    const isFirstVisit = state.messages.length === 0;
    this.resetHistory();

    const hash = `conversations?session=${encodeURIComponent(convId)}`;
    if (location.hash.slice(1) !== hash) location.hash = hash;

    this.isActiveSession = this._isMyInteractive(session);
    // Interactive sessions stay state='active' between turns, so state alone
    // isn't a "turn in flight" signal. status_text is non-empty only mid-turn.
    // lastEventTime gates against the never-started case where the server
    // returns its default "Starting..." for sessions with zero events.
    const cached = state.progress || progressFromPayload(session.progress);
    // Server busy counts as a live turn: a mid-turn reconnect must defer the
    // trailing bubble to the live stream exactly like a locally-observed turn.
    const liveTurn = (!!cached?.statusText && !!cached?.lastEventTime) || !!session.busy;
    this.loadStatus();
    // resetHistory() above wipes loadHistory-derived state (compactionSummary,
    // pagination: _allHistoryMessages / hasMoreHistory). On a revisit we must
    // rebuild it, so reload unless a live turn owns the in-memory bubbles - while
    // sending (our own stream) or mid-turn (liveTurn), loadHistory would clobber
    // the streaming trace, so skip and let the history_update SSE handler refresh.
    const reloadHistory = isFirstVisit || (!state.sending && !liveTurn);
    const historyPromise = reloadHistory ? this.loadHistory({ dropTrailing: liveTurn }) : Promise.resolve();
    this.loadSessionEffort();
    // effortLevels is a single shared field, so re-derive it for the newly
    // selected session's model; otherwise it keeps the prior session's levels.
    this.loadEffortLevels();
    this._restoreDraft();
    this._markSessionViewed(session);
    // Revisits too, not just first visits: the "turn in flight" signal that
    // gates reloadHistory can be stale (crashed turn, dropped terminal event),
    // and a revisit that skips both the reload AND this reconcile strands the
    // old "Working..." bubble forever.
    if (!state.sending && liveTurn) {
      await historyPromise;
      if (this.selectedSessionId !== convId) return;
      this._rehydrateProgressFromEvents(convId);
    }
    // Revisits skip loadHistory's scroll-to-bottom; force it so sidebar clicks always land at the latest message.
    if (!isFirstVisit) this.scrollMessages(true);
  },

  async _markSessionViewed(session) {
    if (!session || !session.id || !session.unread) return;
    session.unread = false;
    try {
      await post(`/api/sessions/${session.id}/mark-viewed`, {});
    } catch { /* non-fatal */ }
  },

  async _rehydrateProgressFromEvents(sessionId) {
    let progress = this._sessionProgress;
    if (progress) {
      if (!this.messages.includes(progress)) this.messages.push(progress);
      // The replay below rebuilds the bubble from the event log; reset so a
      // revisit doesn't duplicate steps already rendered in this bubble.
      progress.steps = [];
      progress.turnCount = 0;
      progress.toolCount = 0;
    } else {
      progress = { type: 'progress', steps: [], statusText: 'Starting...', turnCount: 0, toolCount: 0 };
      this._sessionProgress = progress;
      this.messages.push(progress);
    }
    try {
      const data = await get(`/api/sessions/${sessionId}/events`);
      // Bail if reload/backToSessions/another selectSession swapped our bubble out while the fetch was in flight.
      if (this.selectedSessionId !== sessionId || !this.messages.includes(progress)) return;
      const events = data.events || [];
      // The "turn in flight" signal that got us here can be stale: a reload can
      // race the turn's completion (the session record's status_text clears a
      // beat after the final event lands). If the log says the turn already
      // ended, render the settled history instead of a perpetual running bubble.
      let ended = false;
      for (const ev of events) {
        if (ev.type === 'user_input' || ev.type === 'turn_start') ended = false;
        if (ev.type === 'final_result' || SESSION_END_EVENTS.has(ev.type)) ended = true;
      }
      if (ended) {
        this._sessionProgress = null;
        await this.loadHistory({ dropTrailing: false });
        return;
      }
      for (const ev of events) {
        if (REPLAY_SKIP_EVENTS.has(ev.type)) continue;
        // JSONL events are {type, ts, data}; the progress handler takes the
        // flattened shape the live stream uses.
        this._handleProgressEvent({ type: ev.type, ...(ev.data || {}) }, sessionId);
      }
      this.scrollMessages();
    } catch { /* ignore */ }
  },

  backToSessions() {
    this._saveDraftNow();
    this.selectedSessionId = null;
    this.selectedSessionMeta = null;
    this.resetHistory();
    this.isActiveSession = true;
    this._persistSelectedSession(null);
  },

  async newSession() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    try {
      const data = await post(`/api/agents/${agent}/sessions/new`, { user_id: this.userId });
      await this.loadSessions();
      const session = this.allSessions.find(s => s.id === data.id);
      if (session) {
        this.selectSession(session);
        this.startEditTitle(session, { stopPropagation() {} }, { blank: true });
      }
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
    if (s.id && s.id.startsWith('sched_')) {
      const parts = s.id.replace(/^sched_/, '').split('_');
      const dateIdx = parts.findIndex(p => /^\d{8}$/.test(p));
      const name = dateIdx > 0 ? parts.slice(0, dateIdx).join('-') : parts[0];
      return name || s.id;
    }
    // Fallback priority is date > prompt > short id > username/label so a sidebar
    // full of untitled sessions stays distinguishable instead of repeating the
    // same username or "Web: web-…" prefix once per row.
    const dated = this._sessionDateLabel(s);
    if (dated) return dated;
    const prompt = (s.prompt || '').trim();
    if (prompt) return prompt.length > 50 ? prompt.slice(0, 50).trimEnd() + '…' : prompt;
    if (s.id) return s.id.length > 14 ? s.id.slice(0, 12) + '…' : s.id;
    return s.label || s.conversation_id || 'unknown';
  },

  _sessionDateLabel(s) {
    const iso = s.created_at || s.last_active;
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    const now = new Date();
    const sameDay = d.getFullYear() === now.getFullYear()
      && d.getMonth() === now.getMonth()
      && d.getDate() === now.getDate();
    return sameDay
      ? formatDate(iso)
      : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  },

  startEditTitle(s, event, { blank = false } = {}) {
    event.stopPropagation();
    this.editingSessionId = s.id;
    this.editingTitle = blank ? '' : (s.title || this.sessionLabel(s));
    // Alpine may render the input async (after newSession()), and selectSession's
    // later reflows can steal focus right after we set it. Poll up to 30 attempts
    // (~1.5s) covering both "input not in DOM yet" and "focus stolen post-mount".
    let tries = 0;
    const focusInput = () => {
      if (this.editingSessionId !== s.id) return;  // user navigated away mid-poll
      const input = document.querySelector('.title-input');
      if (input) {
        input.focus();
        input.select();
      }
      if ((!input || document.activeElement !== input) && tries++ < 30) {
        setTimeout(focusInput, 50);
      }
    };
    this.$nextTick(focusInput);
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

  startEditTopic(s, event) {
    event.stopPropagation();
    this.topicError = '';
    this.editingTopicSessionId = s.id;
    this.editingTopic = s.metadata?.topic || '';
    let tries = 0;
    const focusInput = () => {
      if (this.editingTopicSessionId !== s.id) return;
      const input = document.querySelector('.topic-input');
      if (input) {
        input.focus();
        input.select();
      }
      if ((!input || document.activeElement !== input) && tries++ < 30) {
        setTimeout(focusInput, 50);
      }
    };
    this.$nextTick(focusInput);
  },

  async saveTopic(s) {
    const next = this.editingTopic.trim();
    const current = s.metadata?.topic || '';
    this.editingTopicSessionId = null;
    if (next === current) return;
    try {
      if (next === '') {
        const data = await del(`/api/sessions/${s.id}/metadata/topic`);
        s.metadata = data.metadata || {};
      } else {
        const data = await patch(`/api/sessions/${s.id}/metadata`, { topic: next });
        s.metadata = data.metadata || { ...(s.metadata || {}), topic: next };
      }
      this.topicError = '';
    } catch (e) {
      this.topicError = e?.message || 'Failed to update topic';
      console.error('Failed to update topic', e);
    }
  },

  sourceIcon(source) {
    return { interactive: 'I', web: 'W', discord: 'D', cli: '>', schedule: 'S',
             background: 'B', spawned: 'P' }[source] || '?';
  },

  metadataChips(s, opts = {}) {
    const meta = s.metadata || {};
    const skip = new Set(opts.excludeKeys || []);
    const chips = [];
    if (meta.type && !skip.has('type')) chips.push({ label: meta.type, cls: 'chip-type' });
    if (meta.status_text && !skip.has('status_text')) chips.push({ label: meta.status_text, cls: 'chip-status' });
    if (meta.topic && !skip.has('topic')) chips.push({ label: meta.topic, cls: 'chip-topic', title: meta.topic });
    if (meta.task && !skip.has('task')) chips.push({ label: 'Task', href: meta.task, cls: 'chip-link' });
    if (meta.pr && !skip.has('pr')) chips.push({ label: 'PR', href: meta.pr, cls: 'chip-link' });
    return chips;
  },

  _inlineMeta(s, key, max) {
    return (s?.metadata?.[key] || '').replace(/\s+/g, ' ').trim().slice(0, max);
  },

  inlineStatusText(s) { return this._inlineMeta(s, 'status_text', 40); },

  inlineTopic(s) { return this._inlineMeta(s, 'topic', 80); },

  isSessionSelected(s) {
    return this.selectedSessionId === (s?.conversation_id || s?.id);
  },

  lastMessagePreview(s) {
    return (s.result || s.prompt || '').slice(0, 60) || '';
  },

  // True when the session is blocked on the user: an ask_user prompt (set in
  // _updateProgressCache, cleared on reply), a job paused on a question
  // (awaiting_input), or a cc permission prompt (needs_attention).
  _isAwaitingInput(s) {
    const st = this.sessionsState[s?.id];
    if (!st) return false;
    return !!st.awaitingInput || Object.keys(st.jobsNeedingInput || {}).length > 0;
  },

  // Aggregate "needs your input" count across the loaded session list, so
  // blocked sessions are discoverable at a glance without scanning every row.
  // A method, not a getter: mixins merge by object spread, which would freeze
  // a getter's first value.
  awaitingCount() {
    return (this.allSessions || []).filter((s) => this._isAwaitingInput(s)).length;
  },

  // Class for the row2 live-label: a waiting session gets its own (peach) treatment
  // so it doesn't read as the green running/working state.
  liveLabelClass(s) {
    return this._isAwaitingInput(s) ? 'awaiting' : (s?.state || '');
  },

  sessionProgressLabel(s) {
    if (!s) return '';
    if (this._isAwaitingInput(s)) {
      const st = this.sessionsState[s.id];
      if (!st?.awaitingInput) return 'a job needs your input';
      return 'waiting for your reply';
    }
    const compacting = this.sessionCompactingLabel(s);
    if (compacting) return compacting;
    const running = s.state === 'running' || s.state === 'active';
    if (!running) return '';
    const cached = this.sessionsState[s.id]?.progress || progressFromPayload(s.progress);
    const parts = [];
    if (cached?.turnCount) parts.push(`Turn ${cached.turnCount}`);
    if (cached?.toolCount) parts.push(`${cached.toolCount} tool${cached.toolCount > 1 ? 's' : ''}`);
    if (cached?.statusText) parts.push(cached.statusText);
    if (parts.length > 0) return parts.join(' · ');
    // The server's busy flag is authoritative: a turn is in flight even if no
    // progress event has reached this client (reconnect, PWA resume, missed
    // broadcasts) - never show an idle session while the server says busy.
    if (s.busy) return 'Working...';
    // lastEventTime distinguishes "between turns" (events seen, none active) from
    // "session never started" so the idle state doesn't fall back to "Starting...".
    return cached?.lastEventTime ? '' : 'Starting...';
  },

  sessionCompactingLabel(s) {
    if (!s) return '';
    const st = this.sessionsState[s.id];
    if (!st?.compacting) return '';
    const c = st.compactingCounts;
    const phase = st.compactingPhase;
    let base;
    if (c && c.replaced_count != null) {
      // != null catches valid replaced_count=0 (rare but real for sessions
      // compacted again after barely any new content), which strict truthy
      // would route to the same plain "compacting" string the pre-counts
      // window uses, so the two states became indistinguishable.
      const n = c.replaced_count;
      base = `summarizing ${n} turn${n === 1 ? '' : 's'}`;
      if (c.retained_count) base += `, keeping ${c.retained_count}`;
    } else {
      base = 'preparing to compact';
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

  isSessionProgressFresh(s) {
    if (!s) return false;
    if (s.state !== 'running' && s.state !== 'active') return false;
    const cached = this.sessionsState[s.id]?.progress;
    // Mid-turn entries carry a non-empty statusText ('Starting...', 'Turn N...',
    // 'Tool: bash', 'Waiting on LLM (12s)', etc); turn_end resets it to ''. So
    // statusText is the truthy "in flight" signal even during long silent tools
    // like a 20s sleep. No cache yet → optimistically pulse (running but no
    // events seen by us yet — usually a freshly-spawned interactive session).
    if (!cached) return true;
    // The server defaults status_text to 'Starting...' for sessions with zero
    // events; gate on lastEventTime so never-started sessions don't pulse.
    return !!cached.statusText && !!cached.lastEventTime;
  },

  _matchesFilters(s) {
    if (this.sessionFilter) {
      const text = [s.title, s.label, s.id, s.conversation_id, s.source, s.state,
                    s.metadata?.topic, s.metadata?.notes].filter(Boolean).join(' ');
      if (!text.toLowerCase().includes(this.sessionFilter.toLowerCase())) return false;
    }
    return true;
  },

  // The sidebar list only holds the ~100 most-recent sessions; typing a filter
  // also asks the server to search the FULL set (title/topic/metadata,
  // superseded bodies resolved to live heads) and merges any rows the client
  // hasn't loaded, so search never false-negatives on an older session.
  async searchSessionsServer() {
    const q = (this.sessionFilter || '').trim();
    if (!q) return;
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    try {
      const data = await get(`/api/agents/${agent}/sessions?q=${encodeURIComponent(q)}`);
      if ((this.sessionFilter || '').trim() !== q) return;
      const known = new Set(this.allSessions.map(x => x.id));
      const extra = (data.sessions || []).filter(x => !known.has(x.id));
      if (extra.length) this.allSessions.push(...extra);
    } catch { /* best-effort; local filtering still applies */ }
  },

  async togglePin(s) {
    if (!s || !s.id) return;
    const willPin = !s.pinned;
    s.pinned = willPin;  // optimistic
    try {
      await post(`/api/sessions/${s.id}/${willPin ? 'pin' : 'unpin'}`, {});
      await this.loadSessions();
    } catch (e) {
      s.pinned = !willPin;  // revert
      this.messages.push({ type: 'error', text: `Pin failed: ${e.message}` });
    }
  },

  async togglePrimary(s) {
    if (!s || !s.id) return;
    const wasPrimary = !!s.is_primary;
    s.is_primary = !wasPrimary;  // optimistic
    try {
      if (wasPrimary) {
        const agent = s.agent || '';
        const userId = s.user_id || s.metadata?.user_id || 'web-anonymous';
        await post(`/api/sessions/clear-primary?agent=${encodeURIComponent(agent)}&user_id=${encodeURIComponent(userId)}`, {});
      } else {
        await post(`/api/sessions/${s.id}/set-primary`, {});
      }
      await this.loadSessions();
    } catch (e) {
      s.is_primary = wasPrimary;  // revert
      this.messages.push({ type: 'error', text: `Primary toggle failed: ${e.message}` });
    }
  },

  onPinDragStart(s, event) {
    this.draggingPinId = s.id;
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move';
      event.dataTransfer.setData('text/plain', s.id);
    }
  },

  onPinDragOver(s, event) {
    if (!this.draggingPinId || this.draggingPinId === s.id) return;
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
    this.dragOverPinId = s.id;
  },

  onPinDragLeave(s) {
    if (this.dragOverPinId === s.id) this.dragOverPinId = null;
  },

  async onPinDrop(target) {
    const sourceId = this.draggingPinId;
    this.dragOverPinId = null;
    this.draggingPinId = null;
    if (!sourceId || !target || sourceId === target.id) return;
    const ordered = this.groupedSessions.pinned.map(s => s.id);
    const from = ordered.indexOf(sourceId);
    const to = ordered.indexOf(target.id);
    if (from < 0 || to < 0) return;
    const [moved] = ordered.splice(from, 1);
    ordered.splice(to, 0, moved);
    // Optimistic local reorder
    for (let i = 0; i < ordered.length; i++) {
      const s = this.allSessions.find(x => x.id === ordered[i]);
      if (s) s.pin_position = i;
    }
    try {
      await post('/api/sessions/pinned/reorder', { ids: ordered });
    } catch (e) {
      this.messages.push({ type: 'error', text: `Reorder failed: ${e.message}` });
      await this.loadSessions();
    }
  },

  onPinDragEnd() {
    this.draggingPinId = null;
    this.dragOverPinId = null;
  },
};
