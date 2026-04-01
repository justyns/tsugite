import { get, post } from '../api.js';
import { escapeHtml, renderMarkdown, scrollToBottom, formatDate, stateBadgeClass } from '../utils.js';
import { sessionsMixin } from './conversation/sessions.js';
import { historyMixin } from './conversation/history.js';
import { attachmentsMixin } from './conversation/attachments.js';
import { streamingMixin } from './conversation/streaming.js';
import { inputMixin } from './conversation/input.js';

export default () => ({
  ...sessionsMixin,
  ...historyMixin,
  ...attachmentsMixin,
  ...streamingMixin,
  ...inputMixin,

  // Data owned by orchestrator
  messages: [],
  loading: true,
  compacting: false,
  statusInfo: {},
  showAttachments: false,
  attachments: [],
  showSkills: false,
  loadedSkills: [],

  // Getters must stay here — spread loses get descriptors
  get userId() {
    return this.$store.app.userId;
  },

  get filteredCommands() {
    const text = this.messageText;
    if (!text.startsWith('/')) return [];
    const query = text.slice(1).split(/\s/)[0].toLowerCase();
    if (text.includes(' ')) return [];
    return this.availableCommands.filter(c => c.name.startsWith(query));
  },

  get groupedSessions() {
    const userId = this.userId;
    const filter = this.sessionFilter.toLowerCase();
    const interactive = [];
    const scheduled = [];
    const background = [];

    for (const s of this.allSessions) {
      if (filter) {
        const text = (s.title || '') + (s.label || '') + (s.id || '') + (s.conversation_id || '') + (s.source || '') + (s.state || '');
        if (!text.toLowerCase().includes(filter)) continue;
      }
      if (this._isMyInteractive(s)) {
        interactive.push(s);
      } else if (s.source === 'schedule' || (s.id && s.id.startsWith('sched_'))) {
        scheduled.push(s);
      } else {
        background.push(s);
      }
    }

    const byDate = (a, b) => (b.last_active || b.created_at || '').localeCompare(a.last_active || a.created_at || '');
    interactive.sort(byDate);
    scheduled.sort(byDate);
    background.sort(byDate);

    const filteredScheduled = this.showCompletedScheduled
      ? scheduled
      : scheduled.filter(s => s.state === 'active' || s.state === 'running');

    return {
      interactive,
      scheduled: filteredScheduled,
      scheduledTotal: scheduled.length,
      scheduledHidden: scheduled.length - filteredScheduled.length,
      background,
    };
  },

  get statusParts() {
    const info = this.statusInfo;
    const parts = [];
    if (info.model) parts.push({ label: 'Model', value: info.model });
    if (info.tokens != null && info.context_limit) {
      const tk = (info.tokens / 1000).toFixed(1);
      const lk = (info.context_limit / 1000).toFixed(0);
      const pct = ((info.tokens / info.context_limit) * 100).toFixed(0);
      parts.push({ label: 'Context', value: `${tk}k / ${lk}k (${pct}%)` });
    }
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
      if (ev.type === 'history_update' && ev.agent === this.$store.app.selectedAgent) {
        if (this.isActiveSession) this._debouncedLoadHistory();
      }
      if (ev.type === 'session_update') {
        if (ev.action === 'titled' && ev.title) {
          const s = this.allSessions.find(x => x.id === ev.id);
          if (s) { s.title = ev.title; return; }
        }
        this._debouncedLoadSessions();
      }
      if (ev.type === 'compaction_started' && ev.agent === this.$store.app.selectedAgent) {
        this.compacting = true;
      }
      if (ev.type === 'compaction_finished' && ev.agent === this.$store.app.selectedAgent) {
        this.compacting = false;
        if (this.isActiveSession) this.reload();
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
  },

  destroy() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    if (this._scrollTimer) clearTimeout(this._scrollTimer);
    if (this._historyDebounceTimer) clearTimeout(this._historyDebounceTimer);
    if (this._activeReader) this._activeReader.cancel().catch(() => {});
    this.pendingFiles.forEach(f => { if (f.previewUrl) URL.revokeObjectURL(f.previewUrl); });
  },

  async reload() {
    this.messages = [];
    this.resetHistory();
    this.statusInfo = {};
    this.loadedSkills = [];
    this.selectedSessionMeta = null;
    await this.loadSessions();

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
      this.autoSelectInteractive();
    }
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
    try {
      const data = await get(`/api/agents/${agent}/status?user_id=${encodeURIComponent(this.userId)}`);
      this.statusInfo = data;
      if (data.compacting !== undefined) this.compacting = data.compacting;
      if (data.busy && data.pending_message && !this.sending &&
          !this.messages.some(m => m.type === 'user' && m.text === data.pending_message)) {
        this.messages.push({ type: 'user', text: data.pending_message });
        this.messages.push({ type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 });
        this.scrollMessages();
      }
    } catch { /* ignore */ }
  },

  updateStatusFromEvent(event) {
    this.statusInfo = {
      model: event.model,
      tokens: event.tokens,
      context_limit: event.context_limit,
      threshold: event.threshold,
      message_count: event.message_count,
      attachments: event.attachments,
    };
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

  async compactSession(retryMsg = null) {
    const agent = this.$store.app.selectedAgent;
    if (!agent || this.sending || this.compacting) return;
    this.compacting = true;
    try {
      await post(`/api/agents/${agent}/compact`, { user_id: this.userId });
      await this.loadHistory();
      if (retryMsg) {
        this.messageText = retryMsg;
        await this.sendMessage();
      }
    } catch (e) {
      this.messages.push({ type: 'error', text: `Compact failed: ${e.message}` });
    } finally {
      this.compacting = false;
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
      await post(`/api/agents/${agent}/respond`, { response, user_id: this.userId });
    } catch (e) {
      this.messages.push({ type: 'error', text: `Failed to submit answer: ${e.message}` });
    }
  },

  // Rendering helpers
  renderHtml(text) { return renderMarkdown(text); },
  escape(s) { return escapeHtml(s); },
  formatDate(iso) { return formatDate(iso) || '—'; },
  stateBadge(state) { return stateBadgeClass(state); },

  scrollMessages() {
    this.$nextTick(() => {
      const el = this.$refs.messages;
      if (el) scrollToBottom(el);
    });
  },
});
