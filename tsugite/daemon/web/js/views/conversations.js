import { get, post, patch, streamPost, uploadFiles } from '../api.js';
import { escapeHtml, renderMarkdown, scrollToBottom, formatDate, formatFileSize, stateBadgeClass, contentBlockHtml, truncate } from '../utils.js';

export default () => ({
  sidebarOpen: false,
  showCompletedScheduled: false,
  sessionFilter: '',
  messages: [],
  _allHistoryMessages: [],
  _historyLoaded: 0,
  HISTORY_PAGE_SIZE: 20,
  hasMoreHistory: false,
  sending: false,
  _activeReader: null,
  compacting: false,
  statusInfo: {},
  messageText: '',
  showAttachments: false,
  attachments: [],
  showSkills: false,
  loadedSkills: [],
  pendingFiles: [],
  isDragging: false,
  expandedInput: false,
  allSessions: [],
  selectedSessionId: null,
  isActiveSession: true,
  loading: true,
  selectedSessionMeta: null,
  turns: [],
  compactionSummary: null,
  compactedFrom: null,
  _debounceTimer: null,
  _scrollTimer: null,
  availableCommands: [],
  showCommandSuggestions: false,
  commandSelectedIndex: 0,

  get filteredCommands() {
    const text = this.messageText;
    if (!text.startsWith('/')) return [];
    const query = text.slice(1).split(/\s/)[0].toLowerCase();
    if (text.includes(' ')) return [];
    return this.availableCommands.filter(c => c.name.startsWith(query));
  },

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
    // Eagerly load if agent is already selected (from localStorage)
    if (this.$store.app.view === 'conversations' && this.$store.app.selectedAgent) {
      this.reload();
    }
    this.$watch('$store.app.lastEvent', (ev) => {
      if (!ev) return;
      if (ev.type === 'history_update' && ev.agent === this.$store.app.selectedAgent) {
        if (this.isActiveSession) this.loadHistory();
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
  },

  _debouncedLoadSessions() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    this._debounceTimer = setTimeout(() => this.loadSessions(), 200);
  },

  async reload() {
    this.messages = [];
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    this.hasMoreHistory = false;
    this.statusInfo = {};
    this.loadedSkills = [];
    this.turns = [];
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

  get userId() {
    return this.$store.app.userId;
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
      const isMyInteractive = s.source === 'interactive' && (s.user_id === userId || s.conversation_id === userId);
      if (isMyInteractive) {
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
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    this.hasMoreHistory = false;
    this.turns = [];
    this.compactionSummary = null;
    this.compactedFrom = null;

    const isInteractive = session.source === 'interactive' &&
      (session.user_id === this.userId || session.conversation_id === this.userId);
    this.isActiveSession = isInteractive;

    if (isInteractive) {
      this.loadHistory();
      this.loadStatus();
    } else {
      this.loadDetailHistory();
    }
  },

  async loadHistory() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    try {
      let histUrl = `/api/agents/${agent}/history?user_id=${encodeURIComponent(this.userId)}&limit=100&detail=true`;
      if (this.selectedSessionId) histUrl += `&session_id=${encodeURIComponent(this.selectedSessionId)}`;
      const data = await get(histUrl);
      this.compactionSummary = data.compaction_summary || null;
      this.compactedFrom = data.compacted_from || null;
      if (!data.turns || data.turns.length === 0) return;
      let pendingHooks = [];
      for (const turn of data.turns) {
        if (turn.type === 'compaction') {
          this._allHistoryMessages.push({ type: 'compaction', summary: turn.summary || null });
          continue;
        }
        if (turn.type === 'hook_execution') {
          pendingHooks.push(turn);
          continue;
        }
        if (turn.user) this._allHistoryMessages.push({ type: 'user', text: turn.user });

        const steps = [];
        for (const h of pendingHooks) {
          const summary = this._hookStepHtml(h.name, h.phase, h.exit_code);
          const output = [h.stdout, h.stderr].filter(Boolean).join('\n');
          if (output) {
            steps.push({ hasDetails: true, summary, content: escapeHtml(output), open: false });
          } else {
            steps.push({ html: summary });
          }
        }
        pendingHooks = [];
        if (turn.messages) {
          for (const item of this.extractMessages(turn)) {
            if (item.type === 'tool_call') {
              steps.push({ hasDetails: true, summary: `<code>${escapeHtml(item.name)}</code>`, content: escapeHtml(truncate(item.args)), open: false });
            } else if (item.type === 'tool_result') {
              steps.push({ hasDetails: true, summary: `<code>${escapeHtml(item.name || 'result')}</code>`, content: escapeHtml(item.content), open: false });
            }
          }
        }
        if (turn.content_blocks && Object.keys(turn.content_blocks).length) {
          for (const [name, content] of Object.entries(turn.content_blocks)) {
            steps.push({ html: contentBlockHtml(name, content) });
          }
        }
        if (steps.length > 0) {
          this._allHistoryMessages.push({ type: 'progress-done', steps, turnCount: turn.turn_count || 1, toolCount: turn.tools_used?.length || 0 });
        }
        if (turn.assistant) {
          this._allHistoryMessages.push({ type: 'agent', text: turn.assistant });
        }
      }
      this._showRecentHistory();
    } catch { /* ignore */ }
    this.scrollMessages();
  },

  async loadDetailHistory() {
    const agent = this.$store.app.selectedAgent;
    if (!agent || !this.selectedSessionId) return;
    try {
      let url = `/api/agents/${agent}/history?detail=true&session_id=${encodeURIComponent(this.selectedSessionId)}`;
      const data = await get(url);
      this.turns = data.turns || [];
    } catch { /* ignore */ }
  },

  _showRecentHistory() {
    const all = this._allHistoryMessages;
    const page = this.HISTORY_PAGE_SIZE;
    const startIdx = Math.max(0, all.length - page);
    const slice = all.slice(startIdx);
    this._historyLoaded = all.length - startIdx;
    this.hasMoreHistory = startIdx > 0;
    this.messages = [];
    if (this.hasMoreHistory) {
      this.messages.push({ type: 'separator', text: `${startIdx} earlier messages — load more ↑` });
    }
    this.messages.push(...slice);
    if (slice.length > 0) {
      this.messages.push({ type: 'separator', text: `${this._historyLoaded} of ${all.length} messages loaded` });
    }
  },

  loadMoreHistory() {
    if (!this.hasMoreHistory) return;
    const all = this._allHistoryMessages;
    const page = this.HISTORY_PAGE_SIZE;
    const currentlyShown = this._historyLoaded;
    const newShown = Math.min(all.length, currentlyShown + page);
    const startIdx = all.length - newShown;
    const newSlice = all.slice(startIdx, all.length - currentlyShown);
    this._historyLoaded = newShown;
    this.hasMoreHistory = startIdx > 0;

    if (this.messages.length > 0 && this.messages[0].type === 'separator') {
      this.messages.shift();
    }
    if (this.hasMoreHistory) {
      this.messages.unshift(...newSlice, { type: 'separator', text: `${startIdx} earlier messages — load more ↑` });
    } else {
      this.messages.unshift(...newSlice);
    }
    const lastSep = this.messages.findLastIndex(m => m.type === 'separator');
    if (lastSep >= 0) {
      this.messages[lastSep].text = `${this._historyLoaded} of ${all.length} messages loaded`;
    }
  },

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

  async compactSession() {
    const agent = this.$store.app.selectedAgent;
    if (!agent || this.sending || this.compacting) return;
    this.compacting = true;
    try {
      await post(`/api/agents/${agent}/compact`, { user_id: this.userId });
      await this.loadHistory();
    } catch (e) {
      this.messages.push({ type: 'error', text: `Compact failed: ${e.message}` });
    } finally {
      this.compacting = false;
    }
  },

  async compactAndRetry(msg) {
    const agent = this.$store.app.selectedAgent;
    if (!agent || this.sending || this.compacting) return;
    this.compacting = true;
    try {
      await post(`/api/agents/${agent}/compact`, { user_id: this.userId });
      await this.loadHistory();
      if (msg) await this.sendMessage(msg);
    } catch (e) {
      this.messages.push({ type: 'error', text: `Compact & retry failed: ${e.message}` });
    } finally {
      this.compacting = false;
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

  _scrollThrottled() {
    if (this._scrollTimer) return;
    this._scrollTimer = setTimeout(() => {
      this._scrollTimer = null;
      this.scrollMessages();
    }, 150);
  },

  _hookStepHtml(name, phase, exitCode) {
    const cls = exitCode === 0 ? 'ok' : 'err';
    const label = `${name || 'hook'} (${phase})`;
    return `<code>${escapeHtml(label)}</code> <span class="hook-exit ${cls}">exit ${exitCode}</span>`;
  },

  _pushDetailStep(prog, summary, contentHtml) {
    const follow = this.$store.app.autoFollow;
    if (follow && prog._lastOpenIdx != null) {
      const prev = prog.steps[prog._lastOpenIdx];
      if (prev?.hasDetails) prev.open = false;
    }
    const idx = prog.steps.length;
    prog.steps.push({ hasDetails: true, summary, content: contentHtml, open: follow });
    if (follow) {
      prog._lastOpenIdx = idx;
      this._scrollThrottled();
    }
  },

  editingSessionId: null,
  editingTitle: '',

  sessionLabel(s) {
    if (s.title) return s.title;
    if (s.source === 'interactive' && (s.user_id === this.userId || s.conversation_id === this.userId)) {
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

  extractMessages(turn) {
    if (!turn.messages) return [];
    const items = [];
    for (const msg of turn.messages) {
      if (msg.role === 'assistant') {
        const codeMatch = msg.content?.match(/```(?:python)?\n([\s\S]*?)```/);
        if (codeMatch) {
          items.push({ type: 'tool_call', name: 'code', args: codeMatch[1] });
        }
        if (msg.tool_calls) {
          for (const tc of msg.tool_calls) {
            const fn = tc.function || {};
            items.push({ type: 'tool_call', name: fn.name || 'unknown', args: fn.arguments || '{}' });
          }
        }
      } else if (msg.role === 'user' && msg.content?.includes('<tsugite_execution_result>')) {
        const content = msg.content.replace(/<\/?tsugite_execution_result>/g, '').trim();
        items.push({ type: 'tool_result', name: 'result', content: truncate(content) });
      } else if (msg.role === 'tool') {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
        items.push({ type: 'tool_result', name: msg.name || '', content: truncate(content) });
      }
    }
    return items;
  },

  addFiles(fileList) {
    for (const file of fileList) {
      const entry = { file, name: file.name, size: file.size, type: file.type, previewUrl: null };
      if (file.type.startsWith('image/')) entry.previewUrl = URL.createObjectURL(file);
      this.pendingFiles.push(entry);
    }
  },

  removeFile(index) {
    const entry = this.pendingFiles[index];
    if (entry?.previewUrl) URL.revokeObjectURL(entry.previewUrl);
    this.pendingFiles.splice(index, 1);
  },

  openFilePicker() {
    this.$refs.fileInput?.click();
  },

  onFileInputChange(e) {
    if (e.target.files?.length) this.addFiles(e.target.files);
    e.target.value = '';
  },

  onDragEnter(e) { this.isDragging = true; },
  onDragLeave(e) {
    if (!e.currentTarget.contains(e.relatedTarget)) this.isDragging = false;
  },
  onDrop(e) {
    this.isDragging = false;
    if (e.dataTransfer?.files?.length) this.addFiles(e.dataTransfer.files);
  },

  async sendMessage() {
    const msg = this.messageText.trim();
    const agent = this.$store.app.selectedAgent;
    if ((!msg && !this.pendingFiles.length) || !agent || this.sending) return;

    // Intercept slash commands
    const parsed = this._parseCommand(msg);
    if (parsed && !this.pendingFiles.length) {
      this.sending = true;
      this.messageText = '';
      this._resetInputHeight();
      this.showCommandSuggestions = false;
      this.messages.push({ type: 'user', text: msg });
      this.scrollMessages();
      try {
        const result = await this._runCommand(parsed.command, parsed.args);
        this.messages.push({ type: 'agent', text: result });
      } catch (e) {
        this.messages.push({ type: 'error', text: `Command error: ${e.message}` });
      } finally {
        this.sending = false;
        this.scrollMessages();
      }
      return;
    }

    this.sending = true;
    this.messageText = '';
    this._resetInputHeight();

    let uploadedFiles = [];
    const fileNames = this.pendingFiles.map(f => f.name);
    const workspaceFiles = this.pendingFiles.filter(f => f.fromWorkspace);
    const normalFiles = this.pendingFiles.filter(f => !f.fromWorkspace);
    if (normalFiles.length) {
      try {
        const data = await uploadFiles(`/api/agents/${agent}/upload`, normalFiles.map(f => f.file));
        uploadedFiles = data.files || [];
      } catch (e) {
        this.messages.push({ type: 'error', text: `Upload failed: ${e.message}` });
        this.sending = false;
        return;
      }
    }
    for (const wf of workspaceFiles) {
      uploadedFiles.push(wf.uploadInfo);
    }
    if (this.pendingFiles.length) {
      this.pendingFiles.forEach(f => { if (f.previewUrl) URL.revokeObjectURL(f.previewUrl); });
      this.pendingFiles = [];
    }

    const displayMsg = fileNames.length ? `${msg || ''}\n📎 ${fileNames.join(', ')}`.trim() : msg;
    this.messages.push({ type: 'user', text: displayMsg });

    this.scrollMessages();

    const progressIdx = this.messages.length;
    this.messages.push({ type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 });

    try {
      const chatBody = { message: msg, user_id: this.userId };
      if (this.selectedSessionId) chatBody.session_id = this.selectedSessionId;
      if (uploadedFiles.length) chatBody.uploaded_files = uploadedFiles;
      const resp = await streamPost(`/api/agents/${agent}/chat`, chatBody);
      const reader = resp.body.getReader();
      this._activeReader = reader;
      const decoder = new TextDecoder();
      let buffer = '';
      let gotResult = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let event;
          try { event = JSON.parse(line.slice(6)); } catch { continue; }

          if (event.type === 'done') {
            break;
          } else if (event.type === 'cancelled') {
            this.messages.push({ type: 'info', text: 'Generation stopped.' });
            break;
          } else if (event.type === 'compacting') {
            this.compacting = true;
          } else if (event.type === 'compacted') {
            this.compacting = false;
          } else if (event.type === 'skill_loaded') {
            if (!this.loadedSkills.some(s => s.name === event.name)) {
              this.loadedSkills.push({ name: event.name, description: event.description || '' });
            }
          } else if (event.type === 'skill_unloaded') {
            this.loadedSkills = this.loadedSkills.filter(s => s.name !== event.name);
          } else if (event.type === 'ask_user') {
            this.messages.push({
              type: 'ask_user',
              question: event.question,
              questionType: event.question_type || 'text',
              options: event.options || [],
              answered: false,
              answer: '',
              inputValue: '',
            });
            this.scrollMessages();
          } else if (event.type === 'reaction') {
            for (let i = this.messages.length - 1; i >= 0; i--) {
              if (this.messages[i].type === 'user') {
                if (!this.messages[i].reactions) this.messages[i].reactions = [];
                this.messages[i].reactions.push(event.emoji);
                break;
              }
            }
          } else if (event.type === 'final_result') {
            gotResult = true;
            this.messages.push({ type: 'agent', text: event.result });
          } else if (event.type === 'session_info') {
            this.updateStatusFromEvent(event);
          } else if (event.type === 'error') {
            this._handleProgressEvent(progressIdx, event);
          } else {
            this._handleProgressEvent(progressIdx, event);
          }
        }
      }

      reader.cancel().catch(() => {});

      const prog = this.messages[progressIdx];
      if (prog && prog.type === 'progress') {
        if (prog.steps.length > 0) {
          prog.type = 'progress-done';
          if (!gotResult && prog.errorText) {
            prog.failed = true;
            prog.lastMessage = msg;
          }
        } else {
          this.messages.splice(progressIdx, 1);
        }
      }
    } catch (e) {
      if (this.messages[progressIdx]?.type === 'progress') {
        this.messages.splice(progressIdx, 1);
      }
      this.messages.push({ type: 'error', text: `Connection error: ${e.message}` });
    } finally {
      this._activeReader = null;
      this.sending = false;
      this.scrollMessages();
    }
  },

  _handleProgressEvent(idx, event) {
    const prog = this.messages[idx];
    if (!prog || prog.type !== 'progress') return;

    if (event.type === 'turn_start') {
      prog.turnCount++;
      prog.statusText = `Turn ${event.turn}...`;
    } else if (event.type === 'thought') {
      prog.statusText = 'Thinking...';
      if (event.content) {
        this._pushDetailStep(prog, 'thought', escapeHtml(event.content));
      }
    } else if (event.type === 'error') {
      prog.steps.push({ html: `<span class="err">${escapeHtml(event.error)}</span>` });
      prog.errorText = event.error;
    } else if (event.type === 'hook_status') {
      prog.statusText = event.message;
    } else if (event.type === 'hook_execution') {
      const summary = this._hookStepHtml(event.name, event.phase, event.exit_code);
      const output = [event.stdout, event.stderr].filter(Boolean).join('\n');
      if (output) {
        this._pushDetailStep(prog, summary, escapeHtml(output));
      } else {
        prog.steps.push({ html: summary });
      }
    } else if (event.type === 'init') {
      prog.statusText = `Agent: ${event.agent}`;
      if (event.model) this.statusInfo = { ...this.statusInfo, model: event.model };
    } else if (event.type === 'content_block') {
      prog.steps.push({ html: contentBlockHtml(event.name, event.content || '') });
    } else if (event.type === 'code') {
      this._pushDetailStep(prog, `<code>code</code>`, escapeHtml(event.content || ''));
    } else if (event.type === 'tool_result') {
      const isCodeResult = event.tool === 'unknown';
      if (!isCodeResult) prog.toolCount++;
      const cls = event.success ? 'ok' : 'err';
      const label = isCodeResult ? 'output' : event.tool;
      const output = event.output || event.error || '';
      if (output) {
        this._pushDetailStep(prog, `<code>${escapeHtml(label)}</code> <span class="${cls}">${cls}</span>`, escapeHtml(output));
      } else {
        prog.steps.push({ html: `<code>${escapeHtml(label)}</code> <span class="${cls}">${cls}</span>` });
      }
    } else if (event.type === 'file_read') {
      const readSize = formatFileSize(event.byte_count);
      prog.steps.push({ html: `<code>${escapeHtml(event.path)}</code> read (${readSize})` });
    } else if (event.type === 'file_write') {
      const writeSize = formatFileSize(event.byte_count);
      prog.steps.push({ html: `<code>${escapeHtml(event.path)}</code> written (${writeSize})` });
    } else if (event.type === 'warning') {
      prog.steps.push({ html: `<span class="err">${escapeHtml(event.message)}</span>` });
    } else if (event.type === 'info') {
      prog.steps.push({ html: escapeHtml(event.message) });
      this.messages.push({ type: 'info', text: event.message });
      this.scrollMessages();
    }
  },

  progressSummaryText(msg) {
    const parts = [];
    if (msg.turnCount) parts.push(`${msg.turnCount} turn${msg.turnCount > 1 ? 's' : ''}`);
    if (msg.toolCount) parts.push(`${msg.toolCount} tool${msg.toolCount > 1 ? 's' : ''}`);
    return parts.join(', ') || 'trace';
  },

  retryMessage(msg) {
    this.messageText = msg;
    this.sendMessage();
  },

  continueAfterError(lastMessage, errorText) {
    this.messageText = `The previous request failed with: ${errorText}\n\nPlease continue from where you left off. The original request was: ${lastMessage}`;
    this.sendMessage();
  },

  async cancelChat() {
    const agent = this.$store.app.selectedAgent;
    if (!agent || !this.sending) return;
    try {
      await post(`/api/agents/${agent}/chat/cancel`, { user_id: this.userId });
    } catch (e) { /* best effort */ }
    if (this._activeReader) {
      this._activeReader.cancel().catch(() => {});
    }
  },

  async _loadCommands() {
    try {
      const data = await get('/api/commands');
      this.availableCommands = data.commands || [];
    } catch {
      this.availableCommands = [];
    }
  },

  onInputChange() {
    this.showCommandSuggestions = this.messageText.startsWith('/') && !this.messageText.includes(' ') && this.filteredCommands.length > 0;
    this.commandSelectedIndex = 0;
  },

  selectCommand(cmd) {
    this.messageText = `/${cmd.name} `;
    this.showCommandSuggestions = false;
    this.$nextTick(() => {
      const input = document.getElementById('message-input');
      if (input) input.focus();
    });
  },

  _parseCommand(text) {
    const match = text.match(/^\/(\S+)\s*(.*)/s);
    if (!match) return null;
    const name = match[1];
    const rest = match[2].trim();
    const cmd = this.availableCommands.find(c => c.name === name);
    if (!cmd) return null;
    return { command: cmd, args: rest };
  },

  async _runCommand(cmd, argsText) {
    const agent = this.$store.app.selectedAgent;
    const kwargs = {};
    const hasUserId = cmd.params.some(p => p.name === 'user_id');
    const visibleParams = cmd.params.filter(p => p.name !== 'user_id');
    const requiredVisible = visibleParams.filter(p => p.required);
    if (requiredVisible.length === 1 && argsText) {
      kwargs[requiredVisible[0].name] = argsText;
    } else if (requiredVisible.length > 1) {
      const parts = argsText.split(/\s+/);
      for (let i = 0; i < visibleParams.length && i < parts.length; i++) {
        kwargs[visibleParams[i].name] = parts[i];
      }
    } else if (visibleParams.length === 1 && !visibleParams[0].required && argsText) {
      kwargs[visibleParams[0].name] = argsText;
    }
    if (hasUserId) kwargs.user_id = this.userId;
    try {
      const data = await post(`/api/agents/${agent}/commands/${cmd.name}`, kwargs);
      return data.result || JSON.stringify(data);
    } catch (e) {
      return `Command failed: ${e.message}`;
    }
  },

  onInputKeydown(e) {
    // Command autocomplete navigation
    if (this.showCommandSuggestions) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.commandSelectedIndex = Math.min(this.commandSelectedIndex + 1, this.filteredCommands.length - 1);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.commandSelectedIndex = Math.max(this.commandSelectedIndex - 1, 0);
        return;
      }
      if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
        e.preventDefault();
        const selected = this.filteredCommands[this.commandSelectedIndex];
        if (selected) this.selectCommand(selected);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        this.showCommandSuggestions = false;
        return;
      }
    }

    if (e.key === 'Escape' && this.sending) {
      e.preventDefault();
      this.cancelChat();
      return;
    }
    if (e.key === 'Enter' && !e.shiftKey && !this._mobileQuery.matches) {
      e.preventDefault();
      this.sendMessage();
    }
  },

  backToSessions() {
    this.selectedSessionId = null;
    this.selectedSessionMeta = null;
    this.messages = [];
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    this.hasMoreHistory = false;
    this.turns = [];
    this.compactionSummary = null;
    this.compactedFrom = null;
    this.isActiveSession = true;
  },

  autoResize(e) {
    const maxH = parseInt(getComputedStyle(e.target).maxHeight) || 150;
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, maxH) + 'px';
  },

  _resetInputHeight() {
    this.expandedInput = false;
    this.$nextTick(() => {
      const ta = document.getElementById('message-input');
      if (ta) ta.style.height = 'auto';
    });
  },
});
