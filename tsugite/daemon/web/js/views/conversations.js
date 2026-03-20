import { get, post, streamPost, uploadFiles } from '../api.js';
import { escapeHtml, renderMarkdown, scrollToBottom, formatDate, formatFileSize, stateBadgeClass, contentBlockHtml } from '../utils.js';

export default () => ({
  sidebarOpen: false,
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
  allSessions: [],
  selectedSessionId: null,
  isActiveSession: true,
  loading: true,
  selectedSessionMeta: null,
  turns: [],
  compactionSummary: null,
  compactedFrom: null,
  _debounceTimer: null,

  init() {
    this._mobileQuery = window.matchMedia('(max-width: 640px)');
    const maybeReload = () => {
      if (this.$store.app.view === 'conversations' && this.$store.app.selectedAgent) this.reload();
    };
    this.$watch('$store.app.selectedAgent', maybeReload);
    this.$watch('$store.app.view', (view) => {
      if (view === 'conversations' && this.$store.app.selectedAgent) this.reload();
    });
    this.$watch('$store.app.lastEvent', (ev) => {
      if (!ev) return;
      if (ev.type === 'history_update' && ev.agent === this.$store.app.selectedAgent) {
        if (this.isActiveSession) this.loadHistory();
      }
      if (ev.type === 'session_update') {
        this._debouncedLoadSessions();
      }
      if (ev.type === 'compaction_started' && ev.agent === this.$store.app.selectedAgent) {
        this.compacting = true;
      }
      if (ev.type === 'compaction_finished' && ev.agent === this.$store.app.selectedAgent) {
        this.compacting = false;
        if (this.isActiveSession) this.loadHistory();
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

  get sortedSessions() {
    const sessions = [...this.allSessions];
    const userId = this.userId;
    sessions.sort((a, b) => {
      const aInteractive = a.source === 'interactive' && (a.user_id === userId || a.conversation_id === userId);
      const bInteractive = b.source === 'interactive' && (b.user_id === userId || b.conversation_id === userId);
      if (aInteractive && !bInteractive) return -1;
      if (!aInteractive && bInteractive) return 1;
      const aDate = a.last_active || a.created_at || '';
      const bDate = b.last_active || b.created_at || '';
      return bDate.localeCompare(aDate);
    });
    return sessions;
  },

  autoSelectInteractive() {
    const interactive = this.sortedSessions.find(
      s => s.source === 'interactive' && (s.user_id === this.userId || s.conversation_id === this.userId)
    );
    if (interactive) {
      this.selectSession(interactive);
    } else if (this.sortedSessions.length > 0) {
      this.selectSession(this.sortedSessions[0]);
    }
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
      let histUrl = `/api/agents/${agent}/history?user_id=${encodeURIComponent(this.userId)}&limit=100`;
      if (this.selectedSessionId) histUrl += `&session_id=${encodeURIComponent(this.selectedSessionId)}`;
      const data = await get(histUrl);
      this.compactionSummary = data.compaction_summary || null;
      this.compactedFrom = data.compacted_from || null;
      if (!data.turns || data.turns.length === 0) return;
      for (const turn of data.turns) {
        if (turn.type === 'compaction') {
          this._allHistoryMessages.push({ type: 'compaction', summary: turn.summary || null });
          continue;
        }
        if (turn.type === 'hook_execution') {
          this._allHistoryMessages.push({
            type: 'hook',
            phase: turn.phase,
            name: turn.name,
            command: turn.command,
            exit_code: turn.exit_code,
            stdout: turn.stdout,
            stderr: turn.stderr,
            duration_ms: turn.duration_ms,
            timestamp: turn.timestamp,
          });
          continue;
        }
        if (turn.user) this._allHistoryMessages.push({ type: 'user', text: turn.user });
        if (turn.content_blocks && Object.keys(turn.content_blocks).length) {
          const steps = Object.entries(turn.content_blocks).map(([name, content]) => ({
            html: contentBlockHtml(name, content)
          }));
          this._allHistoryMessages.push({ type: 'progress-done', steps, turnCount: 0, toolCount: 0 });
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
    return s?.state === 'running';
  },

  canRestart(s) {
    return s?.state === 'failed' || s?.state === 'cancelled';
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

  sessionLabel(s) {
    if (s.source === 'interactive' && (s.user_id === this.userId || s.conversation_id === this.userId)) {
      return 'Interactive (you)';
    }
    return s.label || s.conversation_id || s.id || 'unknown';
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
        const truncated = content.length > 500 ? content.slice(0, 500) + '...' : content;
        items.push({ type: 'tool_result', name: 'result', content: truncated });
      } else if (msg.role === 'tool') {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
        items.push({ type: 'tool_result', name: msg.name || '', content: content.length > 500 ? content.slice(0, 500) + '...' : content });
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

    this.sending = true;
    this.messageText = '';

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
        prog.steps.push({ html: `<details><summary>thought</summary><pre><code>${escapeHtml(event.content)}</code></pre></details>` });
      }
    } else if (event.type === 'error') {
      prog.steps.push({ html: `<span class="err">${escapeHtml(event.error)}</span>` });
      prog.errorText = event.error;
    } else if (event.type === 'hook_status') {
      prog.statusText = event.message;
    } else if (event.type === 'init') {
      prog.statusText = `Agent: ${event.agent}`;
      if (event.model) this.statusInfo = { ...this.statusInfo, model: event.model };
    } else if (event.type === 'content_block') {
      prog.steps.push({ html: contentBlockHtml(event.name, event.content || '') });
    } else if (event.type === 'code') {
      prog.steps.push({ html: `<details><summary><code>code</code></summary><pre><code>${escapeHtml(event.content || '')}</code></pre></details>` });
    } else if (event.type === 'tool_result') {
      const isCodeResult = event.tool === 'unknown';
      if (!isCodeResult) prog.toolCount++;
      const cls = event.success ? 'ok' : 'err';
      const status = event.success ? 'ok' : 'err';
      const label = isCodeResult ? 'output' : event.tool;
      const output = event.output || event.error || '';
      if (output) {
        prog.steps.push({ html: `<details><summary><code>${escapeHtml(label)}</code> <span class="${cls}">${status}</span></summary><pre><code>${escapeHtml(output)}</code></pre></details>` });
      } else {
        prog.steps.push({ html: `<code>${escapeHtml(label)}</code> <span class="${cls}">${status}</span>` });
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

  onInputKeydown(e) {
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
});
