import { get, post, streamPost, uploadFiles } from '../api.js';
import { escapeHtml, renderMarkdown, scrollToBottom } from '../utils.js';

export default () => ({
  messages: [],
  _allHistoryMessages: [],
  _historyLoaded: 0,
  HISTORY_PAGE_SIZE: 20,
  hasMoreHistory: false,
  sessions: [],
  sending: false,
  compacting: false,
  statusInfo: {},
  messageText: '',
  showAttachments: false,
  attachments: [],
  showSkills: false,
  loadedSkills: [],
  pendingFiles: [],
  isDragging: false,

  init() {
    const maybeReload = () => {
      if (this.$store.app.view === 'chat' && this.$store.app.selectedAgent) this.reload();
    };
    this.$watch('$store.app.selectedAgent', maybeReload);
    this.$watch('$store.app.view', (view) => {
      if (view === 'chat' && this.$store.app.selectedAgent && this.messages.length === 0) this.reload();
    });
    this.$watch('$store.app.lastEvent', (ev) => {
      if (!ev) return;
      if (ev.type === 'history_update' && ev.agent === this.$store.app.selectedAgent) {
        this.reload();
      }
      if (ev.type === 'reconnect' && this.$store.app.selectedAgent) {
        this.reload();
      }
    });
  },

  async reload() {
    this.messages = [];
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    this.hasMoreHistory = false;
    this.statusInfo = {};
    this.loadedSkills = [];
    await this.loadSessions();
    await this.loadHistory();
    await this.loadStatus();
  },

  get userId() {
    return this.$store.app.userId;
  },

  async loadSessions() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    try {
      const data = await get(`/api/agents/${agent}/sessions`);
      this.sessions = data.sessions || [];
    } catch { this.sessions = []; }
  },

  get sessionOptions() {
    const defaultId = this.userId;
    const opts = this.sessions.map(s => ({ value: s.user_id, label: s.label }));
    if (!opts.some(o => o.value === defaultId)) {
      opts.unshift({ value: defaultId, label: `Web: ${defaultId} (new)` });
    }
    return opts;
  },

  async loadHistory() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    try {
      const data = await get(`/api/agents/${agent}/history?user_id=${encodeURIComponent(this.userId)}&limit=100`);
      if (!data.turns || data.turns.length === 0) return;
      for (const turn of data.turns) {
        if (turn.type === 'compaction') {
          this._allHistoryMessages.push({ type: 'compaction', summary: turn.summary || null });
          continue;
        }
        if (turn.user) this._allHistoryMessages.push({ type: 'user', text: turn.user });
        if (turn.assistant) this._allHistoryMessages.push({ type: 'agent', text: turn.assistant });
      }
      this._showRecentHistory();
    } catch { /* ignore */ }
    this.$nextTick(() => {
      const el = this.$refs.messages;
      if (el) scrollToBottom(el);
    });
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

    // Remove old "load more" separator at top
    if (this.messages.length > 0 && this.messages[0].type === 'separator') {
      this.messages.shift();
    }
    // Insert new messages at top + new separator if needed
    if (this.hasMoreHistory) {
      this.messages.unshift(...newSlice, { type: 'separator', text: `${startIdx} earlier messages — load more ↑` });
    } else {
      this.messages.unshift(...newSlice);
    }
    // Update bottom separator
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
      await this.reload();
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

  renderHtml(text) { return renderMarkdown(text); },
  escape(s) { return escapeHtml(s); },

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

    // Upload pending files
    let uploadedFiles = [];
    const fileNames = this.pendingFiles.map(f => f.name);
    if (this.pendingFiles.length) {
      try {
        const data = await uploadFiles(`/api/agents/${agent}/upload`, this.pendingFiles.map(f => f.file));
        uploadedFiles = data.files || [];
      } catch (e) {
        this.messages.push({ type: 'error', text: `Upload failed: ${e.message}` });
        this.sending = false;
        return;
      }
      this.pendingFiles.forEach(f => { if (f.previewUrl) URL.revokeObjectURL(f.previewUrl); });
      this.pendingFiles = [];
    }

    const displayMsg = fileNames.length ? `${msg || ''}\n📎 ${fileNames.join(', ')}`.trim() : msg;
    this.messages.push({ type: 'user', text: displayMsg });

    this.$nextTick(() => {
      const el = this.$refs.messages;
      if (el) scrollToBottom(el);
    });

    const progressIdx = this.messages.length;
    this.messages.push({ type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 });

    try {
      const chatBody = { message: msg, user_id: this.userId };
      if (uploadedFiles.length) chatBody.uploaded_files = uploadedFiles;
      const resp = await streamPost(`/api/agents/${agent}/chat`, chatBody);
      const reader = resp.body.getReader();
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
            reader.cancel().catch(() => {});
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
            this.$nextTick(() => {
              const el = this.$refs.messages;
              if (el) scrollToBottom(el);
            });
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

      // Finalize progress: collapse to summary or remove
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
      // Remove progress on connection error
      if (this.messages[progressIdx]?.type === 'progress') {
        this.messages.splice(progressIdx, 1);
      }
      this.messages.push({ type: 'error', text: `Connection error: ${e.message}` });
    } finally {
      this.sending = false;
      this.$nextTick(() => {
        const el = this.$refs.messages;
        if (el) scrollToBottom(el);
      });
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
      const readSize = this._formatFileSize(event.byte_count);
      prog.steps.push({ html: `<code>${escapeHtml(event.path)}</code> read (${readSize})` });
    } else if (event.type === 'file_write') {
      const writeSize = this._formatFileSize(event.byte_count);
      prog.steps.push({ html: `<code>${escapeHtml(event.path)}</code> written (${writeSize})` });
    } else if (event.type === 'warning') {
      prog.steps.push({ html: `<span class="err">${escapeHtml(event.message)}</span>` });
    } else if (event.type === 'info') {
      prog.steps.push({ html: escapeHtml(event.message) });
      // Show info as a visible chat bubble (issue #31)
      this.messages.push({ type: 'info', text: event.message });
      this.$nextTick(() => {
        const el = this.$refs.messages;
        if (el) scrollToBottom(el);
      });
    }
  },

  _formatFileSize(bytes) {
    if (bytes == null) return '';
    if (bytes < 1024) return `${bytes} bytes`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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

  onInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      this.sendMessage();
    }
  },

  autoResize(e) {
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
  },
});
