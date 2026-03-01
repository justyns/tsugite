import { get, post, streamPost } from '../api.js';
import { escapeHtml, renderMarkdown, scrollToBottom } from '../utils.js';

export default () => ({
  messages: [],
  sessions: [],
  sending: false,
  compacting: false,
  statusInfo: {},
  messageText: '',
  showAttachments: false,
  attachments: [],
  showSkills: false,
  loadedSkills: [],

  init() {
    const maybeReload = () => {
      if (this.$store.app.view === 'chat' && this.$store.app.selectedAgent) this.reload();
    };
    this.$watch('$store.app.selectedAgent', maybeReload);
    this.$watch('$store.app.view', (view) => {
      if (view === 'chat' && this.$store.app.selectedAgent && this.messages.length === 0) this.reload();
    });
  },

  async reload() {
    this.messages = [];
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
    try {
      const data = await get(`/api/agents/${agent}/history?user_id=${encodeURIComponent(this.userId)}`);
      if (!data.turns || data.turns.length === 0) return;
      let msgCount = 0;
      for (const turn of data.turns) {
        if (turn.type === 'compaction') {
          this.messages.push({ type: 'compaction', summary: turn.summary || null });
          continue;
        }
        if (turn.user) this.messages.push({ type: 'user', text: turn.user });
        if (turn.assistant) this.messages.push({ type: 'agent', text: turn.assistant });
        msgCount++;
      }
      if (msgCount > 0) {
        this.messages.push({ type: 'separator', text: `${msgCount} earlier turn${msgCount !== 1 ? 's' : ''} restored` });
      }
    } catch { /* ignore */ }
    this.$nextTick(() => {
      const el = this.$refs.messages;
      if (el) scrollToBottom(el);
    });
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

  async sendMessage() {
    const msg = this.messageText.trim();
    const agent = this.$store.app.selectedAgent;
    if (!msg || !agent || this.sending) return;

    this.sending = true;
    this.messageText = '';
    this.messages.push({ type: 'user', text: msg });

    this.$nextTick(() => {
      const el = this.$refs.messages;
      if (el) scrollToBottom(el);
    });

    const progressIdx = this.messages.length;
    this.messages.push({ type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 });

    try {
      const resp = await streamPost(`/api/agents/${agent}/chat`, { message: msg, user_id: this.userId });
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
            this.messages.splice(progressIdx, 1);
            this.messages.push({ type: 'error', text: event.error });
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
