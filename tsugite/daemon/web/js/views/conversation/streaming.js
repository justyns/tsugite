import { post, streamPost, uploadFiles, parseSSE } from '../../api.js';
import { escapeHtml, formatFileSize, contentBlockHtml } from '../../utils.js';

export const streamingMixin = {
  sending: false,
  _activeReader: null,
  _scrollTimer: null,

  _scrollThrottled() {
    if (this._scrollTimer) return;
    this._scrollTimer = setTimeout(() => {
      this._scrollTimer = null;
      this.scrollMessages();
    }, 150);
  },

  _pushDetailStep(prog, summary, content) {
    const follow = this.$store.app.autoFollow;
    if (follow && prog._lastOpenIdx != null) {
      const prev = prog.steps[prog._lastOpenIdx];
      if (prev?.hasDetails) prev.open = false;
    }
    const idx = prog.steps.length;
    prog.steps.push({ hasDetails: true, summary, content, open: follow });
    if (follow) {
      prog._lastOpenIdx = idx;
      this._scrollThrottled();
    }
  },

  async sendMessage() {
    const msg = this.messageText.trim();
    const agent = this.$store.app.selectedAgent;
    if ((!msg && !this.pendingFiles.length) || !agent || this.sending) return;

    const parsed = this._parseCommand(msg);
    if (parsed && !this.pendingFiles.length) {
      this.sending = true;
      this.messageText = '';
      this._clearDraft();
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
    this._clearDraft();
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
      let gotResult = false;

      for await (const event of parseSSE(reader)) {
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
          } else {
            this._handleProgressEvent(progressIdx, event);
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
        this._pushDetailStep(prog, 'thought', event.content);
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
        this._pushDetailStep(prog, summary, output);
      } else {
        prog.steps.push({ html: summary });
      }
    } else if (event.type === 'init') {
      prog.statusText = `Agent: ${event.agent}`;
      if (event.model) this.statusInfo = { ...this.statusInfo, model: event.model };
    } else if (event.type === 'content_block') {
      prog.steps.push({ html: contentBlockHtml(event.name, event.content || '') });
    } else if (event.type === 'code') {
      this._pushDetailStep(prog, `<code>code</code>`, event.content || '');
    } else if (event.type === 'tool_result') {
      const isCodeResult = event.tool === 'unknown';
      if (!isCodeResult) prog.toolCount++;
      const cls = event.success ? 'ok' : 'err';
      const label = isCodeResult ? 'output' : event.tool;
      const output = event.output || event.error || '';
      if (output) {
        this._pushDetailStep(prog, `<code>${escapeHtml(label)}</code> <span class="${cls}">${cls}</span>`, output);
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
};
