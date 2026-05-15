import { post, streamPost, uploadFiles, parseSSE } from '../../api.js';
import { escapeHtml, formatFileSize, contentBlockHtml } from '../../utils.js';
import { finalResultBubble, appendReasoningChunk, attachReasoningTokens } from './event_types.js';

export const streamingMixin = {
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

  // The active progress bubble is always the LAST element of the messages
  // array. Reasoning/thought/final_result split the bubble: they finalize the
  // live one, push their own bubble, and the next tool event creates a fresh
  // progress bubble. This keeps tool steps interleaved with prose chronologically.
  _currentLiveProgress(arr) {
    const last = arr[arr.length - 1];
    return last?.type === 'progress' ? last : null;
  },

  _ensureLiveProgress(arr) {
    const cur = this._currentLiveProgress(arr);
    if (cur) return cur;
    const prog = { type: 'progress', steps: [], statusText: 'Working...', turnCount: 0, toolCount: 0 };
    arr.push(prog);
    return prog;
  },

  _finalizeLiveProgress(arr) {
    const cur = this._currentLiveProgress(arr);
    if (!cur) return null;
    if (cur.steps.length > 0) cur.type = 'progress-done';
    else arr.pop();
    return cur;
  },

  async sendMessage() {
    const msg = this.messageText.trim();
    const agent = this.$store.app.selectedAgent;
    const sendSessionId = this.selectedSessionId;
    if ((!msg && !this.pendingFiles.length) || !agent || !sendSessionId) return;
    const sendState = this._sessionState(sendSessionId);
    if (sendState.sending) return;

    const parsed = this._parseCommand(msg);
    if (parsed && !this.pendingFiles.length) {
      sendState.sending = true;
      this.messageText = '';
      this._clearDraft();
      this._resetInputHeight();
      this.showCommandSuggestions = false;
      this.messages.push({ type: 'user', text: msg });
      this.scrollMessages(true);
      try {
        const result = await this._runCommand(parsed.command, parsed.args);
        this.messages.push({ type: 'agent', text: result });
      } catch (e) {
        this.messages.push({ type: 'error', text: `Command error: ${e.message}` });
      } finally {
        sendState.sending = false;
        this.scrollMessages(true);
      }
      return;
    }

    sendState.sending = true;
    this.messageText = '';
    this._clearDraft();
    this._resetInputHeight();

    // Re-resolve each call: loadHistory() may swap the array reference mid-stream.
    const sessMessages = () => this._sessionState(sendSessionId).messages;

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
        sendState.sending = false;
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
    sessMessages().push({ type: 'user', text: displayMsg });

    this.scrollMessages(true);

    try {
      const chatBody = { message: msg, user_id: this.userId, session_id: sendSessionId };
      if (uploadedFiles.length) chatBody.uploaded_files = uploadedFiles;
      const resp = await streamPost(`/api/agents/${agent}/chat`, chatBody);
      const reader = resp.body.getReader();
      sendState.reader = reader;
      let gotResult = false;

      for await (const event of parseSSE(reader)) {
          if (event.type === 'done') {
            break;
          } else if (event.type === 'cancelled') {
            this._finalizeLiveProgress(sessMessages());
            sessMessages().push({ type: 'info', text: 'Generation stopped.' });
            break;
          } else if (event.type === 'compacting') {
            this._sessionState(sendSessionId).compacting = true;
          } else if (event.type === 'compacted') {
            this._sessionState(sendSessionId).compacting = false;
          } else if (event.type === 'skill_loaded') {
            const state = this._sessionState(sendSessionId);
            if (!state.loadedSkills.some(s => s.name === event.name)) {
              state.loadedSkills.push({ name: event.name, description: event.description || '' });
            }
          } else if (event.type === 'skill_unloaded') {
            const state = this._sessionState(sendSessionId);
            state.loadedSkills = state.loadedSkills.filter(s => s.name !== event.name);
          } else if (event.type === 'ask_user') {
            this._finalizeLiveProgress(sessMessages());
            sessMessages().push({
              type: 'ask_user',
              question: event.question,
              questionType: event.question_type || 'text',
              options: event.options || [],
              answered: false,
              answer: '',
              inputValue: '',
            });
            this.scrollMessages(true);
          } else if (event.type === 'reaction') {
            const arr = sessMessages();
            for (let i = arr.length - 1; i >= 0; i--) {
              if (arr[i].type === 'user') {
                if (!arr[i].reactions) arr[i].reactions = [];
                arr[i].reactions.push(event.emoji);
                break;
              }
            }
          } else if (event.type === 'final_result') {
            gotResult = true;
            const arr = sessMessages();
            this._finalizeLiveProgress(arr);
            const bubble = finalResultBubble(event);
            if (bubble) this._pushFinalAgentBubble(arr, bubble);
          } else if (event.type === 'session_info') {
            this.updateStatusFromEvent(event, sendSessionId);
          } else {
            this._handleProgressEvent(event, sendSessionId);
          }
      }

      reader.cancel().catch(() => {});

      const arr = sessMessages();
      const prog = this._currentLiveProgress(arr);
      if (prog) {
        if (prog.steps.length > 0) {
          prog.type = 'progress-done';
          if (!gotResult && prog.errorText) {
            prog.failed = true;
            prog.lastMessage = msg;
          }
        } else {
          arr.pop();
        }
      }
    } catch (e) {
      const arr = sessMessages();
      const prog = this._currentLiveProgress(arr);
      if (prog && prog.steps.length === 0) arr.pop();
      arr.push({ type: 'error', text: `Connection error: ${e.message}` });
    } finally {
      sendState.reader = null;
      sendState.sending = false;
      this.scrollMessages();
    }
  },

  // For a turn whose final answer is prose, the `thought` event handler has
  // already pushed the same text as an agent bubble (agent.py emits
  // LLMMessageEvent for parsed.thought before returning the result). Replace
  // in-place instead of pushing a duplicate. Mirrors `sawInlineAgent` in
  // history.js's eventsToBubbles.
  _pushFinalAgentBubble(arr, bubble) {
    const last = arr[arr.length - 1];
    if (
      bubble.type === 'agent' &&
      last?.type === 'agent' &&
      (last.text || '').trim() === (bubble.text || '').trim()
    ) {
      arr.splice(arr.length - 1, 1, bubble);
      return;
    }
    arr.push(bubble);
  },

  _handleProgressEvent(event, sessionId = this.selectedSessionId) {
    const arr = this._sessionState(sessionId).messages;

    if (event.type === 'reasoning_content') {
      this._finalizeLiveProgress(arr);
      if (event.content) {
        appendReasoningChunk(arr, event.step, event.content);
        this._scrollThrottled();
      }
      return;
    }
    if (event.type === 'reasoning_tokens') {
      attachReasoningTokens(arr, event.step, event.tokens);
      return;
    }
    if (event.type === 'thought') {
      if (event.content) {
        this._finalizeLiveProgress(arr);
        arr.push({ type: 'agent', text: event.content });
        this._scrollThrottled();
      } else {
        this._ensureLiveProgress(arr).statusText = 'Thinking...';
      }
      return;
    }
    if (event.type === 'final_result') {
      this._finalizeLiveProgress(arr);
      const bubble = finalResultBubble(event);
      if (bubble) {
        arr.push(bubble);
        this.scrollMessages();
      }
      return;
    }

    const prog = this._ensureLiveProgress(arr);

    if (event.type === 'turn_start') {
      prog.turnCount++;
      prog.statusText = `Turn ${event.turn}...`;
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
      if (event.model) {
        const state = this._sessionState(sessionId);
        if (state) state.statusInfo = { ...state.statusInfo, model: event.model };
      }
    } else if (event.type === 'content_block') {
      prog.steps.push({ html: contentBlockHtml(event.name, event.content || '') });
    } else if (event.type === 'code') {
      this._pushDetailStep(prog, `<code>code</code>`, event.content || '');
    } else if (event.type === 'tool_call') {
      const name = event.tool || 'tool';
      const args = typeof event.arguments === 'string'
        ? event.arguments
        : JSON.stringify(event.arguments || {}, null, 2);
      if (args && args !== '{}') {
        this._pushDetailStep(prog, `<code>${escapeHtml(name)}</code>`, args);
      } else {
        prog.steps.push({ html: `<code>${escapeHtml(name)}</code>` });
      }
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
      arr.push({ type: 'info', text: event.message });
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
    const sid = this.selectedSessionId;
    if (!agent || !sid) return;
    const state = this._sessionState(sid);
    if (!state.sending) return;
    try {
      await post(`/api/agents/${agent}/chat/cancel`, { user_id: this.userId, session_id: sid });
    } catch (e) { /* best effort */ }
    if (state.reader) state.reader.cancel().catch(() => {});
  },
};
