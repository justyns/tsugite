import { get } from '../../api.js';
import { escapeHtml, contentBlockHtml, truncate } from '../../utils.js';

export const historyMixin = {
  _allHistoryMessages: [],
  _historyLoaded: 0,
  HISTORY_PAGE_SIZE: 20,
  hasMoreHistory: false,
  turns: [],
  compactionSummary: null,
  compactedFrom: null,
  compactionReason: null,

  _historyDebounceTimer: null,

  _debouncedLoadHistory() {
    if (this._historyDebounceTimer) clearTimeout(this._historyDebounceTimer);
    this._historyDebounceTimer = setTimeout(() => this.loadHistory(), 200);
  },

  resetHistory() {
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    this.hasMoreHistory = false;
    this.turns = [];
    this.compactionSummary = null;
    this.compactedFrom = null;
    this.compactionReason = null;
  },

  async loadHistory() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    this.resetHistory();
    try {
      let histUrl = `/api/agents/${agent}/history?user_id=${encodeURIComponent(this.userId)}&limit=100&detail=true`;
      if (this.selectedSessionId) histUrl += `&session_id=${encodeURIComponent(this.selectedSessionId)}`;
      const data = await get(histUrl);
      this.compactionSummary = data.compaction_summary || null;
      this.compactedFrom = data.compacted_from || null;
      this.compactionReason = data.compaction_reason || null;
      if (!data.turns || data.turns.length === 0) return;
      let pendingHooks = [];
      for (const turn of data.turns) {
        if (turn.type === 'compaction') {
          this._allHistoryMessages.push({ type: 'compaction', summary: turn.summary || null, reason: turn.reason || null });
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
            steps.push({ hasDetails: true, summary, content: output, open: false });
          } else {
            steps.push({ html: summary });
          }
        }
        pendingHooks = [];
        if (turn.messages) {
          for (const item of this.extractMessages(turn)) {
            if (item.type === 'tool_call') {
              steps.push({ hasDetails: true, summary: `<code>${escapeHtml(item.name)}</code>`, content: truncate(item.args), open: false });
            } else if (item.type === 'tool_result') {
              steps.push({ hasDetails: true, summary: `<code>${escapeHtml(item.name || 'result')}</code>`, content: item.content, open: false });
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
};
