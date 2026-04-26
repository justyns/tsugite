import { get } from '../../api.js';
import { escapeHtml, contentBlockHtml } from '../../utils.js';

const _CONTENT_BLOCK_RE = /<(?:tsu:)?content\s+name="([^"]+)">([\s\S]*?)<\/(?:tsu:)?content>/g;

function _extractContentBlocks(raw) {
  const blocks = {};
  if (!raw) return { prose: '', blocks };
  const prose = raw.replace(_CONTENT_BLOCK_RE, (_, name, content) => {
    blocks[name] = content.replace(/^\n/, '').replace(/\n$/, '');
    return '';
  });
  return { prose, blocks };
}

// Only python fences are tsugite tool-execution blocks (they're surfaced as
// separate code_execution steps). Plain ``` fences are part of the agent's
// prose answer and must stay visible.
function _stripCodeFences(text) {
  return text.replace(/```python\n[\s\S]*?```/g, '').trim();
}

function _firstCodeBlock(text) {
  const m = text.match(/```python\n([\s\S]*?)\n```/);
  return m ? m[1] : null;
}

function _formatDuration(ms) {
  if (!ms || ms < 0) return '';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// Tags the daemon prepends to the user message that are noise in the chat
// bubble (datetime, working_directory, scratchpad, etc). Anything inside one
// of these tags is folded behind a "context" toggle.
const _PREPENDED_CONTEXT_TAGS = ['message_context', 'environment', 'background_task_complete', 'scheduled_task'];

function _splitUserInput(raw) {
  let body = raw || '';
  const folded = [];
  let progress = true;
  while (progress) {
    progress = false;
    body = body.replace(/^\s+/, '');
    for (const tag of _PREPENDED_CONTEXT_TAGS) {
      const re = new RegExp(`^<${tag}\\b[^>]*>([\\s\\S]*?)</${tag}>`, '');
      const m = body.match(re);
      if (m) {
        folded.push({ tag, content: m[0] });
        body = body.slice(m[0].length);
        progress = true;
        break;
      }
    }
  }
  return { text: body.trim(), folded };
}

/**
 * Walk the event log and build display bubbles. One progress bubble spans
 * `user_input → next user_input` (with the final agent text peeled off as a
 * trailing agent bubble). Hooks before any user_input buffer and attach to the
 * next bubble. Reactions wire onto the preceding user bubble.
 */
export function eventsToBubbles(events) {
  const bubbles = [];
  let pendingHooks = [];
  let currentSteps = null;
  let currentUserBubble = null;
  let lastModelText = '';

  function flushBubble() {
    if (currentSteps && currentSteps.length) {
      const tools = currentSteps.filter(s => s._tool).length;
      const turns = currentSteps.filter(s => s._turn).length || 1;
      bubbles.push({
        type: 'progress-done',
        steps: _collapseRepeats(currentSteps),
        turnCount: turns,
        toolCount: tools,
      });
    }
    if (lastModelText) {
      bubbles.push({ type: 'agent', text: lastModelText });
    }
    currentSteps = null;
    currentUserBubble = null;
    lastModelText = '';
  }

  for (const ev of events) {
    const type = ev.type;
    const data = ev.data || {};

    if (type === 'session_start' || type === 'session_end') {
      continue;
    }

    if (type === 'compaction') {
      flushBubble();
      bubbles.push({
        type: 'compaction',
        summary: data.summary || null,
        reason: data.reason || null,
      });
      continue;
    }

    if (type === 'user_input') {
      flushBubble();
      const split = _splitUserInput(data.text || '');
      currentUserBubble = { type: 'user', text: split.text };
      if (split.folded.length) {
        currentUserBubble.context = split.folded;
      }
      bubbles.push(currentUserBubble);
      currentSteps = [];
      // Attach any hooks that fired before any user_input.
      for (const h of pendingHooks) currentSteps.push(_hookStep(h));
      pendingHooks = [];
      continue;
    }

    if (type === 'reaction') {
      if (currentUserBubble) {
        currentUserBubble.reactions = currentUserBubble.reactions || [];
        currentUserBubble.reactions.push(data.emoji);
      }
      continue;
    }

    if (type === 'hook_execution') {
      if (currentSteps !== null) {
        currentSteps.push(_hookStep(data));
      } else {
        pendingHooks.push(data);
      }
      continue;
    }

    if (currentSteps === null) continue; // events before any user_input

    if (type === 'model_response') {
      const raw = data.raw_content || '';
      const { prose, blocks } = _extractContentBlocks(raw);
      const textOnly = _stripCodeFences(prose);
      lastModelText = textOnly;

      // Surface inline content blocks as their own steps.
      for (const [name, content] of Object.entries(blocks)) {
        currentSteps.push({ html: contentBlockHtml(name, content), _turn: false });
      }
      // The thought prose (between user msg and code block) becomes a step.
      const thought = textOnly && _firstCodeBlock(prose) ? textOnly : '';
      if (thought) {
        currentSteps.push({
          hasDetails: true,
          summary: 'thought',
          content: thought,
          open: false,
          _turn: true,
        });
      } else {
        // Mark that a turn happened so turn count is tracked.
        currentSteps.push({ _turn: true, _hidden: true });
      }
      continue;
    }

    if (type === 'code_execution') {
      const code = data.code || '';
      const dur = _formatDuration(data.duration_ms);
      currentSteps.push({
        hasDetails: true,
        summary: `<code>code</code>${dur ? ` <span class="step-dur">(${dur})</span>` : ''}`,
        content: code,
        open: false,
        _tool: true,
      });
      const output = data.output || '';
      const error = data.error || null;
      if (output || error) {
        currentSteps.push({
          hasDetails: true,
          summary: error ? `<code>error</code>` : `<code>result</code>`,
          content: error ? `${error}\n\n${output}`.trim() : output,
          open: false,
        });
      }
      continue;
    }

    if (type === 'tool_invocation') {
      const dur = _formatDuration(data.duration_ms);
      const name = data.name || 'tool';
      currentSteps.push({
        hasDetails: true,
        summary: `<code>${escapeHtml(name)}</code>${dur ? ` <span class="step-dur">(${dur})</span>` : ''}`,
        content: typeof data.args === 'string' ? data.args : JSON.stringify(data.args || {}, null, 2),
        open: false,
        _tool: true,
        _toolName: name,
      });
      if (data.output || data.error) {
        currentSteps.push({
          hasDetails: true,
          summary: data.error ? `<code>error</code>` : `<code>result</code>`,
          content: data.error ? `${data.error}\n\n${data.output || ''}`.trim() : data.output,
          open: false,
        });
      }
      continue;
    }

    if (type === 'format_error') {
      currentSteps.push({
        html: `<div class="step-error">Format error: ${escapeHtml(data.reason || '')}</div>`,
      });
      const rejected = data.rejected_content || '';
      if (rejected) {
        currentSteps.push({
          hasDetails: true,
          summary: '<code>rejected response</code>',
          content: rejected,
          open: false,
        });
      }
      continue;
    }
  }

  flushBubble();

  // Drop steps that only existed to mark a turn (no visible content).
  for (const b of bubbles) {
    if (b.type === 'progress-done') {
      b.steps = b.steps.filter(s => !s._hidden);
    }
  }
  return bubbles.filter(b => b.type !== 'progress-done' || b.steps.length > 0);
}

function _hookStep(data) {
  const name = data.name || data.phase || 'hook';
  const exit = data.exit_code ?? 0;
  const summary = `<code>hook:${escapeHtml(name)}</code>${exit !== 0 ? ` <span class="step-error">(exit ${exit})</span>` : ''}`;
  const output = [data.stdout, data.stderr].filter(Boolean).join('\n');
  if (output) {
    return { hasDetails: true, summary, content: output, open: false };
  }
  return { html: summary };
}

function _collapseRepeats(steps) {
  // When the same tool runs N times consecutively, group as "tool ×N".
  const out = [];
  let i = 0;
  while (i < steps.length) {
    const s = steps[i];
    if (!s._tool || !s._toolName) {
      out.push(s);
      i++;
      continue;
    }
    let j = i + 1;
    let count = 1;
    while (j < steps.length && steps[j]._tool && steps[j]._toolName === s._toolName) {
      count++;
      j++;
    }
    if (count > 2) {
      out.push({
        hasDetails: true,
        summary: `<code>${escapeHtml(s._toolName)}</code> <span class="step-dur">×${count}</span>`,
        content: steps.slice(i, j).map(x => x.content || '').join('\n---\n'),
        open: false,
        _tool: true,
      });
      i = j;
    } else {
      out.push(s);
      i++;
    }
  }
  return out;
}

export const historyMixin = {
  _allHistoryMessages: [],
  _historyLoaded: 0,
  HISTORY_PAGE_SIZE: 20,
  hasMoreHistory: false,
  compactionSummary: null,

  _historyDebounceTimer: null,

  _debouncedLoadHistory() {
    if (this._historyDebounceTimer) clearTimeout(this._historyDebounceTimer);
    this._historyDebounceTimer = setTimeout(() => this.loadHistory(), 200);
  },

  resetHistory() {
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    this.hasMoreHistory = false;
    this.compactionSummary = null;
  },

  async loadHistory() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    this.resetHistory();
    try {
      let url = `/api/agents/${agent}/history?user_id=${encodeURIComponent(this.userId)}&limit=100`;
      if (this.selectedSessionId) url += `&session_id=${encodeURIComponent(this.selectedSessionId)}`;
      const data = await get(url);
      const events = data.events || [];
      // Surface the most recent compaction event for the banner.
      const lastCompact = [...events].reverse().find(e => e.type === 'compaction');
      if (lastCompact) {
        this.compactionSummary = lastCompact.data?.summary || null;
      }
      this._allHistoryMessages = eventsToBubbles(events);
      if (this._allHistoryMessages.length === 0 && this.selectedSessionMeta) {
        const meta = this.selectedSessionMeta;
        if (meta.prompt) this._allHistoryMessages.push({ type: 'user', text: meta.prompt });
        if (meta.error) this._allHistoryMessages.push({ type: 'error', text: meta.error });
        if (meta.result) this._allHistoryMessages.push({ type: 'agent', text: meta.result });
      }
      this._showRecentHistory();
    } catch { /* ignore */ }
    this.scrollMessages(true);
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
};
