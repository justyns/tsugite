import { get } from '../../api.js';
import { escapeHtml, contentBlockHtml } from '../../utils.js';
import { finalResultBubble, appendReasoningChunk, attachReasoningTokens } from './event_types.js';

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
export function eventsToBubbles(events, { dropTrailing = false } = {}) {
  const bubbles = [];
  let pendingHooks = [];
  let currentSteps = null;
  let currentUserBubble = null;
  let lastModelText = '';
  let sawInlineAgent = false;

  // Push the accumulated tool steps as a finalized progress bubble. Called both
  // at user_input boundaries (flushBubble) and when reasoning arrives mid-turn,
  // so reasoning bubbles slot between the tool blocks they were emitted between.
  function pushProgressIfHasSteps() {
    if (currentSteps && currentSteps.length) {
      const tools = currentSteps.filter(s => s._tool).length;
      const turns = currentSteps.filter(s => s._turn).length || 1;
      bubbles.push({
        type: 'progress-done',
        steps: _collapseRepeats(currentSteps),
        turnCount: turns,
        toolCount: tools,
      });
      currentSteps = [];
    }
  }

  function flushBubble() {
    pushProgressIfHasSteps();
    if (lastModelText) {
      bubbles.push({ type: 'agent', text: lastModelText });
    }
    currentSteps = null;
    currentUserBubble = null;
    lastModelText = '';
    sawInlineAgent = false;
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
        source_session_id: data.source_session_id || null,
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

    if (type === 'reasoning_content') {
      if (data.content) {
        pushProgressIfHasSteps();
        appendReasoningChunk(bubbles, data.step, data.content);
      }
      continue;
    }

    if (type === 'reasoning_tokens') {
      attachReasoningTokens(bubbles, data.step, data.tokens);
      continue;
    }

    if (type === 'model_response') {
      const raw = data.raw_content || '';
      const { prose, blocks } = _extractContentBlocks(raw);
      const textOnly = _stripCodeFences(prose);

      for (const [name, content] of Object.entries(blocks)) {
        currentSteps.push({ html: contentBlockHtml(name, content), _turn: false });
      }
      // Each turn's prose becomes its own visible agent bubble inline, so multi-turn
      // flows show every turn's answer (not just the last). Flush any pending tool
      // steps first so the prose lands chronologically AFTER the tools that produced it.
      if (textOnly) {
        pushProgressIfHasSteps();
        bubbles.push({ type: 'agent', text: textOnly });
        sawInlineAgent = true;
      }
      currentSteps.push({ _turn: true, _hidden: true });
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

    if (type === 'info') {
      if (!data.message) continue;
      pushProgressIfHasSteps();
      bubbles.push({ type: 'info', text: data.message });
      continue;
    }

    if (type === 'job_status') {
      if (!data.job_id) continue;
      pushProgressIfHasSteps();
      // Each job emits multiple state-transition events; collapse to the
      // latest state per job by updating the existing tile in place. The
      // first event for a job_id sets the bubble's position in the timeline.
      const existing = bubbles.find(b => b.type === 'job_status' && b.job_id === data.job_id);
      // Drop undefined fields so a later event without `error` (e.g. an early
      // RUNNING event replayed after a terminal STUCK event) doesn't wipe a
      // previously-set error from the collapsed tile.
      const fields = { job_id: data.job_id };
      for (const k of ['state', 'prompt', 'worker_session_id', 'verifier_session_id', 'verify_attempts', 'error']) {
        if (data[k] !== undefined) fields[k] = data[k];
      }
      if (existing) {
        Object.assign(existing, fields);
      } else {
        bubbles.push({ type: 'job_status', ...fields });
      }
      continue;
    }

    if (type === 'final_result') {
      const bubble = finalResultBubble(data);
      if (bubble?.type === 'return_value') {
        flushBubble();
        bubbles.push(bubble);
      } else if (bubble?.type === 'agent' && !sawInlineAgent && !lastModelText) {
        // Scheduled agents that go straight to return_value("...") with no model_response
        // would otherwise show only tool steps and no answer.
        lastModelText = bubble.text;
      }
      continue;
    }
  }

  // When a session is mid-flight, the trailing bubble's steps belong to a turn
  // that hasn't ended yet. We drop it here so rehydration owns the live bubble
  // (otherwise we'd render the same events twice — once as ✓ done, once live).
  if (!dropTrailing) flushBubble();

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

  _historyDebounceTimer: null,

  _debouncedLoadHistory() {
    if (this._historyDebounceTimer) clearTimeout(this._historyDebounceTimer);
    this._historyDebounceTimer = setTimeout(() => this.loadHistory(), 200);
  },

  resetHistory() {
    this._allHistoryMessages = [];
    this._historyLoaded = 0;
    this.hasMoreHistory = false;
    const state = this._sessionState(this.selectedSessionId);
    if (state) {
      state.compactionSummary = null;
      state.compactedIntoEvent = null;
    }
  },

  // dropTrailing only when the caller will spawn a live bubble for the
  // trailing turn (selectSession rehydrate). Default false, otherwise
  // the latest turn vanishes from rendered history.
  async loadHistory({ dropTrailing = false } = {}) {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    this.resetHistory();
    const sid = this.selectedSessionId;
    const state = this._sessionState(sid);
    let usedCache = false;
    try {
      let events;
      if (state && state.prefetchedEvents) {
        events = state.prefetchedEvents;
        state.prefetchedEvents = null;
        usedCache = true;
      } else {
        if (state) state.historyLoading = true;
        let url = `/api/agents/${agent}/history?user_id=${encodeURIComponent(this.userId)}&limit=100`;
        if (sid) url += `&session_id=${encodeURIComponent(sid)}`;
        const data = await get(url);
        events = data.events || [];
      }
      if (state) {
        // Last wins for multi-compaction chains.
        const lastCompact = events.findLast(e => e.type === 'compaction');
        if (lastCompact) state.compactionSummary = lastCompact.data?.summary || null;
        // The trailing `compacted_into` event (post-feature only) carries the
        // forward-direction banner's timestamp + counts. Legacy chains leave
        // compactedIntoEvent null; the banner falls back to bare-link rendering
        // driven by selectedSessionMeta.superseded_by.
        state.compactedIntoEvent = events.findLast(e => e.type === 'compacted_into') || null;
      }
      const meta = this.selectedSessionMeta;
      this._allHistoryMessages = eventsToBubbles(events, { dropTrailing });
      if (this._allHistoryMessages.length === 0 && meta) {
        if (meta.prompt) this._allHistoryMessages.push({ type: 'user', text: meta.prompt });
        if (meta.error) this._allHistoryMessages.push({ type: 'error', text: meta.error });
        if (meta.result) this._allHistoryMessages.push({ type: 'agent', text: meta.result });
      }
      this._showRecentHistory();
    } catch { /* ignore */ }
    finally {
      if (state && !usedCache) state.historyLoading = false;
    }
    this.scrollMessages(true);
  },

  _showRecentHistory() {
    const all = this._allHistoryMessages;
    const page = this.HISTORY_PAGE_SIZE;
    const startIdx = Math.max(0, all.length - page);
    const slice = all.slice(startIdx);
    this._historyLoaded = all.length - startIdx;
    this.hasMoreHistory = startIdx > 0;
    const next = [];
    if (this.hasMoreHistory) {
      next.push({ type: 'separator', text: `${startIdx} earlier messages — load more ↑` });
    }
    next.push(...slice);
    if (slice.length > 0) {
      next.push({ type: 'separator', text: `${this._historyLoaded} of ${all.length} messages loaded` });
    }
    // Mutate the session's message array in place so in-flight stream pushes
    // (which hold a reference to the same array) target the rebuilt content.
    const arr = this.messages;
    arr.length = 0;
    arr.push(...next);
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
