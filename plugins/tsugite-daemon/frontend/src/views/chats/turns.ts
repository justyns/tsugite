/**
 * Event → timeline reducer for the conversation pane. Normalizes the two event
 * shapes into one turn model:
 *   - REPLAY (GET /api/sessions/{id}/events): persisted history, normalized
 *     server-side - user_input{text, injected?, display_text?}, model_request,
 *     model_response{thought, content_blocks?, tail?}, final_result,
 *     session_end, session_info, compaction.
 *   - LIVE (per-chat SSE, api/chat.ts): UI events - init, turn_start,
 *     thought{content}, model_response{thought, content_blocks, tail} (the
 *     settled parse of each model turn), reasoning_content{content},
 *     tool_call/tool_result, final_result{result,result_data}, session_info.
 *
 * The backend is the only parser of model output: this reducer consumes the
 * served parse and never derives structure from raw_content itself.
 *
 * All AI activity between two user messages folds into one AI turn (one bubble
 * per response), carrying an ordered list of blocks.
 * Pure + node-testable; the controller feeds it replay first, then re-runs it
 * over replay+live as frames arrive.
 */

import type { JobLike } from '$lib/stores/jobsFilter';

export interface ProseBlock {
  kind: 'prose';
  text: string;
}
export interface ThinkBlock {
  kind: 'think';
  content: string;
  tokens?: number;
  label: string;
}
export interface ExecBlock {
  kind: 'exec';
  command: string;
  /** `ended` = the turn closed before a result arrived; resolved neutrally so a
   *  block is never a permanent spinner (design: closed, outcome unknown). */
  status: 'running' | 'done' | 'error' | 'ended';
  exitCode?: number;
  output?: string;
  meta?: string;
  /** Native tool_call arguments, surfaced in the disclosure body on live + replay. */
  args?: Record<string, unknown>;
  /** tool call id, so a later tool_result folds into this block. */
  callId?: string;
}
/** One tool call observed inside a code execution. Live: a `tool_call` opens
 *  it running and its `tool_result_audit` resolves it. Replay: persisted
 *  `tool_calls` records carry args/output/duration per call (older history has
 *  only `tools_called` names). `ended` = the run died before the call resolved. */
export interface CodeCall {
  tool: string;
  status: 'running' | 'done' | 'error' | 'ended';
  args?: Record<string, unknown>;
  output?: string;
  meta?: string;
  /** The `tsu_group` this call ran inside, if any. */
  groupId?: string;
}

/** A `tsu_group` section opened during a code execution. */
export interface CodeGroup {
  id: string;
  title: string;
  success?: boolean;
  meta?: string;
  error?: string;
}
export interface CodeBlock {
  kind: 'code';
  code: string;
  lang?: string;
  filename?: string;
  /** `running` while the live execution streams; `done` once the model moves
   *  on (next block / turn end) or for replayed bundles. */
  status: 'running' | 'done';
  calls: CodeCall[];
  /** Named sections opened this execution, in the order they opened. */
  groups?: CodeGroup[];
  /** Persisted code_execution replay carries the combined run output inline
   *  (the live path streams per-call results instead). */
  output?: string;
  returnValue?: string;
  /** Run duration ("0.4s"), from the persisted bundle. */
  meta?: string;
}
/** A named `<content name="x">` block from the model response - file content
 *  defined outside code (injected into the fence as a variable). */
export interface ContentBlock {
  kind: 'content';
  name: string;
  text: string;
}
export interface ResultBlock {
  kind: 'result';
  data: unknown;
}
export interface JobBlock {
  kind: 'job';
  job: JobLike;
}
export interface ErrorBlock {
  kind: 'error';
  message: string;
}
/** A calm, informational line - not a failure. Currently the backend's
 *  resume_reset boundary (the chat's unresumable provider session was severed
 *  and the turn continued from saved history). */
export interface NoticeBlock {
  kind: 'notice';
  message: string;
}
export interface DeliveryBlock {
  kind: 'delivery';
  message: string;
  source: string;
  needsAck: boolean;
  deliveryId?: string;
  title?: string;
}

export type TurnBlock =
  | ProseBlock
  | ThinkBlock
  | ExecBlock
  | CodeBlock
  | ContentBlock
  | ResultBlock
  | JobBlock
  | ErrorBlock
  | NoticeBlock
  | DeliveryBlock;

/** A context-injection block peeled off the front of a user_input (scheduled
 *  task results, environment/context prefixes) - shown folded, never as the
 *  user's own words. `client_context` carries structured `items` (attached
 *  location, ...) rendered as label:value rows instead of a raw `body`. */
export interface InjectedBlock {
  tag: string;
  id?: string;
  body: string;
  items?: { key: string; label: string; value: string; untrusted?: boolean }[];
}

/** A file/photo the person attached to a user message. Rendered under the bubble
 *  (images as thumbnails, other files as chips). `type` is the content-type class
 *  the recorder stored ("image" | "text" | "audio" | "document"); `path` is the
 *  workspace-relative path the raw-bytes endpoint and the files view both read. */
export interface TurnAttachment {
  name: string;
  type: string;
  path: string;
}

export interface Turn {
  id: string;
  role: 'user' | 'ai';
  at: string;
  blocks: TurnBlock[];
  /** Context injections split from this user turn's text. */
  injected?: InjectedBlock[];
  /** Files/photos attached to this user message, rendered once under the bubble. */
  attachments?: TurnAttachment[];
  /** True when the "user" turn is purely an injection (e.g. a scheduled-task
   *  result) - the person never typed it, so it must not render as "you". */
  synthetic?: boolean;
  /** The AI turn has begun but no final_result/session_end has closed it. */
  streaming?: boolean;
  /** Raw token-stream buffer (stream_chunk deltas). Rendered live as markdown,
   *  then DISCARDED when the model_response frame (the settled backend parse)
   *  supersedes it; flushed verbatim only if the turn dies first. */
  stream?: string;
  /** Live-only mid-turn status line (hook_status "Running x..." ticks); shown
   *  by the waiting Work line, cleared when the hook resolves or the turn ends.
   *  Never persisted, so replay carries none. */
  liveStatus?: string;
  /** Cache usage across the turn's model_response steps (present AND live).
   *  `cacheRead` is the LAST step's read - the current cached-prefix size, the
   *  footer headline that matches the context meter's scale (not the misleading
   *  cross-step sum). `cacheReadTotal`/`cacheWriteTotal` sum every step for the
   *  tooltip; `cacheSteps` counts steps that reported cache. All absent when no
   *  step reported cache, so the footer never fabricates a "0 cached". */
  meta?: {
    tokens?: number;
    cost?: number;
    turns?: number;
    cacheRead?: number;
    cacheReadTotal?: number;
    cacheWriteTotal?: number;
    cacheSteps?: number;
  };
}

export interface Compaction {
  id: string;
  at: string;
  replacedCount?: number;
  retainedCount?: number;
  sourceId?: string;
  /** The retained summary text the compaction wrote in place of the folded turns. */
  summary?: string;
}

/** A still-open interactive prompt, derived purely from the durable event log: an
 *  `ask_user` with no later `ask_answered` for the same `askId`. Carrying it on the
 *  timeline is what lets a reloaded page re-show a pending approval / question. */
export interface PendingAsk {
  askId?: string;
  question: string;
  questionType: 'text' | 'yes_no' | 'choice' | 'approval';
  options: string[];
}

export interface Timeline {
  turns: Turn[];
  compactions: Compaction[];
  context?: { tokens: number; limit: number };
  /** The prompt still awaiting an answer, or null. Reconciled from persisted
   *  `ask_user` / `ask_answered` events, so it survives a reload. */
  pendingAsk: PendingAsk | null;
}

type Event = Record<string, unknown>;

const str = (v: unknown): string | undefined => (typeof v === 'string' ? v : undefined);
const num = (v: unknown): number | undefined => (typeof v === 'number' ? v : undefined);
const strArr = (v: unknown): string[] | undefined =>
  Array.isArray(v) && v.every((x) => typeof x === 'string') ? (v as string[]) : undefined;
const rec = (v: unknown): Record<string, unknown> | undefined =>
  v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;

function formatMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)}s`;
}

/** Normalize a failure string so an error-echoing prose block can be matched
 *  against the error frame's message: strip the `[Error: …]` wrapper the backend
 *  adds when it surfaces a provider failure as model content, plus the SDK's
 *  trailing ` (subtype=…)`. The stream_chunk form omits the subtext, so the two
 *  normalize equal. */
function errorCore(text: string): string {
  let t = text.trim();
  const wrapped = /^\[Error:\s*([\s\S]*)]$/.exec(t);
  if (wrapped) t = wrapped[1]!.trim();
  return t.replace(/\s*\(subtype=[^)]*\)\s*$/, '').trim();
}

/** Persisted per-call records ({tool, arguments, success, duration_ms,
 *  output|error}) -> CodeCall rows. Undefined when the event carries none
 *  (older history), so the caller can fall back to bare tools_called names. */
function parseToolCalls(v: unknown): CodeCall[] | undefined {
  if (!Array.isArray(v)) return undefined;
  const calls: CodeCall[] = [];
  for (const item of v) {
    const r = rec(item);
    if (!r) continue;
    const dur = num(r.duration_ms);
    const output = str(r.output) ?? str(r.error);
    calls.push({
      tool: str(r.tool) ?? 'tool',
      status: r.success === false ? 'error' : 'done',
      ...(rec(r.arguments) ? { args: rec(r.arguments) } : {}),
      ...(output != null ? { output } : {}),
      ...(dur != null ? { meta: formatMs(dur) } : {}),
      ...(str(r.group_id) != null ? { groupId: str(r.group_id) } : {}),
    });
  }
  return calls;
}

/** Persisted group records ({group_id, title, success, duration_ms, error})
 *  -> CodeGroup rows; undefined when the event carries none. */
function parseGroups(v: unknown): CodeGroup[] | undefined {
  if (!Array.isArray(v)) return undefined;
  const groups: CodeGroup[] = [];
  for (const item of v) {
    const r = rec(item);
    const id = r && str(r.group_id);
    if (!id) continue;
    const dur = num(r.duration_ms);
    groups.push({
      id,
      title: str(r.title) ?? '',
      ...(typeof r.success === 'boolean' ? { success: r.success } : {}),
      ...(dur != null ? { meta: formatMs(dur) } : {}),
      ...(str(r.error) != null ? { error: str(r.error) } : {}),
    });
  }
  return groups.length ? groups : undefined;
}

function namedCalls(names: string[] | undefined): CodeCall[] {
  return (names ?? []).map((tool) => ({ tool, status: 'done' as const }));
}

/** Structured {key,label,value} rows off a client_context injection; undefined
 *  when the block carries none (a plain body-only injection). */
function parseContextItems(v: unknown): InjectedBlock['items'] {
  if (!Array.isArray(v)) return undefined;
  const items: NonNullable<InjectedBlock['items']> = [];
  for (const item of v) {
    const r = rec(item);
    const key = r && str(r.key);
    if (!key) continue;
    items.push({
      key,
      label: str(r?.label) ?? key,
      value: str(r?.value) ?? '',
      ...(r?.untrusted === true ? { untrusted: true } : {}),
    });
  }
  return items.length ? items : undefined;
}

/** Server-split injection blocks off a user_input event ({tag, id?, body}); a
 *  client_context block also carries structured `items`. */
function parseInjected(v: unknown): InjectedBlock[] {
  if (!Array.isArray(v)) return [];
  const blocks: InjectedBlock[] = [];
  for (const item of v) {
    const r = rec(item);
    const tag = r && str(r.tag);
    if (!tag) continue;
    const id = str(r?.id);
    const items = parseContextItems(r?.items);
    blocks.push({
      tag,
      ...(id ? { id } : {}),
      body: str(r?.body) ?? '',
      ...(items ? { items } : {}),
    });
  }
  return blocks;
}

/** Uploaded files/photos recorded on a user_input event as {name, type,
 *  source_url}. source_url is null for uploads, so the bytes are located by name
 *  under the workspace uploads dir; that `uploads/<name>` path is what the raw
 *  endpoint and the files view both read. */
function parseAttachments(v: unknown): TurnAttachment[] | undefined {
  if (!Array.isArray(v)) return undefined;
  const out: TurnAttachment[] = [];
  for (const item of v) {
    const r = rec(item);
    const name = r && str(r.name);
    if (!name) continue;
    out.push({ name, type: str(r?.type) ?? '', path: `uploads/${name}` });
  }
  return out.length ? out : undefined;
}

/** Split a live token-stream buffer at an UNCLOSED code fence, so the tail
 *  can render in the real code panel while it streams. Closed fences stay in
 *  the text (Prose renders them normally). */
export function splitStreamFence(raw: string): { text: string; code?: string } {
  const opener = /```[^\n]*\n?/g;
  let open: { start: number; bodyStart: number } | null = null;
  let m: RegExpExecArray | null;
  while ((m = opener.exec(raw))) {
    if (open)
      open = null; // this match closes the previous fence
    else open = { start: m.index, bodyStart: m.index + m[0].length };
  }
  if (!open) return { text: raw };
  return {
    text: raw.slice(0, open.start).replace(/\s+$/, ''),
    code: raw.slice(open.bodyStart),
  };
}

function key(e: Event, i: number): string {
  const id = e.id ?? e.call_id ?? e.timestamp;
  return id != null ? `e${String(id)}` : `i${i}`;
}

class Builder {
  turns: Turn[] = [];
  compactions: Compaction[] = [];
  context: { tokens: number; limit: number } | undefined;
  private ai: Turn | null = null;
  private seq = 0;
  /** return_value reprs already shown on a code block, so a final_result that
   *  merely echoes one isn't rendered a second time as trailing prose. */
  private returnReprs = new Set<string>();
  /** Hook rows recorded BEFORE their user_input (pre_message hooks fire before
   *  the message persists): held here and attached to the next AI turn, never
   *  rendered as a phantom agent turn floating above the user's message. */
  private pendingHooks: ExecBlock[] = [];
  /** The interactive prompt still awaiting an answer, folded from the durable
   *  ask_user / ask_answered events (not a timeline block). */
  pendingAsk: PendingAsk | null = null;

  private uid(prefix: string): string {
    this.seq += 1;
    return `${prefix}-${this.seq}`;
  }

  private ensureAi(at: string): Turn {
    if (!this.ai) {
      this.ai = {
        id: this.uid('ai'),
        role: 'ai',
        at,
        blocks: [...this.pendingHooks],
        streaming: true,
      };
      this.pendingHooks = [];
      this.turns.push(this.ai);
      // return_value dedupe is scoped to the turn: a later turn's final_result
      // must not be suppressed by an identical repr shown in an earlier one.
      this.returnReprs.clear();
    }
    return this.ai;
  }

  /** Flush a trailing stashed hook (fired with no model events after it) so it
   *  still renders; the turn is settled, not a live spinner. */
  finish(): void {
    if (this.pendingHooks.length) {
      const turn = this.ensureAi(this.turns[this.turns.length - 1]?.at ?? '');
      turn.streaming = false;
    }
  }

  /** True when a prose block already carries exactly this text (the settled
   *  thought / flushed stream of a prose-final turn), so final_result would
   *  repeat the reply. Endings count too: consecutive thoughts merge into one
   *  block, and the result echoes only the final turn's text. */
  private hasProseEchoing(turn: Turn, text: string): boolean {
    const wanted = text.trim();
    if (!wanted) return true;
    return turn.blocks.some((b) => {
      if (b.kind !== 'prose') return false;
      const have = b.text.trim();
      return have === wanted || have.endsWith(wanted);
    });
  }

  /** The most recent still-running exec block a result should fold into (matched
   *  by call id when present; native tool_call/tool_result_audit pairs carry none,
   *  so they resolve against the last open exec, which sequential execution keeps
   *  correct). */
  private openExec(turn: Turn, callId?: string): ExecBlock | undefined {
    for (let i = turn.blocks.length - 1; i >= 0; i--) {
      const b = turn.blocks[i]!;
      if (b.kind === 'exec' && b.status === 'running' && (callId == null || b.callId === callId))
        return b;
    }
    return undefined;
  }

  /** Close any exec still marked running when its turn ends: `done` if a result
   *  arrived, else `ended` (neutral) - never a permanent spinner. */
  private resolveOpenExecs(turn: Turn): void {
    for (const b of turn.blocks) {
      if (b.kind === 'exec' && b.status === 'running')
        b.status = b.output != null ? 'done' : 'ended';
    }
    this.closeOpenCode(turn);
    this.flushStream(turn);
    turn.liveStatus = undefined;
  }

  /** The live code block still collecting tool_result observations. */
  private openCode(turn: Turn): CodeBlock | undefined {
    for (let i = turn.blocks.length - 1; i >= 0; i--) {
      const b = turn.blocks[i]!;
      if (b.kind === 'code' && b.status === 'running') return b;
    }
    return undefined;
  }

  /** A live execution is over once the model moves on (any next block) or the
   *  turn resolves - there is no explicit end event on the live stream. A call
   *  still running then closes as `ended` (outcome unknown, never a spinner). */
  private closeOpenCode(turn: Turn): void {
    for (const b of turn.blocks) {
      if (b.kind !== 'code' || b.status !== 'running') continue;
      b.status = 'done';
      for (const call of b.calls) {
        if (call.status === 'running') call.status = call.output != null ? 'done' : 'ended';
      }
    }
  }

  /** Safety net for a turn that dies with an unsettled token buffer: render it
   *  verbatim as prose. In the normal flow the model_response frame (the
   *  settled backend parse) discards the buffer before anything else lands. */
  private flushStream(turn: Turn): void {
    if (!turn.stream) return;
    const text = turn.stream.trim();
    if (text) this.addProse(turn, text);
    turn.stream = undefined;
  }

  /** Add a content block once per turn: on the live stream the same block
   *  arrives twice - inside the settled model_response frame AND as the
   *  backend's extracted content_block event - in either order. */
  private pushContent(turn: Turn, name: string, text: string): void {
    const dupe = turn.blocks.some(
      (b) => b.kind === 'content' && b.name === name && b.text === text,
    );
    if (!dupe) turn.blocks.push({ kind: 'content', name, text });
  }

  /** The last still-running call on the open code block (optionally by tool). */
  private openCodeCall(code: CodeBlock, tool?: string): CodeCall | undefined {
    for (let i = code.calls.length - 1; i >= 0; i--) {
      const c = code.calls[i]!;
      if (c.status === 'running' && (tool == null || c.tool === tool)) return c;
    }
    return undefined;
  }

  /** Fold one model_response step's usage dump into the turn's cache meta. The
   *  headline `cacheRead` tracks the LAST step's read (the current cached-prefix
   *  size); the totals sum every step for the tooltip and cacheSteps counts the
   *  steps. A field the provider didn't report is absent from the dump (never 0),
   *  so it stays undefined until some step carries it - honest absence. */
  private addCacheUsage(turn: Turn, usage: Record<string, unknown> | undefined): void {
    if (!usage) return;
    // OpenAI-family providers (openai_compat, codex_cli) report cached prompt reads
    // on the unified `cached_tokens`, not Anthropic's cache_read_input_tokens. Prefer
    // the explicit Anthropic read (Anthropic folds creation+read into cached_tokens,
    // so it must never win over the explicit read); otherwise take cached_tokens.
    const read = num(usage.cache_read_input_tokens) ?? num(usage.cached_tokens);
    const write = num(usage.cache_creation_input_tokens);
    if (read == null && write == null) return;
    turn.meta ??= {};
    if (read != null) {
      turn.meta.cacheRead = read; // last step wins - the current cached prefix
      turn.meta.cacheReadTotal = (turn.meta.cacheReadTotal ?? 0) + read;
    }
    if (write != null) turn.meta.cacheWriteTotal = (turn.meta.cacheWriteTotal ?? 0) + write;
    turn.meta.cacheSteps = (turn.meta.cacheSteps ?? 0) + 1;
  }

  private addProse(turn: Turn, text: string): void {
    if (!text) return;
    const last = turn.blocks[turn.blocks.length - 1];
    if (last && last.kind === 'prose') last.text += `\n\n${text}`;
    else turn.blocks.push({ kind: 'prose', text });
  }

  /** The last turn iff it's an AI turn - the row a trailing error frame folds
   *  into. A provider error is persisted AFTER session_end closed the turn
   *  (`this.ai` is null by then), so without this the error would open a second
   *  empty AI row instead of joining the failure it belongs to. */
  private tailAi(): Turn | undefined {
    const last = this.turns[this.turns.length - 1];
    return last?.role === 'ai' ? last : undefined;
  }

  /** Drop any prose block that merely echoes the failure text, so the styled
   *  error block is the turn's only rendering of it (the model_response content
   *  carries the same failure as `[Error: …]`). */
  private dropErrorEcho(turn: Turn, message: string): void {
    const core = errorCore(message);
    turn.blocks = turn.blocks.filter((b) => !(b.kind === 'prose' && errorCore(b.text) === core));
  }

  private addThink(turn: Turn, content: string): void {
    const last = turn.blocks[turn.blocks.length - 1];
    if (last && last.kind === 'think') last.content += content;
    else turn.blocks.push({ kind: 'think', content, label: 'thinking' });
  }

  pushUser(e: Event, at: string): void {
    this.ai = null;
    // The endpoint splits context injections server-side; display_text is the
    // message the person actually typed ("" for a pure injection turn).
    const blocks = parseInjected(e.injected);
    const rest = (blocks.length ? (str(e.display_text) ?? '') : (str(e.text) ?? '')).trim();
    const attachments = parseAttachments(e.attachments);
    this.turns.push({
      id: this.uid('user'),
      role: 'user',
      at,
      blocks: rest ? [{ kind: 'prose', text: rest }] : [],
      ...(blocks.length ? { injected: blocks } : {}),
      ...(attachments ? { attachments } : {}),
      ...(blocks.length && !rest ? { synthetic: true } : {}),
    });
  }

  apply(e: Event, i: number): void {
    const type = str(e.type) ?? '';
    const at = str(e.timestamp) ?? '';
    switch (type) {
      case 'user_input':
        this.pushUser(e, at);
        return;
      case 'turn_start':
        this.ensureAi(at);
        return;
      case 'stream_chunk': {
        // A token delta from the in-flight model turn. Chunks starting after a
        // code run are the NEXT model step, so the running block closes first.
        const turn = this.ensureAi(at);
        this.closeOpenCode(turn);
        turn.stream = (turn.stream ?? '') + (str(e.chunk) ?? '');
        return;
      }
      case 'stream_complete': {
        // Keep the buffer rendering: the model_response frame that follows
        // carries the settled parse and discards it. A turn that dies first
        // flushes the buffer verbatim at turn end.
        return;
      }
      case 'thought': {
        const turn = this.ensureAi(at);
        this.closeOpenCode(turn);
        this.flushStream(turn);
        this.addProse(turn, str(e.content) ?? str(e.text) ?? '');
        return;
      }
      case 'content_block': {
        // Live named content block ({name, content}) - the backend already
        // extracted it from the response, so it arrives separately from the
        // (cleaned) thought text.
        const turn = this.ensureAi(at);
        this.closeOpenCode(turn);
        this.flushStream(turn);
        const text = str(e.content) ?? str(e.text) ?? '';
        const name = str(e.name);
        if (name) this.pushContent(turn, name, text);
        else this.addProse(turn, text);
        return;
      }
      case 'model_response': {
        // The settled parse of one model turn (replay and live alike):
        // thought prose, named content panels, then post-fence tail prose.
        // The code itself renders from its code/code_execution event.
        const turn = this.ensureAi(at);
        this.closeOpenCode(turn);
        turn.stream = undefined; // superseded, never re-parsed
        this.addCacheUsage(turn, rec(e.usage));
        const thought = str(e.thought) ?? '';
        // Blocking surfaces emit the thought frame right before this one;
        // don't render the same text twice.
        const last = turn.blocks[turn.blocks.length - 1];
        const echoed = last?.kind === 'prose' && last.text.trim().endsWith(thought);
        if (thought && !echoed) this.addProse(turn, thought);
        const named = rec(e.content_blocks);
        if (named) {
          for (const [name, text] of Object.entries(named)) {
            if (typeof text === 'string') this.pushContent(turn, name, text);
          }
        }
        const tail = str(e.tail);
        if (tail) this.addProse(turn, tail);
        return;
      }
      case 'info': {
        // A mid-run message to the user - send_message() emits InfoEvent, and
        // the daemon persists it as `info`. Rendered as agent prose in the turn,
        // live and on replay alike.
        const turn = this.ensureAi(at);
        this.closeOpenCode(turn);
        this.flushStream(turn);
        this.addProse(turn, str(e.message) ?? str(e.text) ?? '');
        return;
      }
      case 'hook_status': {
        // Live tick while a hook runs ("Running precommit..."); surfaced by the
        // conversation's waiting line, not a timeline block.
        this.ensureAi(at).liveStatus = str(e.message);
        return;
      }
      case 'hook_execution': {
        // One hook's captured result (persisted, so replay shows it too):
        // an exec row named hook:{name}, failed exits open with their output.
        const name = str(e.name) ?? str(e.phase) ?? 'hook';
        const exit = num(e.exit_code) ?? 0;
        const output = [str(e.stdout), str(e.stderr)].filter(Boolean).join('\n');
        const dur = num(e.duration_ms);
        const block: ExecBlock = {
          kind: 'exec',
          command: `hook:${name}`,
          status: exit === 0 ? 'done' : 'error',
          exitCode: exit,
          ...(output ? { output } : {}),
          ...(dur != null ? { meta: formatMs(dur) } : {}),
        };
        if (!this.ai) {
          // pre_message hooks are recorded before their user_input; hold the
          // row for the turn that answers that message.
          this.pendingHooks.push(block);
          return;
        }
        this.closeOpenCode(this.ai);
        this.flushStream(this.ai);
        this.ai.blocks.push(block);
        this.ai.liveStatus = undefined;
        return;
      }
      case 'reasoning': {
        // Persisted reasoning (recorded per model turn), so thinking blocks
        // survive a reload - unlike the live-only reasoning_content below.
        this.addThink(this.ensureAi(at), str(e.content) ?? '');
        return;
      }
      case 'reasoning_content': {
        // Reasoning is a live-only signal: the daemon streams it mid-turn (always
        // after turn_start, so an AI turn is open) and, by backend design, never
        // persists it - replay carries none and nothing renders on reload. It also
        // re-emits a full recap with step:null AFTER final_result; fold reasoning
        // only into an open turn so that recap can't spawn a phantom trailing bubble.
        if (!this.ai) return;
        this.closeOpenCode(this.ai);
        this.addThink(this.ai, str(e.content) ?? '');
        return;
      }
      case 'reasoning_tokens': {
        if (!this.ai) return;
        const think = this.ai.blocks.findLast((b) => b.kind === 'think') as ThinkBlock | undefined;
        if (think) think.tokens = num(e.tokens);
        return;
      }
      case 'tool_call': {
        const turn = this.ensureAi(at);
        // Inside a running code block this is one of ITS calls (the executor's
        // wrapper streams a tool_call per invocation) - never a separate exec.
        const code = this.openCode(turn);
        if (code) {
          code.calls.push({
            tool: str(e.tool) ?? str(e.name) ?? 'tool',
            status: 'running',
            args: rec(e.arguments),
            ...(str(e.group_id) != null ? { groupId: str(e.group_id) } : {}),
          });
          return;
        }
        this.flushStream(turn);
        turn.blocks.push({
          kind: 'exec',
          command: str(e.command) ?? str(e.tool) ?? str(e.name) ?? '',
          status: 'running',
          args: rec(e.arguments),
          callId: str(e.call_id) ?? str(e.id),
        });
        return;
      }
      case 'tool_result': {
        // A tool's observed result. Folds into the exec a matching tool_call
        // opened (native path); inside a running code block it's either the
        // end-of-run observation (the combined stdout, tool None/"unknown") or
        // a per-call result.
        const turn = this.ensureAi(at);
        const output = str(e.output) ?? str(e.result) ?? str(e.stdout) ?? str(e.error);
        const exec = this.openExec(turn, str(e.call_id) ?? str(e.id));
        if (exec) {
          if (output != null) exec.output = output;
          exec.exitCode = num(e.exit_code) ?? num(e.exitCode);
          const failed = e.success === false || (exec.exitCode != null && exec.exitCode !== 0);
          exec.status = failed ? 'error' : 'done';
          return;
        }
        const code = this.openCode(turn);
        if (!code) return;
        const tool = str(e.tool) ?? str(e.name);
        const named = tool && tool !== 'unknown' ? tool : undefined;
        const running = named ? this.openCodeCall(code, named) : undefined;
        if (running) {
          running.status = e.success === false ? 'error' : 'done';
          if (output != null) running.output = output;
        } else if (named) {
          code.calls.push({
            tool: named,
            status: e.success === false ? 'error' : 'done',
            ...(output != null ? { output } : {}),
          });
        } else if (output != null) {
          code.output = output;
        }
        return;
      }
      case 'tool_result_audit': {
        // The completion half of a tool_call: success, duration, and a summary
        // but no call id. During a code run it resolves that block's running
        // call; otherwise it resolves the last open exec (without this the exec
        // block spins forever).
        const turn = this.ensureAi(at);
        const dur = num(e.duration_ms);
        const summary = str(e.summary);
        const code = this.openCode(turn);
        const call = code
          ? (this.openCodeCall(code, str(e.tool) ?? str(e.name)) ?? this.openCodeCall(code))
          : undefined;
        if (call) {
          call.status = e.success === false ? 'error' : 'done';
          if (dur != null) call.meta = formatMs(dur);
          if (summary && call.output == null) call.output = summary;
          return;
        }
        const exec = this.openExec(turn, str(e.call_id) ?? str(e.id));
        if (exec) {
          if (dur != null) exec.meta = formatMs(dur);
          if (summary && exec.output == null) exec.output = summary;
          exec.status = e.success === false ? 'error' : 'done';
        }
        return;
      }
      case 'code':
      case 'code_execution': {
        const turn = this.ensureAi(at);
        this.closeOpenCode(turn);
        this.flushStream(turn);
        // Persisted code_execution bundles the finished run (output, tool names,
        // duration); a live `code` frame opens running and collects per-call
        // tool_result observations until the model moves on.
        const returnValue = str(e.return_value_repr);
        if (returnValue) this.returnReprs.add(returnValue);
        const dur = num(e.duration_ms);
        turn.blocks.push({
          kind: 'code',
          // The executor runs python; events carry no language field of their own.
          lang: str(e.language) ?? str(e.lang) ?? 'python',
          code: str(e.code) ?? str(e.content) ?? '',
          filename: str(e.filename) ?? str(e.path),
          status: type === 'code_execution' ? 'done' : 'running',
          calls: parseToolCalls(e.tool_calls) ?? namedCalls(strArr(e.tools_called)),
          groups: parseGroups(e.groups),
          output: str(e.output) || undefined,
          returnValue,
          meta: dur != null ? formatMs(dur) : undefined,
        });
        return;
      }
      case 'job_status':
      case 'job_update': {
        const turn = this.ensureAi(at);
        const raw = (e.data as Event) ?? e;
        // Narrow the wire dict to the shared JobLike shape at this one boundary, so
        // the tile renders typed fields instead of stringly-casting downstream.
        const job: JobLike = {
          job_id: str(raw.job_id) ?? str(e.job_id),
          state: str(raw.state),
          agent: str(raw.agent),
          prompt: str(raw.prompt),
          verify_attempts: num(raw.verify_attempts),
          max_attempts: num(raw.max_attempts),
        };
        // One tile per job, not per status event: fold repeat events into the
        // existing tile so a job's lifecycle updates in place.
        const existing = job.job_id
          ? (turn.blocks.find((b) => b.kind === 'job' && b.job.job_id === job.job_id) as
              JobBlock | undefined)
          : undefined;
        if (existing) existing.job = job;
        else turn.blocks.push({ kind: 'job', job });
        return;
      }
      case 'delivery': {
        // Deliberately never assigned to `this.ai`: the next model frame must open
        // its own bubble rather than append to a row that was never a model turn.
        let turn = this.ai;
        if (!turn) {
          turn = { id: this.uid('ai'), role: 'ai', at, blocks: [] };
          this.turns.push(turn);
        }
        turn.blocks.push({
          kind: 'delivery',
          message: str(e.message) ?? '',
          source: str(e.source) ?? '',
          needsAck: str(e.kind) === 'needs_ack',
          ...(str(e.delivery_id) ? { deliveryId: str(e.delivery_id) } : {}),
          ...(str(e.title) ? { title: str(e.title) } : {}),
        });
        return;
      }
      // Group frames only mean anything inside a running code block. Never
      // ensureAi() here: a stray frame after the turn closed would spawn an
      // empty AI bubble, the same hazard reasoning_content guards against.
      case 'group_start': {
        const code = this.ai && this.openCode(this.ai);
        const id = str(e.group_id);
        if (!code || !id) return;
        const groups = (code.groups ??= []);
        // A broadcast echo can redeliver a frame the per-chat stream already
        // applied, and a duplicate key throws inside the {#each}.
        if (groups.some((g) => g.id === id)) return;
        groups.push({ id, title: str(e.title) ?? '' });
        return;
      }
      case 'group_end': {
        const code = this.ai && this.openCode(this.ai);
        const group = code?.groups?.find((g) => g.id === str(e.group_id));
        if (!group) return;
        // Absent success stays unknown, matching what replay reads off the record.
        if (typeof e.success === 'boolean') group.success = e.success;
        const dur = num(e.duration_ms);
        if (dur != null) group.meta = formatMs(dur);
        if (str(e.error) != null) group.error = str(e.error);
        return;
      }
      case 'final_result': {
        const turn = this.ensureAi(at);
        this.flushStream(turn);
        // A dict/object final answer arrives with `result` as its Python repr (a
        // single-quoted blob); prose-render a human string field off the real
        // `result_data` object instead. The raw object still renders as the
        // collapsed ResultBlock below, so nothing is lost.
        const data = rec(e.result_data);
        const human = data
          ? (str(data.message) ?? str(data.text) ?? str(data.summary) ?? str(data.answer))
          : undefined;
        // The final result is the reply of record - render it unless it merely
        // echoes prose already shown or a return_value repr on a code block
        // (commentary from earlier turns must NOT suppress it).
        const result = human ?? str(e.result) ?? '';
        if (result && !this.returnReprs.has(result) && !this.hasProseEchoing(turn, result)) {
          this.addProse(turn, result);
        }
        if (e.result_data != null) turn.blocks.push({ kind: 'result', data: e.result_data });
        // Spread first so the cache split summed from model_response events
        // survives; final_result owns tokens/cost/turns.
        turn.meta = { ...turn.meta, tokens: num(e.tokens), cost: num(e.cost), turns: num(e.turns) };
        this.resolveOpenExecs(turn);
        turn.streaming = false;
        this.ai = null;
        return;
      }
      case 'error': {
        // A provider/turn error, rendered ONCE as the styled error block. The same
        // failure also arrives as content (a `[Error: …]` model_response) and, live,
        // as a stream_chunk - both would double it. Fold into the turn that ran
        // (reuse the open one, else the AI row a preceding session_end already
        // closed) rather than opening a second empty AI row for the error alone.
        const message = str(e.error) ?? str(e.message) ?? 'The turn failed.';
        const turn = this.ai ?? this.tailAi() ?? this.ensureAi(at);
        turn.stream = undefined; // the buffered chunk is the same failure text
        this.dropErrorEcho(turn, message);
        if (!turn.blocks.some((b) => b.kind === 'error' && b.message === message)) {
          turn.blocks.push({ kind: 'error', message });
        }
        this.resolveOpenExecs(turn);
        turn.streaming = false;
        this.ai = null;
        return;
      }
      case 'resume_reset': {
        // The backend severed this chat's unresumable provider session and
        // continued the turn from saved history. A calm notice at the head of the
        // turn it healed - informational, never a failure. The UI dict lifts event
        // data to the top level, so read `message` there (nested `data` and a
        // default cover any other shape).
        const message =
          str(e.message) ??
          str(rec(e.data)?.message) ??
          "The chat's model session was reset; continuing from saved history.";
        const turn = this.ensureAi(at);
        if (!turn.blocks.some((b) => b.kind === 'notice' && b.message === message)) {
          turn.blocks.push({ kind: 'notice', message });
        }
        return;
      }
      case 'session_end':
      case 'session_complete':
      case 'session_error':
      case 'session_cancelled':
        if (this.ai) {
          this.resolveOpenExecs(this.ai);
          this.ai.streaming = false;
        }
        this.ai = null;
        return;
      case 'session_info': {
        const tokens = num(e.tokens);
        const limit = num(e.context_limit);
        if (tokens != null && limit != null) this.context = { tokens, limit };
        return;
      }
      case 'compaction':
      case 'compaction_started':
      case 'compacted': {
        this.compactions.push({
          id: key(e, i),
          at,
          replacedCount: num(e.replaced_count),
          retainedCount: num(e.retained_count),
          sourceId: str(e.source_session_id) ?? str(e.successor_id),
          summary: str(e.summary),
        });
        return;
      }
      case 'ask_user': {
        // Not a timeline block: an interactive prompt held to one side. A durable
        // ask_user with no later ask_answered (below) is a still-open prompt the
        // controller re-shows on reload.
        this.pendingAsk = {
          ...(str(e.ask_id) ? { askId: str(e.ask_id) } : {}),
          question: str(e.question) ?? '',
          questionType: (str(e.question_type) as PendingAsk['questionType']) ?? 'text',
          options: strArr(e.options) ?? [],
        };
        return;
      }
      case 'ask_answered': {
        // Resolves the open prompt (answered, cancelled, or timed out). A missing
        // id on either side is treated as a match, for a backend that sends none.
        const id = str(e.ask_id);
        if (
          this.pendingAsk &&
          (id == null || this.pendingAsk.askId == null || this.pendingAsk.askId === id)
        ) {
          this.pendingAsk = null;
        }
        return;
      }
      default:
        // prompt_snapshot / model_request / init / llm_wait_progress
        // / warning / file_read: no timeline block.
        return;
    }
  }
}

export function buildTimeline(events: Event[]): Timeline {
  const b = new Builder();
  events.forEach((e, i) => b.apply(e, i));
  b.finish();
  return {
    turns: b.turns,
    compactions: b.compactions,
    context: b.context,
    pendingAsk: b.pendingAsk,
  };
}
