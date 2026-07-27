import { describe, expect, it } from 'vitest';
import {
  buildTimeline,
  splitStreamFence,
  type ProseBlock,
  type ExecBlock,
  type ResultBlock,
  type CodeBlock,
  type ErrorBlock,
  type JobBlock,
  type NoticeBlock,
} from './turns';

// Shapes copied from a live daemon capture (GET /api/sessions/{id}/events).
// model_response events arrive normalized: the backend parses raw_content once
// (at record time, or backfilled on read) and serves thought/content_blocks/tail.
const replay = [
  { type: 'user_input', text: 'say hello', timestamp: '2026-07-14T15:00:00Z', id: 2 },
  { type: 'prompt_snapshot', token_breakdown: {}, id: 3 },
  { type: 'model_request', turn: 0, id: 4 },
  {
    type: 'model_response',
    turn: 0,
    raw_content: 'Hello there.',
    thought: 'Hello there.',
    cost: 0.0007,
    id: 5,
  },
  { type: 'final_result', result: 'Hello there.', turns: 1, tokens: 972, cost: 0.0007, id: 6 },
  { type: 'session_end', status: 'success', id: 7 },
];

describe('buildTimeline (replay shape)', () => {
  it('pairs a user turn with an AI turn and does not duplicate model_response into final_result', () => {
    const t = buildTimeline(replay);
    expect(t.turns.map((x) => x.role)).toEqual(['user', 'ai']);
    const user = t.turns[0]!;
    const ai = t.turns[1]!;
    expect((user.blocks[0] as ProseBlock).text).toBe('say hello');
    // model_response text renders once; final_result only contributes meta.
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses).toHaveLength(1);
    expect(proses[0]!.text).toBe('Hello there.');
    expect(ai.meta).toEqual({ tokens: 972, cost: 0.0007, turns: 1 });
    expect(ai.streaming).toBeFalsy();
  });

  it('captures context from session_info', () => {
    const t = buildTimeline([
      ...replay,
      { type: 'session_info', tokens: 966, context_limit: 128000, id: 8 },
    ]);
    expect(t.context).toEqual({ tokens: 966, limit: 128000 });
  });
});

describe('buildTimeline (live shape)', () => {
  it('renders thought as the AI prose and merges consecutive thoughts', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'hi', timestamp: 'z', id: 1 },
      { type: 'turn_start', turn: 1, id: 2 },
      { type: 'thought', content: 'First.', id: 3 },
      { type: 'thought', content: 'Second.', id: 4 },
    ]);
    const ai = t.turns[1]!;
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses).toHaveLength(1);
    expect(proses[0]!.text).toBe('First.\n\nSecond.');
    // No final_result yet -> the AI turn is still streaming.
    expect(ai.streaming).toBe(true);
  });

  it('folds tool_call + tool_result into one exec block matched by id', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'grep', id: 1 },
      { type: 'tool_call', tool: 'run', command: 'rg -n EventSource src/', call_id: 'c1', id: 2 },
      { type: 'tool_result', call_id: 'c1', output: 'src/sse.ts:14', exit_code: 0, id: 3 },
      { type: 'final_result', result: 'done', id: 4 },
    ]);
    const ai = t.turns[1]!;
    const exec = ai.blocks.find((b) => b.kind === 'exec') as ExecBlock;
    expect(exec.command).toBe('rg -n EventSource src/');
    expect(exec.output).toBe('src/sse.ts:14');
    expect(exec.status).toBe('done');
    expect(exec.exitCode).toBe(0);
  });

  it('renders a non-null result_data as a return_value block', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'thought', content: 'here', id: 2 },
      { type: 'final_result', result: 'here', result_data: { ok: true }, id: 3 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.blocks.some((b) => b.kind === 'result')).toBe(true);
    const res = ai.blocks.find((b) => b.kind === 'result') as ResultBlock;
    expect(res.data).toEqual({ ok: true });
    // The prose came from thought, so result text isn't duplicated as prose.
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses).toHaveLength(1);
  });

  it('uses final_result.result as prose when the turn produced no other text', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'final_result', result: 'Only answer.', id: 2 },
    ]);
    const ai = t.turns[1]!;
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['Only answer.']);
  });
});

describe('buildTimeline (misc)', () => {
  it('collects compaction events into the compactions list', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'final_result', result: 'a', id: 2 },
      {
        type: 'compaction',
        replaced_count: 40,
        retained_count: 6,
        source_session_id: 'old-1',
        timestamp: 'z',
        id: 3,
      },
    ]);
    expect(t.compactions).toHaveLength(1);
    expect(t.compactions[0]!.replacedCount).toBe(40);
    expect(t.compactions[0]!.retainedCount).toBe(6);
    expect(t.compactions[0]!.sourceId).toBe('old-1');
  });

  it('appends a job_status event as a job block in the timeline', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'spawn', id: 1 },
      { type: 'thought', content: 'spawning', id: 2 },
      { type: 'job_status', job_id: 'job-abc', state: 'running', prompt: 'do it', id: 3 },
    ]);
    const ai = t.turns[1]!;
    const job = ai.blocks.find((b) => b.kind === 'job');
    expect(job).toBeTruthy();
  });

  it('ignores empty input and returns an empty timeline', () => {
    expect(buildTimeline([])).toEqual({
      turns: [],
      compactions: [],
      context: undefined,
      pendingAsk: null,
    });
  });
});

describe('buildTimeline (pending ask reconciliation)', () => {
  it('derives a pending ask from a durable ask_user with no answer (survives reload)', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'summarize https://example.com', id: 1 },
      {
        type: 'ask_user',
        ask_id: 'ask-abc',
        question: 'Fetch content from example.com?',
        question_type: 'approval',
        options: ['Approve', 'Deny'],
        id: 2,
      },
    ]);
    expect(t.pendingAsk).toEqual({
      askId: 'ask-abc',
      question: 'Fetch content from example.com?',
      questionType: 'approval',
      options: ['Approve', 'Deny'],
    });
    // The ask_user is not a timeline block: only the user turn renders.
    expect(t.turns.map((x) => x.role)).toEqual(['user']);
  });

  it('clears the pending ask once the matching ask_answered lands (no re-prompt)', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'ship it', id: 1 },
      { type: 'ask_user', ask_id: 'ask-1', question: 'Push?', question_type: 'yes_no', id: 2 },
      { type: 'ask_answered', ask_id: 'ask-1', answer: 'yes', id: 3 },
      { type: 'final_result', result: 'done', id: 4 },
    ]);
    expect(t.pendingAsk).toBeNull();
  });

  it('leaves the ask pending when an ask_answered targets a different ask id', () => {
    const t = buildTimeline([
      { type: 'ask_user', ask_id: 'ask-2', question: 'Pick', question_type: 'choice', id: 1 },
      { type: 'ask_answered', ask_id: 'ask-other', answer: 'x', id: 2 },
    ]);
    expect(t.pendingAsk?.askId).toBe('ask-2');
  });
});

describe('buildTimeline (reasoning is live-only)', () => {
  it('folds a post-final reasoning recap away instead of spawning a phantom turn', () => {
    // The daemon re-emits the whole reasoning with step:null AFTER final_result.
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'turn_start', turn: 1, id: 2 },
      { type: 'reasoning_content', content: 'weighing it', step: 1, id: 3 },
      { type: 'final_result', result: 'answer', id: 4 },
      { type: 'reasoning_content', content: 'weighing it', step: null, id: 5 },
    ]);
    expect(t.turns.map((x) => x.role)).toEqual(['user', 'ai']);
    const ai = t.turns[1]!;
    // Exactly one think block (the live one), and the turn is closed, not streaming.
    expect(ai.blocks.filter((b) => b.kind === 'think')).toHaveLength(1);
    expect(ai.streaming).toBeFalsy();
  });

  it('renders no think blocks when replay history carries no reasoning events', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'model_response', raw_content: 'plain answer', thought: 'plain answer', id: 2 },
      { type: 'final_result', result: 'plain answer', id: 3 },
      { type: 'session_end', status: 'success', id: 4 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.blocks.some((b) => b.kind === 'think')).toBe(false);
  });
});

describe('buildTimeline (code execution dedupe + output)', () => {
  const codeExecReplay = [
    { type: 'user_input', text: 'compute', id: 1 },
    {
      type: 'code_execution',
      code: 'return_value({"a": 1})',
      output: 'ran',
      error: null,
      tools_called: ['read_note'],
      last_statement_type: 'expr',
      return_value_repr: "{'a': 1}",
      return_value_type: 'dict',
      id: 2,
    },
    {
      type: 'model_response',
      raw_content: 'Here you go:\n\n```python-exec\nreturn_value({"a": 1})\n```',
      thought: 'Here you go:',
      id: 3,
    },
    { type: 'final_result', result: "{'a': 1}", turns: 1, id: 4 },
    { type: 'session_end', status: 'success', id: 5 },
  ];

  it('does not duplicate the code_execution block into model_response prose', () => {
    const t = buildTimeline(codeExecReplay);
    const ai = t.turns[1]!;
    // Exactly one code block; raw_content (with its fence) is never rendered -
    // only the backend-parsed thought is.
    expect(ai.blocks.filter((b) => b.kind === 'code')).toHaveLength(1);
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.every((p) => !p.text.includes('python-exec'))).toBe(true);
    // The surrounding commentary survives.
    expect(proses.some((p) => p.text.includes('Here you go:'))).toBe(true);
  });

  it('carries the replayed output, per-tool calls, and return value on the code block', () => {
    const t = buildTimeline(codeExecReplay);
    const code = t.turns[1]!.blocks.find((b) => b.kind === 'code') as CodeBlock;
    expect(code.output).toBe('ran');
    expect(code.status).toBe('done');
    expect(code.calls).toEqual([{ tool: 'read_note', status: 'done' }]);
    expect(code.returnValue).toBe("{'a': 1}");
  });

  it('live: tool_call/tool_result_audit pairs during a code run fold into the code block', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'code', content: 'run("ls")\nread_note("x")', id: 3 },
      { type: 'tool_call', tool: 'run', arguments: { cmd: 'ls' }, id: 4 },
      {
        type: 'tool_result_audit',
        tool: 'run',
        success: true,
        duration_ms: 120,
        summary: 'file-a',
        id: 5,
      },
      { type: 'tool_call', tool: 'read_note', arguments: { path: 'x' }, id: 6 },
      {
        type: 'tool_result_audit',
        tool: 'read_note',
        success: false,
        duration_ms: 8,
        summary: 'no such note',
        id: 7,
      },
      { type: 'tool_result', tool: 'unknown', success: true, output: 'combined stdout', id: 8 },
    ]);
    const blocks = t.turns[1]!.blocks;
    // No top-level exec blocks: the calls belong to the code block.
    expect(blocks.filter((b) => b.kind === 'exec')).toHaveLength(0);
    const code = blocks.find((b) => b.kind === 'code') as CodeBlock;
    expect(code.status).toBe('running');
    expect(code.calls).toEqual([
      { tool: 'run', status: 'done', args: { cmd: 'ls' }, meta: '120ms', output: 'file-a' },
      {
        tool: 'read_note',
        status: 'error',
        args: { path: 'x' },
        meta: '8ms',
        output: 'no such note',
      },
    ]);
    // The end-of-block observation is the combined output, not another call.
    expect(code.output).toBe('combined stdout');
  });

  it('live: a named tool_result without a tool_call still lands as a completed call', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'code', content: 'run("ls")', id: 2 },
      { type: 'tool_result', tool: 'run', success: false, error: 'boom', id: 3 },
    ]);
    const code = t.turns[1]!.blocks.find((b) => b.kind === 'code') as CodeBlock;
    expect(code.calls).toEqual([{ tool: 'run', status: 'error', output: 'boom' }]);
  });

  it('a call still running when the turn dies resolves to ended, never a spinner', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'code', content: 'run("sleep")', id: 2 },
      { type: 'tool_call', tool: 'run', arguments: {}, id: 3 },
      { type: 'session_end', status: 'error', id: 4 },
    ]);
    const code = t.turns[1]!.blocks.find((b) => b.kind === 'code') as CodeBlock;
    expect(code.status).toBe('done');
    expect(code.calls[0]!.status).toBe('ended');
  });

  it('replay: persisted tool_calls records carry args, output, and duration per call', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      {
        type: 'code_execution',
        code: 'fetch(url="https://x.test")',
        output: 'combined',
        tools_called: ['fetch'],
        tool_calls: [
          {
            tool: 'fetch',
            arguments: { url: 'https://x.test' },
            success: true,
            duration_ms: 412,
            output: 'token=*** for https://x.test',
          },
        ],
        id: 2,
      },
    ]);
    const code = t.turns[1]!.blocks.find((b) => b.kind === 'code') as CodeBlock;
    expect(code.calls).toEqual([
      {
        tool: 'fetch',
        status: 'done',
        args: { url: 'https://x.test' },
        meta: '412ms',
        output: 'token=*** for https://x.test',
      },
    ]);
  });

  it('live: the running code block closes when the model moves on, and at turn end', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'code', content: 'run("a")', id: 2 },
      { type: 'tool_result', tool: 'run', success: true, output: 'ok', id: 3 },
      { type: 'thought', content: 'now the second step', id: 4 },
      { type: 'code', content: 'run("b")', id: 5 },
      { type: 'final_result', result: 'done', id: 6 },
    ]);
    const codes = t.turns[1]!.blocks.filter((b) => b.kind === 'code') as CodeBlock[];
    expect(codes).toHaveLength(2);
    expect(codes[0]!.status).toBe('done');
    expect(codes[1]!.status).toBe('done');
  });

  it('live: a tool_result still folds into a native exec before falling back to code calls', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'tool_call', tool: 'session_metadata', arguments: { key: 'topic' }, id: 2 },
      { type: 'tool_result', tool: 'session_metadata', success: true, output: 'saved', id: 3 },
    ]);
    const blocks = t.turns[1]!.blocks;
    expect(blocks.filter((b) => b.kind === 'code')).toHaveLength(0);
    const exec = blocks.find((b) => b.kind === 'exec');
    expect(exec).toMatchObject({ status: 'done', output: 'saved' });
  });

  it('renders the final_result once when it just echoes a return_value already shown', () => {
    const t = buildTimeline(codeExecReplay);
    const ai = t.turns[1]!;
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    // The return value repr must not also appear as a trailing prose block.
    expect(proses.some((p) => p.text === "{'a': 1}")).toBe(false);
  });

  it('does not dedupe a later turn against an earlier turn return value', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'one', id: 1 },
      { type: 'code_execution', code: 'return_value(7)', return_value_repr: '7', id: 2 },
      { type: 'final_result', result: '7', id: 3 },
      // A distinct exchange whose plain-text answer coincidentally equals '7'.
      { type: 'user_input', text: 'two', id: 4 },
      { type: 'final_result', result: '7', id: 5 },
    ]);
    const second = t.turns[3]!;
    const proses = second.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['7']);
  });
});

describe('buildTimeline (error frames)', () => {
  it('renders an error block and closes the streaming turn on an error frame', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', turn: 1, id: 2 },
      { type: 'thought', content: 'working', id: 3 },
      { type: 'error', error: 'provider exploded', id: 4 },
    ]);
    const ai = t.turns[1]!;
    const err = ai.blocks.find((b) => b.kind === 'error') as ErrorBlock;
    expect(err).toBeTruthy();
    expect(err.message).toBe('provider exploded');
    // No perpetual aria-busy: the turn is no longer streaming.
    expect(ai.streaming).toBeFalsy();
  });
});

describe('buildTimeline (failed turn renders one error, never a double)', () => {
  // Captured from the live :18461 repro (a bogus model set via /model, then a
  // send that fails at the provider). The failure text surfaces three ways: the
  // model_response carries it as `[Error: …]` content, a live stream_chunk carries
  // it raw, and the `error` frame carries the canonical message. Left alone the
  // reducer rendered it twice - as prose AND as the styled error block, in two
  // separate AI rows on replay (session_end clears the open turn before the error).
  const MODEL_ERR =
    "There's an issue with the selected model (bogus-model-xyz-999). It may not exist" +
    ' or you may not have access to it. Run --model to pick a different model.';
  const ERR_FRAME = `${MODEL_ERR} (subtype=success)`;
  const ERR_CONTENT = `[Error: ${ERR_FRAME}]`;

  const aiTurns = (t: ReturnType<typeof buildTimeline>) => t.turns.filter((x) => x.role === 'ai');
  const errorBlocks = (t: ReturnType<typeof buildTimeline>) =>
    t.turns.flatMap((x) => x.blocks).filter((b) => b.kind === 'error');
  const proseBlocks = (t: ReturnType<typeof buildTimeline>) =>
    t.turns.flatMap((x) => x.blocks).filter((b) => b.kind === 'prose') as ProseBlock[];

  it('replay: folds the error into the closed turn, dropping the raw [Error:] prose echo', () => {
    // The persisted shape (GET /events): model_response content echo, session_end,
    // then a trailing error frame.
    const t = buildTimeline([
      { type: 'session_start', id: 557 },
      { type: 'user_input', text: 'hello', id: 558 },
      { type: 'prompt_snapshot', token_breakdown: {}, id: 559 },
      { type: 'model_request', turn: 0, id: 560 },
      { type: 'model_response', raw_content: ERR_CONTENT, thought: ERR_CONTENT, id: 562 },
      { type: 'session_end', status: 'success', id: 563 },
      { type: 'error', error: ERR_FRAME, id: 564 },
    ]);
    // Exactly one AI turn holds the failure - the error frame does not spawn a
    // second tsugite row after session_end closed the first.
    expect(aiTurns(t)).toHaveLength(1);
    // One styled error block, carrying the canonical frame message.
    const errs = errorBlocks(t) as ErrorBlock[];
    expect(errs).toHaveLength(1);
    expect(errs[0]!.message).toBe(ERR_FRAME);
    // The `[Error: …]` prose echo is gone - the failure renders once.
    expect(proseBlocks(t).some((b) => b.text.includes('[Error:'))).toBe(false);
    expect(proseBlocks(t).some((b) => b.text.includes(MODEL_ERR))).toBe(false);
  });

  it('live: discards the buffered stream chunk so the error frame is the only render', () => {
    // The live per-chat stream: turn_start, the error as a stream_chunk, then the
    // error frame. The buffered chunk must not flush to a duplicate prose block.
    const t = buildTimeline([
      { type: 'turn_start', turn: 1 },
      { type: 'prompt_snapshot', token_breakdown: {} },
      { type: 'stream_chunk', chunk: MODEL_ERR },
      { type: 'error', error: ERR_FRAME },
    ]);
    expect(aiTurns(t)).toHaveLength(1);
    expect(errorBlocks(t)).toHaveLength(1);
    expect(proseBlocks(t)).toHaveLength(0);
    expect(aiTurns(t)[0]!.streaming).toBeFalsy();
  });

  it('keeps legitimate prose the model produced before it errored', () => {
    // A turn that said something real, then hit an error: the real prose survives,
    // only the error itself renders once (not doubled).
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', turn: 1, id: 2 },
      { type: 'thought', content: 'Let me try that.', id: 3 },
      { type: 'error', error: 'provider exploded', id: 4 },
    ]);
    expect(aiTurns(t)).toHaveLength(1);
    expect(errorBlocks(t)).toHaveLength(1);
    const aiProse = aiTurns(t)[0]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(aiProse.map((b) => b.text)).toEqual(['Let me try that.']);
  });
});

describe('buildTimeline (resume_reset notice)', () => {
  // The backend severs an unresumable provider session and continues from saved
  // history, recording a durable resume_reset. The UI dict flattens event data,
  // so `message` arrives at the top level (replay and live alike).
  const RESET_MSG =
    "The chat's resumable model session was reset because it could no longer be " +
    'resumed; continuing from saved history.';

  const notices = (t: ReturnType<typeof buildTimeline>) =>
    t.turns.flatMap((x) => x.blocks).filter((b) => b.kind === 'notice') as NoticeBlock[];

  it('replay: renders the notice at the head of the healed turn, never an error', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'continue', id: 1 },
      { type: 'resume_reset', reason: 'poisoned_transcript', message: RESET_MSG, id: 2 },
      { type: 'model_response', raw_content: 'Back on track.', thought: 'Back on track.', id: 3 },
      { type: 'final_result', result: 'Back on track.', id: 4 },
      { type: 'session_end', status: 'success', id: 5 },
    ]);
    expect(t.turns.map((x) => x.role)).toEqual(['user', 'ai']);
    const ai = t.turns[1]!;
    // A calm notice, not an error block - and it leads the AI turn.
    const ns = notices(t);
    expect(ns).toHaveLength(1);
    expect(ns[0]!.message).toBe(RESET_MSG);
    expect(ai.blocks[0]!.kind).toBe('notice');
    expect(ai.blocks.some((b) => b.kind === 'error')).toBe(false);
    // The reset healed the turn: it closes normally, not stuck streaming.
    expect(ai.streaming).toBeFalsy();
  });

  it('live: folds the emitted reset frame into the in-flight turn as a notice', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go on', id: 1 },
      { type: 'resume_reset', reason: 'poisoned_transcript', message: RESET_MSG },
      { type: 'thought', content: 'resuming fresh' },
    ]);
    const ai = t.turns[1]!;
    expect(ai.blocks[0]!.kind).toBe('notice');
    expect((ai.blocks[0] as NoticeBlock).message).toBe(RESET_MSG);
    // The retry is still streaming - the notice must not close the turn.
    expect(ai.streaming).toBe(true);
  });

  it('falls back to a default message when the reset frame carries none', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'resume_reset', reason: 'poisoned_transcript', id: 2 },
      { type: 'final_result', result: 'ok', id: 3 },
    ]);
    const ns = notices(t);
    expect(ns).toHaveLength(1);
    expect(ns[0]!.message).toContain('reset');
  });
});

describe('buildTimeline (native tool calls: args + audit completion)', () => {
  // Captured end-to-end from GET /api/sessions/{id}/events on a live daemon
  // (event_to_ui_dict flattens data → top level). Native tools complete via
  // `tool_result_audit`, NOT `tool_result`; the audit carries success/duration/summary.
  const nativeReplay = [
    { type: 'user_input', text: 'check the ledger', timestamp: 't0', id: 13 },
    { type: 'turn_start', turn: 0, id: 14 },
    {
      type: 'tool_call',
      tool: 'session_metadata',
      arguments: { key: 'topic', value: 'ledger totals' },
      step: 0,
      id: 15,
    },
    {
      type: 'tool_result_audit',
      tool: 'session_metadata',
      success: true,
      duration_ms: 12,
      summary: 'metadata updated',
      step: 0,
      id: 16,
    },
    {
      type: 'tool_call',
      tool: 'read_file',
      arguments: { path: '/nope/missing.txt' },
      step: 1,
      id: 18,
    },
    {
      type: 'tool_result_audit',
      tool: 'read_file',
      success: false,
      duration_ms: 3,
      summary: 'file not found',
      step: 1,
      id: 19,
    },
    {
      type: 'model_response',
      raw_content: 'Totals computed.',
      thought: 'Totals computed.',
      id: 20,
    },
    { type: 'final_result', result: 'Totals computed.', turns: 2, tokens: 1440, id: 21 },
    { type: 'session_end', status: 'success', id: 22 },
  ];

  it('resolves a native tool_call to done via tool_result_audit (no permanent spinner)', () => {
    const ai = buildTimeline(nativeReplay).turns[1]!;
    const execs = ai.blocks.filter((b) => b.kind === 'exec') as ExecBlock[];
    // The load-bearing regression: nothing is left spinning once the turn ends.
    expect(execs.some((e) => e.status === 'running')).toBe(false);
    const meta = execs[0]!;
    expect(meta.command).toBe('session_metadata');
    expect(meta.status).toBe('done');
    // Args are captured for the detail view, and the audit's duration + summary land.
    expect(meta.args).toEqual({ key: 'topic', value: 'ledger totals' });
    expect(meta.meta).toBe('12ms');
    expect(meta.output).toBe('metadata updated');
  });

  it('marks a failed native tool (success:false audit) as error and keeps its args', () => {
    const ai = buildTimeline(nativeReplay).turns[1]!;
    const read = (ai.blocks.filter((b) => b.kind === 'exec') as ExecBlock[])[1]!;
    expect(read.command).toBe('read_file');
    expect(read.status).toBe('error');
    expect(read.args).toEqual({ path: '/nope/missing.txt' });
  });

  it('folds a live observation tool_result (output + success) into the open exec', () => {
    const ai = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'tool_call', tool: 'run', arguments: { command: 'ls' }, id: 2 },
      { type: 'tool_result', tool: 'run', success: true, output: 'a\nb', id: 3 },
      { type: 'final_result', result: 'done', id: 4 },
    ]).turns[1]!;
    const exec = ai.blocks.find((b) => b.kind === 'exec') as ExecBlock;
    expect(exec.status).toBe('done');
    expect(exec.output).toBe('a\nb');
  });

  it('resolves an exec left open at turn end (final_result) to a closed-neutral state', () => {
    const ai = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'tool_call', tool: 'slow_tool', arguments: {}, id: 2 },
      { type: 'final_result', result: 'partial', id: 3 },
    ]).turns[1]!;
    const exec = ai.blocks.find((b) => b.kind === 'exec') as ExecBlock;
    expect(exec.status).not.toBe('running');
    expect(exec.status).toBe('ended');
    expect(ai.streaming).toBeFalsy();
  });

  it('resolves an open exec when the turn closes via session_end (no output → ended)', () => {
    const ai = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', turn: 0, id: 2 },
      { type: 'tool_call', tool: 'x', arguments: {}, id: 3 },
      { type: 'session_end', status: 'cancelled', id: 4 },
    ]).turns[1]!;
    const exec = ai.blocks.find((b) => b.kind === 'exec') as ExecBlock;
    expect(exec.status).toBe('ended');
  });
});

describe('buildTimeline (job tiles)', () => {
  it('upserts a job tile in place instead of one tile per status event', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'spawn', id: 1 },
      { type: 'job_status', job_id: 'job-1', state: 'queued', prompt: 'do it', id: 2 },
      { type: 'job_status', job_id: 'job-1', state: 'running', prompt: 'do it', id: 3 },
      { type: 'job_status', job_id: 'job-1', state: 'done', prompt: 'do it', id: 4 },
    ]);
    const jobs = t.turns[1]!.blocks.filter((b) => b.kind === 'job') as JobBlock[];
    expect(jobs).toHaveLength(1);
    expect(jobs[0]!.job.state).toBe('done');
  });

  it('keeps distinct jobs as separate tiles', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'spawn', id: 1 },
      { type: 'job_status', job_id: 'job-1', state: 'running', id: 2 },
      { type: 'job_status', job_id: 'job-2', state: 'running', id: 3 },
    ]);
    expect(t.turns[1]!.blocks.filter((b) => b.kind === 'job')).toHaveLength(2);
  });
});

describe('buildTimeline (dict final answer)', () => {
  it('prose-renders a human field from a dict result_data, not the repr blob', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'spawn', id: 1 },
      {
        type: 'final_result',
        // The repr blob the model returned - must NOT appear as prose.
        result: "{'status': 'started', 'message': 'Review session launched.'}",
        result_data: { status: 'started', message: 'Review session launched.' },
        id: 2,
      },
    ]);
    const ai = t.turns[1]!;
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['Review session launched.']);
    expect(proses.some((p) => p.text.includes("{'status'"))).toBe(false);
    // The raw object still renders as the collapsed result affordance.
    const res = ai.blocks.find((b) => b.kind === 'result') as ResultBlock;
    expect(res.data).toEqual({ status: 'started', message: 'Review session launched.' });
  });

  it('falls back to result_data.text / summary / answer in order', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      {
        type: 'final_result',
        result: "{'answer': 'forty-two'}",
        result_data: { answer: 'forty-two' },
        id: 2,
      },
    ]);
    const proses = t.turns[1]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['forty-two']);
  });

  it('keeps the raw result when result_data carries no human field', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'final_result', result: 'plain answer', result_data: { count: 3 }, id: 2 },
    ]);
    const proses = t.turns[1]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['plain answer']);
  });

  it('leaves a plain-string final_result unchanged', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'final_result', result: 'Only answer.', id: 2 },
    ]);
    const proses = t.turns[1]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['Only answer.']);
  });
});

describe('buildTimeline (token streaming)', () => {
  it('accumulates stream_chunk deltas on the open AI turn', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'stream_chunk', chunk: 'Let me ', id: 3 },
      { type: 'stream_chunk', chunk: 'look.', id: 4 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.stream).toBe('Let me look.');
    expect(ai.streaming).toBe(true);
  });

  it('the settled model_response frame supersedes the raw stream buffer', () => {
    // Live order per model turn: chunks -> stream_complete -> model_response
    // (the backend's parse) -> code. The buffer is discarded, never re-parsed.
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'stream_chunk', chunk: 'Checking.\n\n```python-exec\nrun("ls")\n```', id: 2 },
      { type: 'stream_complete', id: 3 },
      { type: 'model_response', thought: 'Checking.', id: 4 },
      { type: 'code', content: 'run("ls")', id: 5 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.stream).toBeUndefined();
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['Checking.']);
    // The fence renders once, as the code block.
    expect(ai.blocks.filter((b) => b.kind === 'code')).toHaveLength(1);
  });

  it('keeps rendering the buffer between stream_complete and the settled frame', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'stream_chunk', chunk: 'Almost there.', id: 2 },
      { type: 'stream_complete', id: 3 },
    ]);
    expect(t.turns[1]!.stream).toBe('Almost there.');
  });

  it('live blocking turns do not double the thought (thought frame + model_response frame)', () => {
    // Non-streaming surfaces emit LLMMessage (thought) AND the settled frame.
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'thought', content: 'One answer.', id: 3 },
      { type: 'model_response', thought: 'One answer.', id: 4 },
    ]);
    const proses = t.turns[1]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['One answer.']);
  });

  it('a final_result echoing the streamed text is not duplicated', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'stream_chunk', chunk: 'The answer is 4.', id: 2 },
      { type: 'stream_complete', id: 3 },
      { type: 'final_result', result: 'The answer is 4.', id: 4 },
    ]);
    const proses = t.turns[1]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['The answer is 4.']);
  });

  it('a turn that dies mid-stream still flushes the buffer at turn end', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'stream_chunk', chunk: 'partial thou', id: 2 },
      { type: 'session_end', status: 'error', id: 3 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.stream).toBeUndefined();
    const proses = ai.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['partial thou']);
  });

  it('chunks from the next model turn close the previous running code block', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'code', content: 'run("a")', id: 2 },
      { type: 'stream_chunk', chunk: 'Now step two.', id: 3 },
    ]);
    const code = t.turns[1]!.blocks.find((b) => b.kind === 'code') as CodeBlock;
    expect(code.status).toBe('done');
  });
});

describe('buildTimeline (final_result visibility)', () => {
  it('renders the final result even when earlier turns produced commentary prose', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'stream_chunk', chunk: 'Let me compute that.', id: 2 },
      { type: 'stream_complete', id: 3 },
      { type: 'code', content: 'return_value(6 * 7)', id: 4 },
      { type: 'final_result', result: '42', id: 5 },
    ]);
    const proses = t.turns[1]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['Let me compute that.', '42']);
  });

  it('still skips a final_result that exactly echoes the settled prose', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'stream_chunk', chunk: 'The answer is 4.', id: 2 },
      { type: 'stream_complete', id: 3 },
      { type: 'final_result', result: 'The answer is 4.', id: 4 },
    ]);
    const proses = t.turns[1]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['The answer is 4.']);
  });

  it('replay: a string return value renders as the reply, not just the repr footer', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      {
        type: 'model_response',
        raw_content:
          'Working on it.\n\n```python-exec\nreturn_value("All three notes match.")\n```',
        thought: 'Working on it.',
        id: 2,
      },
      {
        type: 'code_execution',
        code: 'return_value("All three notes match.")',
        return_value_repr: "'All three notes match.'",
        id: 3,
      },
      { type: 'final_result', result: 'All three notes match.', id: 4 },
      { type: 'session_end', status: 'success', id: 5 },
    ]);
    const proses = t.turns[1]!.blocks.filter((b) => b.kind === 'prose') as ProseBlock[];
    expect(proses.map((p) => p.text)).toEqual(['Working on it.', 'All three notes match.']);
  });
});

describe('splitStreamFence', () => {
  it('splits an open python-exec fence tail from the prose before it', () => {
    expect(splitStreamFence('Checking.\n\n```python-exec\nrun("ls")\nx = 1')).toEqual({
      text: 'Checking.',
      code: 'run("ls")\nx = 1',
    });
  });

  it('returns prose-only when no fence is open (closed fences stay in the text)', () => {
    expect(splitStreamFence('Done.\n\n```python-exec\nrun("a")\n```\nAll good.')).toEqual({
      text: 'Done.\n\n```python-exec\nrun("a")\n```\nAll good.',
    });
    expect(splitStreamFence('plain words')).toEqual({ text: 'plain words' });
  });

  it('handles a fence opener with no body yet', () => {
    expect(splitStreamFence('Now:\n```python-exec\n')).toEqual({ text: 'Now:', code: '' });
  });
});

describe('buildTimeline (persisted reasoning)', () => {
  it('replays a reasoning event as a think block before the response prose', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'q', id: 1 },
      { type: 'reasoning', content: 'weighing the options', id: 2 },
      {
        type: 'model_response',
        raw_content: 'Going with option A.',
        thought: 'Going with option A.',
        id: 3,
      },
      { type: 'final_result', result: 'Going with option A.', id: 4 },
      { type: 'session_end', status: 'success', id: 5 },
    ]);
    const ai = t.turns[1]!;
    const kinds = ai.blocks.map((b) => b.kind);
    expect(kinds).toEqual(['think', 'prose']);
    expect((ai.blocks[0] as { content: string }).content).toBe('weighing the options');
  });
});

describe('buildTimeline (content blocks)', () => {
  it('renders normalized content_blocks as panels between the thought and tail', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'write the report', id: 1 },
      {
        type: 'model_response',
        thought: 'Drafting it now.',
        content_blocks: { 'report.md': '# Q3 Report\ntotals below' },
        tail: 'Saved.',
        id: 2,
      },
      { type: 'session_end', status: 'success', id: 3 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.blocks.map((b) => b.kind)).toEqual(['prose', 'content', 'prose']);
    const content = ai.blocks[1] as { name: string; text: string };
    expect(content.name).toBe('report.md');
    expect(content.text).toBe('# Q3 Report\ntotals below');
  });

  it('renders a content-only response as just its panel', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'model_response', thought: '', content_blocks: { 'notes.txt': 'hello' }, id: 2 },
    ]);
    expect(t.turns[1]!.blocks).toEqual([{ kind: 'content', name: 'notes.txt', text: 'hello' }]);
  });

  it('renders a live content_block event as a named block, not prose', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'content_block', name: 'config.yaml', content: 'port: 80', id: 3 },
    ]);
    expect(t.turns[1]!.blocks).toEqual([
      { kind: 'content', name: 'config.yaml', text: 'port: 80' },
    ]);
  });
});

describe('buildTimeline (content block live double-emit)', () => {
  it('renders once when the settled frame AND the content_block event both carry it', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'stream_chunk', chunk: 'Saving.\n<content name="a.txt">\nbody\n</content>', id: 3 },
      { type: 'stream_complete', id: 4 },
      { type: 'model_response', thought: 'Saving.', content_blocks: { 'a.txt': 'body' }, id: 5 },
      { type: 'content_block', name: 'a.txt', content: 'body', id: 6 },
    ]);
    const blocks = t.turns[1]!.blocks;
    expect(blocks.filter((b) => b.kind === 'content')).toEqual([
      { kind: 'content', name: 'a.txt', text: 'body' },
    ]);
    expect(blocks.filter((b) => b.kind === 'prose').map((b) => (b as ProseBlock).text)).toEqual([
      'Saving.',
    ]);
  });

  it('dedupes regardless of arrival order (event first, frame second)', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'content_block', name: 'a.txt', content: 'body', id: 3 },
      { type: 'model_response', thought: '', content_blocks: { 'a.txt': 'body' }, id: 4 },
      { type: 'final_result', result: 'done', id: 5 },
    ]);
    const contents = t.turns[1]!.blocks.filter((b) => b.kind === 'content');
    expect(contents).toHaveLength(1);
  });
});

describe('buildTimeline (hooks)', () => {
  it('replays hook_execution as an exec row named hook:{name}', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      {
        type: 'hook_execution',
        phase: 'pre_message',
        name: 'precommit',
        command: './hooks/precommit.sh',
        exit_code: 0,
        duration_ms: 42,
        id: 2,
      },
      { type: 'model_response', raw_content: 'Done.', thought: 'Done.', id: 3 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.blocks[0]).toMatchObject({
      kind: 'exec',
      command: 'hook:precommit',
      status: 'done',
      exitCode: 0,
      meta: '42ms',
    });
  });

  it('a failing hook renders as an error row carrying its output', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      {
        type: 'hook_execution',
        phase: 'pre_message',
        exit_code: 1,
        stdout: 'checking',
        stderr: 'lint failed',
        id: 2,
      },
    ]);
    expect(t.turns[1]!.blocks[0]).toMatchObject({
      kind: 'exec',
      command: 'hook:pre_message',
      status: 'error',
      exitCode: 1,
      output: 'checking\nlint failed',
    });
  });

  it('hook_status sets the live status line; the execution result clears it', () => {
    const events = [
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'hook_status', message: 'Running precommit...', id: 3 },
    ];
    let t = buildTimeline(events);
    expect(t.turns[1]!.liveStatus).toBe('Running precommit...');

    t = buildTimeline([
      ...events,
      { type: 'hook_execution', name: 'precommit', exit_code: 0, id: 4 },
    ]);
    expect(t.turns[1]!.liveStatus).toBeUndefined();
  });

  it('turn end clears a stale live status', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'hook_status', message: 'Running notify...', id: 3 },
      { type: 'final_result', result: 'ok', id: 4 },
      { type: 'session_end', status: 'success', id: 5 },
    ]);
    expect(t.turns[1]!.liveStatus).toBeUndefined();
  });
});

describe('buildTimeline (context injections)', () => {
  // The events endpoint splits injections server-side: user_input arrives with
  // `injected` blocks and `display_text` (the words the person actually typed).
  it('renders a pure scheduled_task injection as a synthetic turn, not user prose', () => {
    const t = buildTimeline([
      {
        type: 'user_input',
        text: '<scheduled_task id="nightly-backup">\nThis task ran in the background.\n</scheduled_task>',
        injected: [
          { tag: 'scheduled_task', id: 'nightly-backup', body: 'This task ran in the background.' },
        ],
        display_text: '',
        id: 1,
      },
      {
        type: 'model_response',
        raw_content: 'Backup verified, 12 files.',
        thought: 'Backup verified, 12 files.',
        id: 2,
      },
    ]);
    const user = t.turns[0]!;
    expect(user.synthetic).toBe(true);
    expect(user.blocks).toEqual([]);
    expect(user.injected).toEqual([
      { tag: 'scheduled_task', id: 'nightly-backup', body: 'This task ran in the background.' },
    ]);
  });

  it('keeps the typed text and folds the served injection blocks', () => {
    const t = buildTimeline([
      {
        type: 'user_input',
        text: '<message_context>\nnow: 9am\n</message_context>\n<environment>\ncwd: /w\n</environment>\nfix the failing test',
        injected: [
          { tag: 'message_context', body: 'now: 9am' },
          { tag: 'environment', body: 'cwd: /w' },
        ],
        display_text: 'fix the failing test',
        id: 1,
      },
    ]);
    const user = t.turns[0]!;
    expect(user.synthetic).toBeUndefined();
    expect(user.blocks).toEqual([{ kind: 'prose', text: 'fix the failing test' }]);
    expect(user.injected?.map((b) => b.tag)).toEqual(['message_context', 'environment']);
  });

  it('renders ordinary messages (no injected/display_text fields) from text', () => {
    const t = buildTimeline([{ type: 'user_input', text: 'plain question', id: 1 }]);
    expect(t.turns[0]!.injected).toBeUndefined();
    expect(t.turns[0]!.blocks).toEqual([{ kind: 'prose', text: 'plain question' }]);
  });

  it('reads a client_context injection as structured items alongside the typed text', () => {
    const t = buildTimeline([
      {
        type: 'user_input',
        text: 'where am i',
        injected: [
          {
            tag: 'client_context',
            items: [{ key: 'location', label: 'Location', value: '37.77490, -122.41940 (±20m)' }],
          },
        ],
        display_text: 'where am i',
        id: 1,
      },
    ]);
    const user = t.turns[0]!;
    expect(user.blocks).toEqual([{ kind: 'prose', text: 'where am i' }]);
    expect(user.injected).toEqual([
      {
        tag: 'client_context',
        body: '',
        items: [{ key: 'location', label: 'Location', value: '37.77490, -122.41940 (±20m)' }],
      },
    ]);
  });

  it('leaves plain injections (a body, no items) with items undefined', () => {
    const t = buildTimeline([
      {
        type: 'user_input',
        text: 'x',
        injected: [{ tag: 'message_context', body: 'now: 9am' }],
        display_text: 'x',
        id: 1,
      },
    ]);
    expect(t.turns[0]!.injected).toEqual([{ tag: 'message_context', body: 'now: 9am' }]);
  });
});

describe('buildTimeline (attachments)', () => {
  it('surfaces a user_input event’s attachments onto the user turn as uploads paths', () => {
    const t = buildTimeline([
      {
        type: 'user_input',
        text: 'look at these',
        attachments: [
          { name: 'photo.png', type: 'image', source_url: null },
          { name: 'notes.pdf', type: 'document', source_url: null },
        ],
        id: 1,
      },
    ]);
    const user = t.turns[0]!;
    expect(user.role).toBe('user');
    expect(user.attachments).toEqual([
      { name: 'photo.png', type: 'image', path: 'uploads/photo.png' },
      { name: 'notes.pdf', type: 'document', path: 'uploads/notes.pdf' },
    ]);
    // The typed message still renders as prose; attachments are turn-level metadata.
    expect((user.blocks[0] as ProseBlock).text).toBe('look at these');
  });

  it('leaves attachments undefined for a message that carried none', () => {
    const t = buildTimeline([{ type: 'user_input', text: 'hi', id: 1 }]);
    expect(t.turns[0]!.attachments).toBeUndefined();
  });

  it('skips malformed attachment entries (no name) rather than throwing', () => {
    const t = buildTimeline([
      {
        type: 'user_input',
        text: 'q',
        attachments: [{ type: 'image' }, { name: 'ok.png', type: 'image' }],
        id: 1,
      },
    ]);
    expect(t.turns[0]!.attachments).toEqual([
      { name: 'ok.png', type: 'image', path: 'uploads/ok.png' },
    ]);
  });
});

describe('buildTimeline (info / send_message)', () => {
  it('renders a persisted info event as agent prose (send_message replay)', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'process the files', id: 1 },
      { type: 'info', message: 'Starting file analysis...', id: 2 },
      { type: 'code_execution', code: 'run("analyze")', output: 'done', id: 3 },
      { type: 'info', message: 'Found 42 files, processing...', id: 4 },
      { type: 'final_result', result: 'Analysis complete', id: 5 },
    ]);
    const ai = t.turns[1]!;
    const kinds = ai.blocks.map((b) => b.kind);
    expect(kinds).toEqual(['prose', 'code', 'prose']);
    expect((ai.blocks[0] as { text: string }).text).toBe('Starting file analysis...');
    // Consecutive prose merges (info + final flow as one passage).
    expect((ai.blocks[2] as { text: string }).text).toBe(
      'Found 42 files, processing...\n\nAnalysis complete',
    );
  });

  it('a live info frame lands in the open turn between tool activity', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'turn_start', id: 2 },
      { type: 'info', message: 'Heads up: this will take a minute.', id: 3 },
    ]);
    expect(t.turns[1]!.blocks).toEqual([
      { kind: 'prose', text: 'Heads up: this will take a minute.' },
    ]);
  });
});

describe('buildTimeline (pre-message hooks recorded before user_input)', () => {
  it('attaches a hook that fired before the user message to the FOLLOWING ai turn', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'first', id: 1 },
      { type: 'final_result', result: 'done', id: 2 },
      { type: 'session_end', status: 'success', id: 3 },
      // pre_message hook lands BEFORE its user_input in the history.
      { type: 'hook_execution', name: 'rag_context', exit_code: 0, stdout: 'chunks', id: 4 },
      { type: 'user_input', text: 'what about a queue?', id: 5 },
      {
        type: 'model_response',
        raw_content: 'Yes, that makes sense.',
        thought: 'Yes, that makes sense.',
        id: 6,
      },
    ]);
    const roles = t.turns.map((x) => x.role);
    expect(roles).toEqual(['user', 'ai', 'user', 'ai']);
    const answer = t.turns[3]!;
    expect(answer.blocks[0]).toMatchObject({ kind: 'exec', command: 'hook:rag_context' });
    expect(answer.blocks[1]).toMatchObject({ kind: 'prose', text: 'Yes, that makes sense.' });
  });

  it('a trailing stashed hook still renders (no following model events)', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      { type: 'hook_execution', name: 'precommit', exit_code: 1, stderr: 'lint failed', id: 2 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.blocks[0]).toMatchObject({
      kind: 'exec',
      command: 'hook:precommit',
      status: 'error',
    });
  });
});

describe('buildTimeline (thought + tail around the code panel)', () => {
  // Fence parsing (nested fences in strings, unprovable closes, first-fence-
  // only) lives in the backend parser and its pytest suite; the reducer just
  // renders the served thought/tail around the code_execution block.
  it('renders thought before and tail after the executed code block', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'design it', id: 1 },
      {
        type: 'model_response',
        raw_content:
          'Designing the routing.\n\n```python-exec\nwrite_file(...)\n``` \n\nWrote the routing sketch, want edits?',
        thought: 'Designing the routing.',
        tail: 'Wrote the routing sketch, want edits?',
        id: 2,
      },
      { type: 'code_execution', code: 'write_file(...)', output: '', id: 3 },
    ]);
    const ai = t.turns[1]!;
    expect(ai.blocks.map((b) => b.kind)).toEqual(['prose', 'code']);
    const prose = (ai.blocks[0] as ProseBlock).text;
    expect(prose).toContain('Designing the routing.');
    expect(prose).toContain('want edits?');
    // raw_content never renders: the fence and its nested yaml stay out.
    expect(prose).not.toContain('python-exec');
  });

  it('never renders raw_content when the parse has no tail (dropped as unprovable)', () => {
    const t = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      {
        type: 'model_response',
        raw_content: 'Thinking.\n\n```python-exec\ns = """\nnever closed',
        thought: 'Thinking.',
        id: 2,
      },
    ]);
    const prose = t.turns[1]!.blocks.filter((b) => b.kind === 'prose')
      .map((b) => (b as { text: string }).text)
      .join('\n');
    expect(prose).toBe('Thinking.');
  });
});

describe('buildTimeline (per-turn cache usage)', () => {
  it("headline cacheRead is the LAST step's read; summed totals + step count ride the meta", () => {
    const ai = buildTimeline([
      { type: 'user_input', text: 'compute totals', id: 1 },
      {
        type: 'model_response',
        turn: 0,
        thought: 'step one',
        usage: { total_tokens: 115, cache_read_input_tokens: 80, cache_creation_input_tokens: 20 },
        id: 2,
      },
      { type: 'code_execution', code: 'x = 1', id: 3 },
      {
        type: 'model_response',
        turn: 1,
        thought: 'step two',
        usage: { total_tokens: 90, cache_read_input_tokens: 40, cache_creation_input_tokens: 0 },
        id: 4,
      },
      { type: 'final_result', result: 'done', turns: 2, tokens: 200, cost: 0.01, id: 5 },
    ]).turns[1]!;
    // Headline = the last step's cached-prefix size (matches the meter's scale, not
    // the misleading cross-step sum). Summed reads/writes + the step count ride the
    // meta for the tooltip; final_result's tokens/cost/turns survive alongside.
    expect(ai.meta).toEqual({
      tokens: 200,
      cost: 0.01,
      turns: 2,
      cacheRead: 40,
      cacheReadTotal: 120,
      cacheWriteTotal: 20,
      cacheSteps: 2,
    });
  });

  it('omits cache fields entirely when no model_response reported them (honest absence)', () => {
    const ai = buildTimeline([
      { type: 'user_input', text: 'hi', id: 1 },
      { type: 'model_response', turn: 0, thought: 'hey', usage: { total_tokens: 50 }, id: 2 },
      { type: 'final_result', result: 'hey', turns: 1, tokens: 50, id: 3 },
    ]).turns[1]!;
    expect(ai.meta).toEqual({ tokens: 50, cost: undefined, turns: 1 });
    expect(ai.meta).not.toHaveProperty('cacheRead');
    expect(ai.meta).not.toHaveProperty('cacheWrite');
  });

  it('carries cache onto a still-streaming turn (before final_result lands)', () => {
    const ai = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      {
        type: 'model_response',
        turn: 0,
        thought: 'working',
        usage: { cache_read_input_tokens: 8100 },
        id: 2,
      },
    ]).turns[1]!;
    // Single step: headline and total coincide, one step counted.
    expect(ai.meta?.cacheRead).toBe(8100);
    expect(ai.meta?.cacheReadTotal).toBe(8100);
    expect(ai.meta?.cacheSteps).toBe(1);
    expect(ai.streaming).toBe(true);
  });

  it('reads OpenAI-family cached_tokens as the cache-read headline (no Anthropic fields)', () => {
    // openai_compat / codex_cli report the cached prompt prefix on the unified
    // `cached_tokens` field, not Anthropic's cache_read_input_tokens. The footer
    // must still render a cache headline from it, else those turns look uncached.
    const ai = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      {
        type: 'model_response',
        turn: 0,
        thought: 'working',
        usage: { total_tokens: 23816, cached_tokens: 18000 },
        id: 2,
      },
      { type: 'final_result', result: 'working', turns: 1, tokens: 23816, id: 3 },
    ]).turns[1]!;
    expect(ai.meta?.cacheRead).toBe(18000);
    expect(ai.meta?.cacheReadTotal).toBe(18000);
    expect(ai.meta?.cacheSteps).toBe(1);
  });

  it('prefers an explicit cache_read_input_tokens over cached_tokens (Anthropic composite)', () => {
    // Anthropic folds creation+read into cached_tokens but also sets the explicit
    // read; the headline must take the explicit read, never the composite.
    const ai = buildTimeline([
      { type: 'user_input', text: 'go', id: 1 },
      {
        type: 'model_response',
        turn: 0,
        thought: 'working',
        usage: {
          cached_tokens: 300,
          cache_read_input_tokens: 120,
          cache_creation_input_tokens: 180,
        },
        id: 2,
      },
    ]).turns[1]!;
    expect(ai.meta?.cacheRead).toBe(120);
    expect(ai.meta?.cacheWriteTotal).toBe(180);
  });
});

describe('buildTimeline (windowed suffix)', () => {
  // The tail-window initial load can start mid-history: the first event in the
  // window may be a bare AI-turn fragment, a pre-message hook, a compaction
  // banner, or a user turn cut from its predecessors. The reducer must fold a
  // suffix without crashing or mis-pairing.
  it('renders a suffix that opens mid-AI-turn as a single AI turn (no crash, no phantom user)', () => {
    const t = buildTimeline([
      { type: 'model_response', turn: 3, thought: 'partial answer', id: 40 },
      { type: 'final_result', result: 'partial answer', id: 41 },
    ]);
    expect(t.turns.map((x) => x.role)).toEqual(['ai']);
    const ai = t.turns[0]!;
    expect(ai.streaming).toBe(false);
    expect((ai.blocks[0] as ProseBlock).text).toBe('partial answer');
  });

  it('a hook_execution first in the window attaches to the following AI turn, never a phantom bubble', () => {
    const t = buildTimeline([
      { type: 'hook_execution', name: 'precommit', exit_code: 0, id: 50 },
      { type: 'user_input', text: 'next', id: 51 },
      { type: 'model_response', turn: 0, thought: 'done', id: 52 },
      { type: 'final_result', result: 'done', id: 53 },
    ]);
    expect(t.turns.map((x) => x.role)).toEqual(['user', 'ai']);
    const ai = t.turns[1]!;
    expect(ai.blocks.some((b) => b.kind === 'exec' && b.command === 'hook:precommit')).toBe(true);
  });

  it('a trailing pre-message hook with nothing after it still renders (settled, not spinning)', () => {
    const t = buildTimeline([
      { type: 'final_result', result: 'earlier', id: 60 },
      { type: 'hook_execution', name: 'notify', exit_code: 0, id: 61 },
    ]);
    const ai = t.turns[t.turns.length - 1]!;
    expect(ai.role).toBe('ai');
    expect(ai.streaming).toBe(false);
    expect(ai.blocks.some((b) => b.kind === 'exec' && b.command === 'hook:notify')).toBe(true);
  });

  it('a compaction banner inside the window is collected (with its summary), not dropped or crashed', () => {
    const t = buildTimeline([
      {
        type: 'compaction',
        replaced_count: 12,
        retained_count: 2,
        summary: 'folded 12 turns into a summary',
        id: 70,
      },
      { type: 'user_input', text: 'after compaction', id: 71 },
      { type: 'model_response', turn: 0, thought: 'ok', id: 72 },
      { type: 'final_result', result: 'ok', id: 73 },
    ]);
    expect(t.compactions).toHaveLength(1);
    expect(t.compactions[0]!.replacedCount).toBe(12);
    expect(t.compactions[0]!.summary).toBe('folded 12 turns into a summary');
    expect(t.turns.map((x) => x.role)).toEqual(['user', 'ai']);
  });

  it('a lone tool_result at the window edge (its call cut off) is ignored, not a crash', () => {
    const t = buildTimeline([
      { type: 'tool_result', output: 'orphaned', id: 80 },
      { type: 'user_input', text: 'q', id: 81 },
      { type: 'final_result', result: 'a', id: 82 },
    ]);
    expect(t.turns.map((x) => x.role)).toEqual(['ai', 'user', 'ai']);
  });
});
