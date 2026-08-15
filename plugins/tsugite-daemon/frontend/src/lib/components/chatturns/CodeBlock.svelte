<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import ExecBlock from './ExecBlock.svelte';
  import { codeRows } from './codeRows';

  interface CodeCallRow {
    tool: string;
    status: 'running' | 'done' | 'error' | 'ended';
    args?: Record<string, unknown>;
    output?: string;
    meta?: string;
    groupId?: string;
  }

  interface CodeGroupRow {
    id: string;
    title: string;
    success?: boolean;
    meta?: string;
    error?: string;
  }

  let {
    code = '',
    lang,
    filename,
    streaming = false,
    running = false,
    collapsible = true,
    collapsed = false,
    output,
    calls = [],
    groups = [],
    returnValue,
    meta,
    children,
    onCopy,
  }: {
    // Raw code text - source of truth for copy + the line count, and the default
    // rendering when no highlighted `children` are supplied.
    code?: string;
    lang?: string;
    filename?: string;
    // Info-blue border while tokens are still arriving.
    streaming?: boolean;
    // The execution is in flight (ok-green border + header spinner).
    running?: boolean;
    // Whether the collapse control is offered.
    collapsible?: boolean;
    // Collapsed state; followed across prop changes so a finished run
    // auto-collapses its code (a manual toggle wins until the prop next flips).
    collapsed?: boolean;
    // Combined run output (persisted code_execution replay; the live path
    // carries per-call outputs on `calls` instead).
    output?: string;
    // Individual tool calls of this execution, rendered as exec rows.
    calls?: CodeCallRow[];
    // `tsu_group` sections; a call carrying a group id renders under that heading.
    groups?: CodeGroupRow[];
    returnValue?: string;
    // Run duration for the header ("0.4s").
    meta?: string;
    // Optional pre-highlighted `<code>` inner markup (e.g. tok-* spans). When
    // given it renders instead of the plain `code` text; copy still uses `code`.
    children?: Snippet;
    onCopy?: (text: string) => void;
  } = $props();

  // A finished block folds to its header so the conversation reads as a
  // conversation; a failure stays open, since that is the block worth reading.
  const failed = $derived(
    calls.some((c) => c.status === 'error') || groups.some((g) => g.success === false),
  );
  const foldable = $derived(collapsed && !failed);

  let userOverride = $state<{ prop: boolean; value: boolean } | null>(null);
  const isCollapsed = $derived(userOverride?.prop === foldable ? userOverride.value : foldable);
  // Conditionally rendered now (the summary replaces it), so the binding is reactive.
  let codeEl = $state<HTMLElement | undefined>(undefined);

  const rows = $derived(codeRows(calls, groups));
  // Folded, the group titles are what the agent said it was doing; without any,
  // the tool names are the next best account of it. Repeats collapse, since three
  // read_file rows say nothing three times. A block that called nothing keeps its
  // code peek, which is then the only thing left to show.
  const summary = $derived.by(() => {
    if (!isCollapsed) return '';
    const labels = groups.length > 0 ? groups.map((g) => g.title) : calls.map((c) => c.tool);
    return [...new Set(labels)].join(' · ');
  });

  // The result (combined output) section expands on click like the code does:
  // capped + masked by default, full height once opened. The affordance only
  // renders when the output actually overflows its cap.
  let outOpen = $state(false);
  let outOverflow = $state(false);
  let outPre = $state<HTMLElement>();
  $effect(() => {
    void output;
    if (!outOpen && outPre) outOverflow = outPre.scrollHeight > outPre.clientHeight + 1;
  });

  // While tokens stream in, keep the newest lines in view (the pre is
  // height-capped, so an unpinned scroll would show only the stale top).
  $effect(() => {
    if (!streaming) return;
    void code;
    const pre = codeEl?.parentElement;
    if (pre) pre.scrollTop = pre.scrollHeight;
  });

  function toggleCollapsed() {
    userOverride = { prop: foldable, value: !isCollapsed };
  }

  // Count newlines without allocating a split array (code can be large + stream).
  function countLines(s: string): number {
    if (!s) return 0;
    let n = 1;
    for (let i = s.indexOf('\n'); i !== -1; i = s.indexOf('\n', i + 1)) n++;
    return n;
  }
  const lineCount = $derived(countLines(code));

  async function copy() {
    const text = code || codeEl?.textContent || '';
    try {
      await navigator.clipboard?.writeText(text);
    } catch {
      // Clipboard may be unavailable (insecure context / denied) - the callback
      // still fires so the app can surface its own affordance.
    }
    onCopy?.(text);
  }
</script>

<div
  class="t-code"
  class:is-collapsed={isCollapsed}
  class:is-streaming={streaming}
  class:is-run={running}
>
  <div class="t-code-hd">
    {#if lang}<span class="lang">{lang}</span>{/if}
    {#if filename}<span>{filename}</span>{/if}
    {#if running}<span class="run"><Spin />running</span>{/if}
    {#if streaming}
      <span class="streamflag"
        ><span class="ic-stream" aria-hidden="true"><i></i><i></i><i></i></span>streaming</span
      >
    {/if}
    <div class="grow"></div>
    {#if isCollapsed && calls.length > 0}
      <span>{calls.length} {calls.length === 1 ? 'tool' : 'tools'}</span>
    {/if}
    {#if meta}<span class="meta">{meta}</span>{/if}
    {#if lineCount > 0}<span>{lineCount} {lineCount === 1 ? 'line' : 'lines'}</span>{/if}
    <button type="button" class="t-iconbtn" onclick={copy} aria-label="Copy code">
      <Icon name="copy" size={11} />copy
    </button>
    {#if collapsible}
      <button
        type="button"
        class="t-iconbtn"
        aria-expanded={!isCollapsed}
        aria-label={isCollapsed ? 'Expand code' : 'Collapse code'}
        onclick={toggleCollapsed}
      >
        {isCollapsed ? 'expand' : 'collapse'}
      </button>
    {/if}
  </div>
  {#if summary}
    <!-- Folded with groups: the agent's own labels say more than a line of code.
         Pointer-only, like .pre-expand below; the header button is the keyboard control. -->
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="t-code-summary" onclick={toggleCollapsed}>{summary}</div>
  {:else}
    <div class="pre-wrap">
      <pre><code bind:this={codeEl}
          >{#if children}{@render children()}{:else}{code}{/if}</code
        ></pre>
      {#if isCollapsed && collapsible}
        <!-- Pointer-only click target over the collapsed peek (same pattern as the
             mux tab-close glyph); the header button is the keyboard/AT control. -->
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <span class="pre-expand" aria-hidden="true" onclick={toggleCollapsed}></span>
      {/if}
    </div>
  {/if}
  {#snippet callRow(call: CodeCallRow)}
    <!-- Running rows stay CLOSED (the header spinner carries the signal):
         auto-opening each call while a block streams many of them makes the
         timeline flap open/shut. Only failures open themselves. -->
    <ExecBlock
      command={call.tool}
      status={call.status}
      args={call.args}
      output={call.output}
      meta={call.meta}
      open={call.status === 'error'}
    />
  {/snippet}

  {#if rows.length > 0}
    <div class="t-code-calls">
      {#each rows as row, i (i)}
        {#if row.kind === 'call'}
          {@render callRow(row.call)}
        {:else}
          <div class="t-code-group" class:is-err={row.group.success === false}>
            <div class="grp-hd">
              <span class="grp-title">{row.group.title}</span>
              {#if row.group.meta}<span class="grp-meta">{row.group.meta}</span>{/if}
              {#if row.group.success === false}<span class="grp-flag">failed</span>{/if}
            </div>
            {#each row.calls as call, j (j)}
              {@render callRow(call)}
            {/each}
            {#if row.group.error}<div class="grp-err">{row.group.error}</div>{/if}
          </div>
        {/if}
      {/each}
    </div>
  {/if}
  {#if output}
    <div class="t-code-out" class:is-open={outOpen}>
      <pre bind:this={outPre}>{output}</pre>
      {#if outOverflow && !outOpen}
        <!-- Pointer-only click target over the capped output (same pattern as the
             collapsed code peek); the strip below is the keyboard/AT control. -->
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <span class="out-expand" aria-hidden="true" onclick={() => (outOpen = true)}></span>
      {/if}
      {#if outOverflow || outOpen}
        <button
          type="button"
          class="out-toggle"
          aria-expanded={outOpen}
          onclick={() => (outOpen = !outOpen)}
          >{outOpen ? 'collapse output' : 'expand output'}</button
        >
      {/if}
    </div>
  {/if}
  {#if returnValue}
    <div class="t-code-rv"><span class="arrow">→</span> {returnValue}</div>
  {/if}
</div>

<style>
  .t-code {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    background: var(--bg1);
    overflow: hidden;
    position: relative;
  }
  .t-code-hd {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 9px;
    border-bottom: 1px solid var(--bd0);
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .t-code-hd .lang {
    color: var(--tx2);
    font-weight: 600;
    letter-spacing: 0.05em;
  }
  .t-code-hd .grow {
    flex: 1;
  }
  .t-code pre {
    margin: 0;
    padding: 9px 11px;
    font: 400 var(--fs-sm) / 1.65 var(--font-mono);
    color: var(--tx1);
    overflow: auto;
    max-height: 260px;
    tab-size: 2;
  }
  .pre-wrap {
    position: relative;
  }
  .t-code-summary {
    padding: 9px 11px;
    font: 400 var(--fs-sm) / 1.65 var(--font-ui);
    color: var(--tx2);
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  /* One line of peek: enough to say what ran, short enough that a finished turn
     reads as a line of conversation. */
  .t-code.is-collapsed pre {
    /* One line plus the pre's own 9px vertical padding (border-box). */
    max-height: calc(1lh + 18px);
    overflow: hidden;
    mask-image: linear-gradient(#000 55%, transparent);
  }
  .pre-expand {
    position: absolute;
    inset: 0;
    cursor: pointer;
  }
  .t-code.is-streaming {
    border-color: color-mix(in oklab, var(--st-info) 40%, transparent);
  }
  .t-code.is-streaming .t-code-hd .lang {
    color: var(--st-info);
  }
  .t-code.is-run {
    border-color: color-mix(in oklab, var(--st-ok) 40%, transparent);
  }
  .t-code-hd .run {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--st-ok);
    font-weight: 600;
  }
  .t-code-hd .streamflag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--st-info);
    font-weight: 600;
  }
  /* streaming bars - same no-shared-owner precedent as Pill's copy. */
  .ic-stream {
    display: inline-flex;
    align-items: flex-end;
    gap: 1.5px;
    height: 10px;
    width: 11px;
  }
  .ic-stream i {
    width: 2px;
    background: currentColor;
    border-radius: 1px;
    animation: tbars 1s var(--ease) infinite;
  }
  .ic-stream i:nth-child(1) {
    height: 40%;
    animation-delay: 0ms;
  }
  .ic-stream i:nth-child(2) {
    height: 90%;
    animation-delay: 160ms;
  }
  .ic-stream i:nth-child(3) {
    height: 60%;
    animation-delay: 320ms;
  }
  @keyframes tbars {
    0%,
    100% {
      transform: scaleY(0.5);
    }
    50% {
      transform: scaleY(1);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .ic-stream i {
      animation: none;
      transform: none;
    }
  }
  .t-code-hd .meta {
    font-variant-numeric: tabular-nums;
  }

  /* Individual tool calls of this execution, each an exec disclosure row -
     visible even while the code itself is collapsed. */
  .t-code-calls {
    display: grid;
    gap: 6px;
    padding: 8px;
    border-top: 1px solid var(--bd0);
    background: var(--bg0);
  }
  /* Indents a tsu_group's calls under its label so a long execution reads as
     named steps. */
  .t-code-group {
    display: grid;
    gap: 6px;
    padding-left: 8px;
    border-left: 2px solid var(--bd1);
  }
  .t-code-group.is-err {
    border-left-color: var(--st-err);
  }
  .t-code-group .grp-hd {
    display: flex;
    align-items: baseline;
    gap: 6px;
  }
  .t-code-group .grp-title {
    font: 600 var(--fs-xs) / 1.4 var(--font-mono);
    color: var(--tx2);
  }
  .t-code-group .grp-meta {
    font: 500 var(--fs-2xs) / 1.4 var(--font-mono);
    color: var(--tx3);
  }
  /* Failure is a word, not just the red rail (state is never color alone). */
  .t-code-group .grp-flag {
    font: 600 var(--fs-2xs) / 1.4 var(--font-mono);
    color: var(--st-err);
  }
  .t-code-group .grp-err {
    font: 500 var(--fs-2xs) / 1.4 var(--font-mono);
    color: var(--st-err);
  }

  /* Folded: only the header stays, carrying the tool count and duration. */
  .t-code.is-collapsed .t-code-calls,
  .t-code.is-collapsed .t-code-out {
    display: none;
  }

  .t-code-out {
    position: relative;
  }
  .t-code-out pre {
    margin: 0;
    padding: 8px 11px;
    border-top: 1px solid var(--bd0);
    background: var(--bg0);
    color: var(--tx2);
    font: 400 var(--fs-xs) / 1.6 var(--font-mono);
    max-height: 180px;
    overflow: hidden;
  }
  .t-code-out:not(.is-open) pre {
    mask-image: linear-gradient(#000 70%, transparent);
  }
  /* Open = the full result, viewport-capped so a huge dump can't swallow the
     timeline; inner scroll takes over past the cap. */
  .t-code-out.is-open pre {
    max-height: 60vh;
    overflow: auto;
    mask-image: none;
  }
  .out-expand {
    position: absolute;
    inset: 0 0 22px;
    cursor: pointer;
  }
  .out-toggle {
    display: block;
    width: 100%;
    padding: 2px 11px 4px;
    background: var(--bg0);
    border: 0;
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    text-align: left;
    cursor: pointer;
  }
  .out-toggle:hover {
    color: var(--tx0);
  }
  .t-code-rv {
    padding: 5px 11px;
    border-top: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .t-code-rv .arrow {
    color: var(--st-ok);
    font-weight: 600;
  }

  /* Syntax tokens live inside the {@html}/snippet `children`, so :global(). */
  .t-code :global(.tok-k) {
    color: var(--brand);
  }
  .t-code :global(.tok-s) {
    color: var(--st-ok);
  }
  .t-code :global(.tok-f) {
    color: var(--acc);
  }
  .t-code :global(.tok-c) {
    color: var(--tx3);
    font-style: italic;
  }
  .t-code :global(.tok-n) {
    color: var(--st-warn);
  }

  /* .t-iconbtn is a bare icon-button with no shared component (Toast keeps its
     own too); its 11px icon sizing lives globally in tokens.css (.t-iconbtn .ic). */
  .t-iconbtn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: none;
    border: 0;
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    cursor: pointer;
    padding: 2px 4px;
    border-radius: var(--r-sm);
  }
  .t-iconbtn:hover {
    color: var(--tx0);
    background: var(--bg3);
  }
</style>
