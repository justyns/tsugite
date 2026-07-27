<script lang="ts">
  import Msg from './Msg.svelte';
  import Prose from './Prose.svelte';
  import Think from './Think.svelte';
  import CodeBlock from './CodeBlock.svelte';
  import ExecBlock from './ExecBlock.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  const reasoning =
    'Sleep/wake kills the TCP connection without an error event in some browsers, ' +
    'so reconnect must be driven by a heartbeat gap, not just `onerror`. Backoff ' +
    'needs jitter to avoid thundering-herd against the local daemon after wake.';

  const aiProse =
    'Found the client — one `EventSource` wrapper, reconnect currently naïve ' +
    '(fixed 1s). Plan: **decorrelated jitter** capped at 30s, reset on any received ' +
    'event, and a `state` field the UI can subscribe to. Complexity stays ' +
    '<span class="math">O(1)</span> per tick.';

  const sseCode = `function nextDelay(attempt: number, prev: number) {
  const base = 1_000, cap = 30_000
  // decorrelated jitter (AWS-style)
  const next = Math.min(cap, rand(base, prev * 3))
  return attempt === 0 ? base : next
}

es.addEventListener('open', () => {
  attempt = 0
  store.set({ state: 'open', lastEventAt: Date.now() })
})`;

  // Pre-highlighted rendering of sseCode (tok-* spans). Injected via {@html} so
  // the literal braces don't collide with Svelte's expression syntax.
  const sseHighlighted = `<span class="tok-k">function</span> <span class="tok-f">nextDelay</span>(attempt: <span class="tok-k">number</span>, prev: <span class="tok-k">number</span>) {
  <span class="tok-k">const</span> base = <span class="tok-n">1_000</span>, cap = <span class="tok-n">30_000</span>
  <span class="tok-c">// decorrelated jitter (AWS-style)</span>
  <span class="tok-k">const</span> next = Math.<span class="tok-f">min</span>(cap, <span class="tok-f">rand</span>(base, prev * <span class="tok-n">3</span>))
  <span class="tok-k">return</span> attempt === <span class="tok-n">0</span> ? base : next
}

es.<span class="tok-f">addEventListener</span>(<span class="tok-s">'open'</span>, () => {
  attempt = <span class="tok-n">0</span>
  store.<span class="tok-f">set</span>({ state: <span class="tok-s">'open'</span>, lastEventAt: Date.<span class="tok-f">now</span>() })
})`;

  const rgOutput = `src/lib/sse.ts:14:  const es = new EventSource(url, { withCredentials: true })
src/lib/sse.ts:41:  // TODO: reconnect is naive — fixed 1s retry
src/routes/+layout.svelte:22:  import { sse } from '$lib/sse'`;

  const testOutput = `> @tsugite/sse@0.0.0 test
> vitest --watch

 ✓ backoff resets on open (4ms)
 ✓ caps at 30s (2ms)
 ⠋ waiting for changes…`;

  const errOutput = `tests/test_reconnect.py::test_stale_event FAILED
E   assert 'stale' in emitted
E    +  where emitted = ['open', 'backoff']
1 failed, 6 passed in 2.31s`;
</script>

<section data-testid="gallery-chatturns">
  <p class="note">
    Chat-turn components — a realistic conversation using every turn type, then each state in
    isolation. Hover a message to reveal its action bar.
  </p>

  <div class="convo-demo">
    <div class="convo-day">today · jul 12</div>

    <Msg role="user" who="you" at="14:22" index={1}>
      <Prose
        content="The SSE client drops on laptop sleep and never recovers. Add exponential backoff with jitter, cap at 30s, and surface reconnect state to the UI."
      />
    </Msg>

    <Msg role="ai" who="tsugite" at="14:23" index={2}>
      <Think label="thought for 6s" tokens={1024} content={reasoning} />
      <Prose content={aiProse} />
      <ExecBlock
        command={'rg -n "EventSource" src/'}
        status="done"
        exitCode={0}
        meta="0.4s"
        output={rgOutput}
      />
      <CodeBlock code={sseCode} lang="ts" filename="src/lib/sse.ts"
        >{@html sseHighlighted}</CodeBlock
      >
      <Prose
        content="Patch attached — reconnect state lands in the `sse` store as `open | backoff | closed`."
      />
      <!-- .row is a bare flex util (no owner). This .t-chip stays inline: it's an
           <a download> with a 9px trailing icon and a link hover, whereas Chip is a
           <span> that forces its icons to 10px — swapping would change pixels. -->
      <div class="row">
        <a class="t-chip" href="#gallery" download>
          <Icon name="file" size={10} />backoff.patch · 2.1 KB<Icon name="down" size={9} />
        </a>
      </div>
      <ExecBlock
        command="npm test -w @tsugite/sse --watch"
        status="running"
        meta="12:34"
        output={testOutput}
        open
        onOpenExternal={() => {}}
      />
    </Msg>

    <Msg role="user" who="you" at="14:29" index={3} pinnedActs>
      <Prose
        content="Good. Also emit a `stale` event when the last event is >15s old — and ship it, but ask me before pushing."
      />
    </Msg>

    <Msg role="ai" who="tsugite" at="14:31" index={4} streaming>
      <Prose content="Emitting the `stale` event now and wiring it into the store…" />
    </Msg>
  </div>

  <div class="swatches">
    <div class="cell">
      <span class="lbl">Think · collapsed</span>
      <Think label="thought for 6s" tokens={1024} content={reasoning} />
    </div>
    <div class="cell">
      <span class="lbl">Think · open</span>
      <Think label="thought for 6s" tokens={2048} content={reasoning} open />
    </div>

    <div class="cell">
      <span class="lbl">CodeBlock · plain</span>
      <CodeBlock
        code={"def slugify(text):\n    return text.strip().lower().replace(' ', '-')"}
        lang="py"
        filename="util.py"
      />
    </div>
    <div class="cell">
      <span class="lbl">CodeBlock · collapsed</span>
      <CodeBlock code={sseCode} lang="ts" filename="src/lib/sse.ts" collapsed
        >{@html sseHighlighted}</CodeBlock
      >
    </div>
    <div class="cell">
      <span class="lbl">CodeBlock · streaming</span>
      <CodeBlock
        code={'const es = new EventSource(url, {\n  withCredentials: true'}
        lang="ts"
        filename="src/lib/sse.ts"
        streaming
      />
    </div>

    <div class="cell">
      <span class="lbl">ExecBlock · done (exit 0)</span>
      <ExecBlock
        command={'rg -n "EventSource" src/'}
        status="done"
        exitCode={0}
        meta="0.4s"
        output={rgOutput}
        open
      />
    </div>
    <div class="cell">
      <span class="lbl">ExecBlock · error (exit 1)</span>
      <ExecBlock
        command="pytest tests/test_reconnect.py"
        status="error"
        exitCode={1}
        meta="2.3s"
        output={errOutput}
        open
      />
    </div>
    <div class="cell">
      <span class="lbl">ExecBlock · running</span>
      <ExecBlock
        command="npm run build"
        status="running"
        meta="0:07"
        output="vite v8.0.0 building for production…"
        open
        onOpenExternal={() => {}}
      />
    </div>

    <div class="cell wide">
      <span class="lbl">Msg · action bar (pinned)</span>
      <div class="convo-demo">
        <Msg role="ai" who="tsugite" at="09:15" index={7} pinnedActs>
          <Prose
            content="Retry regenerates this turn; the bar is normally revealed on hover or focus."
          />
        </Msg>
      </div>
    </div>
    <div class="cell wide">
      <span class="lbl">Prose · markdown sampler</span>
      <Prose
        content={'Supports **strong**, *emphasis*, inline `code`, links like [the daemon](#gallery), and lists:\n\n- decorrelated jitter\n- capped at 30s\n- resets on any event\n\nInline math renders as <span class="math">O(log n)</span>.'}
      />
    </div>
  </div>
</section>

<style>
  .note {
    max-width: 76ch;
    margin: 0 0 var(--sp-4);
    color: var(--tx2);
    font-size: var(--fs-sm);
    line-height: 1.6;
  }
  /* Messages are transparent and rely on the conversation surface behind them. */
  .convo-demo {
    background: var(--bg0);
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
    overflow: hidden;
    max-width: 860px;
  }
  .convo-day {
    text-align: center;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    padding: 12px 0 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .swatches {
    margin-top: var(--sp-5);
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: var(--sp-4);
    align-items: start;
  }
  .cell {
    display: grid;
    gap: var(--sp-2);
    min-width: 0;
  }
  .cell.wide {
    grid-column: 1 / -1;
    max-width: 860px;
  }
  .lbl {
    font: 500 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }

  /* .row is a bare flex util. This .t-chip is kept inline (anchor + 9px trailing
     icon + link hover) — see the note at its markup above. */
  .row {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }
  .t-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
  }
  a.t-chip:hover {
    border-color: var(--acc);
    color: var(--acc);
    text-decoration: none;
  }
</style>
