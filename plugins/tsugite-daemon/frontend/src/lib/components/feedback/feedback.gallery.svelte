<script lang="ts">
  import Toast from './Toast.svelte';
  import Toasts from './Toasts.svelte';
  import Spin from './Spin.svelte';
  import Skel from './Skel.svelte';
  import Prog from './Prog.svelte';
  import Work from './Work.svelte';
  import Caret from './Caret.svelte';
  import { toasts } from './toast-store.svelte';

  const noop = () => {};
  const now = Date.now();

  // Seed the shared stack-host demo once (sticky so it doesn't animate away
  // while the gallery is being reviewed/screenshotted).
  $effect(() => {
    if (toasts.items.length === 0) {
      toasts.push('ok', 'Job done', { body: '5/5 criteria passed', sticky: true });
      toasts.push('warn', 'Job needs an answer', {
        body: 'blocked on a retention question.',
        sticky: true,
      });
      toasts.push('info', 'Compaction complete', { body: '74k → 9k tokens.', sticky: true });
    }
  });
</script>

<section data-testid="gallery-feedback" class="ts-gallery">
  <div class="variant">
    <span class="vlabel">toast · variants</span>
    <div class="items">
      <figure>
        <figcaption>ok</figcaption>
        <Toast
          variant="ok"
          title="Job done"
          body="add jitter to reconnect backoff · 5/5 criteria passed"
          sticky
          onDismiss={noop}
        />
      </figure>
      <figure>
        <figcaption>warn</figcaption>
        <Toast
          variant="warn"
          title="Job needs an answer"
          body="nightly backup prune is blocked on a retention question."
          sticky
          onDismiss={noop}
        />
      </figure>
      <figure>
        <figcaption>err (always sticky)</figcaption>
        <Toast
          variant="err"
          title="Schedule failed"
          body="nightly-backup exited 1 — job created and linked."
          onDismiss={noop}
        />
      </figure>
      <figure>
        <figcaption>info</figcaption>
        <Toast
          variant="info"
          title="Compaction complete"
          body="74k → 9k tokens · summary pinned to context."
          sticky
          onDismiss={noop}
        />
      </figure>
      <figure>
        <figcaption>warn · action (card's pinned example)</figcaption>
        <Toast
          variant="warn"
          icon="q"
          title="Job needs an answer"
          body="nightly-backup prune is blocked on a retention question."
          actionLabel="Answer"
          onAction={noop}
          sticky
          onDismiss={noop}
        />
      </figure>
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">toasts · stack host (real store, 3 queued)</span>
    <div class="stack-demo">
      <Toasts />
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">spin</span>
    <div class="items">
      <figure>
        <figcaption>ok</figcaption>
        <Spin color="var(--st-ok)" />
      </figure>
      <figure>
        <figcaption>verify</figcaption>
        <Spin color="var(--st-verify)" />
      </figure>
      <figure>
        <figcaption>muted</figcaption>
        <Spin color="var(--tx2)" />
      </figure>
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">prog</span>
    <div class="items items--col">
      <figure>
        <figcaption>determinate · 64%</figcaption>
        <Prog value={64} label="attempt progress" />
      </figure>
      <figure>
        <figcaption>indeterminate</figcaption>
        <Prog label="working" />
      </figure>
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">caret · token stream</span>
    <div class="items">
      <figure>
        <figcaption>appended to in-flight streamed text</figcaption>
        <span class="caret-demo">{"emit('stale', { age })"}<Caret /></span>
      </figure>
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">work</span>
    <div class="items items--col">
      <figure>
        <figcaption>running</figcaption>
        <Work operation="npm test -w @tsugite/sse" startedAt={now - 7000} onStop={noop} />
      </figure>
      <figure>
        <figcaption>running · progress-detail extension</figcaption>
        <Work
          operation="bash"
          detail="turn 3 · 2 tools · tool: bash"
          startedAt={now - 45000}
          onStop={noop}
        />
      </figure>
      <figure>
        <figcaption>reconnecting</figcaption>
        <Work
          operation="npm test -w @tsugite/sse"
          startedAt={now - 7000}
          reconnecting
          onStop={noop}
        />
      </figure>
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">skel</span>
    <div class="items items--col">
      <figure>
        <figcaption>default · 4 rows</figcaption>
        <div class="framed"><Skel /></div>
      </figure>
      <figure>
        <figcaption>rows=2</figcaption>
        <div class="framed"><Skel rows={2} /></div>
      </figure>
    </div>
  </div>
</section>

<style>
  .ts-gallery {
    display: grid;
    gap: var(--sp-5);
  }
  .variant {
    display: grid;
    gap: var(--sp-2);
  }
  .vlabel {
    font: 600 var(--fs-2xs)/1 var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .items {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-4);
    align-items: start;
  }
  .items--col {
    flex-direction: column;
    align-items: stretch;
    /* Wide enough that Work's widest row (reconnecting: icon + flag +
       operation + elapsed + Stop, ~488px of unwrapped content) still fits on
       one line - prog/skel just stretch their full-width bars to match. */
    max-width: 560px;
  }
  /* Sized for toast cards, which want a readable measure - the col groups
     (prog/work/skel) stretch to fill .items--col's own max-width instead,
     since their content (e.g. Work's operation + elapsed + Stop button) is a
     single row that isn't meant to wrap. */
  .items:not(.items--col) figure {
    max-width: 320px;
  }
  figure {
    margin: 0;
    display: grid;
    gap: var(--sp-2);
  }
  figcaption {
    font: 500 var(--fs-2xs)/1 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .framed {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 10px 12px;
  }
  .caret-demo {
    font-family: var(--font-mono);
    font-size: var(--fs-sm);
    color: var(--tx1);
  }

  /* Real <Toasts> is position:fixed to the viewport corner, which would float
     it over the rest of the page here. Contain it to this box instead by
     giving it a positioned ancestor and overriding its own fixed offsets -
     gallery-only display concern, not a change to the component's contract. */
  .stack-demo {
    position: relative;
    min-height: 190px;
    padding: 8px;
    border: 1px dashed var(--bd1);
    border-radius: var(--r-md);
  }
  .stack-demo :global(.t-toasts) {
    position: absolute;
    inset: auto 8px 8px auto;
    width: min(300px, 100%);
  }
</style>
