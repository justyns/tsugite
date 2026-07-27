<script lang="ts">
  import Scrim from '$lib/components/overlays/Scrim.svelte';
  import Modal from '$lib/components/overlays/Modal.svelte';
  import Drawer from '$lib/components/overlays/Drawer.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  // Specimens render already-open inside bounded ".stage" boxes. Each stage has
  // a transform, which makes it the containing block for the modal's
  // position:fixed scrim (and the drawer's position:absolute), so the overlays
  // are clipped to their box instead of taking over the viewport. Mounting
  // already-open is intentionally focus-neutral (see the components' $effect).
</script>

<section data-testid="gallery-overlays">
  <div class="g-group">
    <h4 class="g-sub">Scrim — shared backdrop</h4>
    <p class="g-note">
      Dims + blurs the pane, centers its child, and dismisses on a backdrop click.
    </p>
    <div class="g-row">
      <div class="g-cell">
        <div class="stage">
          <Scrim open={true}>
            <div class="scrim-card">
              <b>Scrim</b>
              <span>backdrop click dismisses · the dialog owns Esc</span>
            </div>
          </Scrim>
        </div>
        <div class="lbl">open</div>
      </div>
    </div>
  </div>

  <div class="g-group">
    <h4 class="g-sub">Modal — decision dialog</h4>
    <p class="g-note">
      Focus is trapped, <span class="k">Esc</span> cancels, initial focus lands on the safe action, and
      the destructive verb is spelled out — never “OK”.
    </p>
    <div class="g-row">
      <div class="g-cell">
        <div class="stage">
          <Modal open={true} title="Cancel job?" tone="danger">
            <code>fix flaky sse reconnect test</code> is mid-attempt 2. Cancelling stops the worker,
            keeps the branch, and moves the job to <b>cancelled</b>. This can’t be resumed.
            {#snippet footer()}
              <button type="button" class="t-btn" data-autofocus>Keep running</button>
              <button type="button" class="t-btn t-btn--danger">Cancel job</button>
            {/snippet}
          </Modal>
        </div>
        <div class="lbl">tone: danger</div>
      </div>
      <div class="g-cell">
        <div class="stage">
          <Modal open={true} title="Discard draft?">
            Your unsaved changes to <code>ops-runner.md</code> will be lost.
            {#snippet footer()}
              <button type="button" class="t-btn" data-autofocus>Keep editing</button>
              <button type="button" class="t-btn t-btn--pri">Discard</button>
            {/snippet}
          </Modal>
        </div>
        <div class="lbl">tone: default</div>
      </div>
    </div>
  </div>

  <div class="g-group">
    <h4 class="g-sub">Drawer — right-side inspection panel</h4>
    <p class="g-note">
      Non-modal (tab back to the page), <span class="k">Esc</span> closes, <code>inert</code> while off-screen.
      Shares the scrim + motion tokens but ships no scrim of its own — it slides in over a positioned
      pane.
    </p>
    <div class="g-row">
      <div class="g-cell">
        <div class="stage stage--tall">
          <Drawer open={true} title="nightly backup prune policy" label="Job detail">
            {#snippet status()}
              <span class="t-pill" data-st="awaiting"><Icon name="q" />awaiting input</span>
            {/snippet}
            <div class="d-sec">
              <h4>acceptance criteria · 2/5</h4>
              <p class="d-line">backup completes in under 10 minutes</p>
              <p class="d-line">disk usage below 80% after prune</p>
            </div>
            <div class="d-sec">
              <h4>links</h4>
              <p class="d-line">session · pty 3 · worktree jobs/4c9f</p>
            </div>
            {#snippet footer()}
              <button type="button" class="t-btn t-btn--pri t-btn--sm">Answer question</button>
              <button type="button" class="t-btn t-btn--sm"><Icon name="retry" />Retry</button>
              <span class="grow"></span>
              <button type="button" class="t-btn t-btn--sm t-btn--danger">Cancel</button>
              <button type="button" class="t-btn t-btn--sm t-btn--ghost">Dismiss</button>
            {/snippet}
          </Drawer>
        </div>
        <div class="lbl">job detail · awaiting</div>
      </div>
      <div class="g-cell">
        <div class="stage stage--tall">
          <Drawer open={true} title="nightly-backup" label="Schedule detail">
            {#snippet status()}
              <span class="t-pill" data-st="errored"><Icon name="x" />errored</span>
            {/snippet}
            <div class="d-sec">
              <h4>recent runs</h4>
              <p class="d-line">errored · 9m 12s · exit 1 — today 03:00</p>
              <p class="d-line">done · 7m 44s — jul 11 03:00</p>
            </div>
            {#snippet footer()}
              <button type="button" class="t-btn t-btn--pri t-btn--sm">Save changes</button>
              <button type="button" class="t-btn t-btn--sm"><Icon name="play" />Run now</button>
              <span class="grow"></span>
              <button type="button" class="t-btn t-btn--sm t-btn--danger">Delete</button>
            {/snippet}
          </Drawer>
        </div>
        <div class="lbl">schedule detail · errored</div>
      </div>
    </div>
  </div>
</section>

<style>
  .g-group {
    margin-bottom: var(--sp-5);
  }
  .g-sub {
    margin: 0 0 4px;
    font: 600 var(--fs-sm) / 1.2 var(--font-ui);
    color: var(--tx1);
  }
  .g-note {
    margin: 0 0 10px;
    font-size: var(--fs-xs);
    line-height: 1.5;
    color: var(--tx3);
    max-width: 70ch;
  }
  .g-note .k {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    padding: 2px 5px 3px;
    border: 1px solid var(--bd1);
    border-bottom-width: 2px;
    border-radius: 4px;
    background: var(--bg2);
    color: var(--tx2);
  }
  .g-note code {
    font-family: var(--font-mono);
    color: var(--tx2);
    background: var(--bg2);
    padding: 0 4px;
    border-radius: 3px;
    font-size: 10.5px;
  }
  .g-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: var(--sp-3);
  }
  .g-cell {
    display: grid;
    gap: 6px;
    min-width: 0;
  }
  .lbl {
    font: 500 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.04em;
    color: var(--tx3);
  }

  /* Bounded stage: the transform makes this the containing block for the
     modal's position:fixed scrim, clipping the overlay to the box. */
  .stage {
    position: relative;
    height: 264px;
    overflow: hidden;
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
    background: var(--bg2);
    transform: translateZ(0);
  }
  .stage--tall {
    height: 400px;
  }

  .scrim-card {
    display: grid;
    gap: 4px;
    justify-items: center;
    text-align: center;
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    padding: 16px 18px;
    color: var(--tx1);
    font-size: var(--fs-sm);
  }
  .scrim-card span {
    color: var(--tx3);
    font-size: var(--fs-xs);
  }

  /* Specimen-content primitives (.grow / .t-btn / .t-pill) kept inline - the
     gallery statically composes them to show the overlays in context and they have
     no shared owner component here. Their glyphs are shared <Icon>s: the base
     sizing/stroke comes from the global .ic (tokens.css); .t-pill's 11px override
     is made :global below so it reaches the child <Icon>'s svg. */
  .grow {
    flex: 1;
    min-width: 0;
  }
  .t-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    height: 28px;
    padding: 0 11px;
    border-radius: var(--r-md);
    border: 1px solid var(--bd1);
    background: var(--bg3);
    color: var(--tx0);
    font: 500 var(--fs-md) / 1 var(--font-ui);
    cursor: pointer;
    white-space: nowrap;
  }
  .t-btn--sm {
    height: 23px;
    padding: 0 8px;
    font-size: var(--fs-sm);
    gap: 5px;
  }
  .t-btn--pri {
    background: var(--acc);
    border-color: transparent;
    color: var(--on-acc);
    font-weight: 600;
  }
  .t-btn--danger {
    background: color-mix(in oklab, var(--st-err) 13%, transparent);
    border-color: color-mix(in oklab, var(--st-err) 38%, transparent);
    color: var(--st-err);
  }
  .t-btn--ghost {
    background: transparent;
    border-color: transparent;
    color: var(--tx1);
  }
  .t-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 8px 0 7px;
    border-radius: var(--r-full);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    letter-spacing: 0.02em;
    white-space: nowrap;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 13%, transparent);
    border: 1px solid color-mix(in oklab, var(--c) 32%, transparent);
  }
  .t-pill :global(.ic) {
    width: 11px;
    height: 11px;
  }
  .t-pill[data-st='awaiting'] {
    --c: var(--st-warn);
  }
  .t-pill[data-st='errored'] {
    --c: var(--st-err);
  }
  .d-sec > h4 {
    margin: 0 0 7px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .d-line {
    margin: 2px 0;
    font-size: var(--fs-sm);
    line-height: 1.5;
    color: var(--tx2);
  }
</style>
