<script lang="ts">
  import { rafThrottle } from './rafThrottle';
  import { parseMarkdown } from './chatturns.util';

  let {
    content = '',
    breaks = false,
  }: {
    // Markdown source. Rendered to HTML with `marked`; raw inline HTML in the
    // source (e.g. `<span class="math">`) passes through.
    content?: string;
    // Render soft line breaks as hard ones. Set for the person's own turns.
    breaks?: boolean;
  } = $props();

  // NOTE: content originates from the trusted daemon/agent stream. If untrusted
  // markdown is ever rendered here, sanitize before this point - this component
  // deliberately owns no sanitization policy.

  // Seed synchronously so first paint is never empty. During a live token stream
  // `content` grows every few ms; re-parsing the whole string each time is
  // superlinear, so deltas coalesce to one parse per frame with the trailing
  // (final) value guaranteed to render.
  // svelte-ignore state_referenced_locally -- the props seed the first parse; the effect owns updates.
  let html = $state(parseMarkdown(content, breaks));
  const throttle = rafThrottle<{ src: string; breaks: boolean }>(
    (next) => (html = parseMarkdown(next.src, next.breaks)),
  );

  let seeded = false;
  $effect(() => {
    // Reads `breaks` too, so flipping it re-parses bubbles already on screen.
    const next = { src: content, breaks };
    if (seeded) throttle.push(next);
    else seeded = true;
  });
  $effect(() => () => throttle.dispose());
</script>

<div class="t-prose">{@html html}</div>

<style>
  /* Descendant rules use :global() because marked output is injected via {@html}
     and never carries this component's scope hash. */
  .t-prose {
    font-size: var(--fs-md);
    line-height: 1.6;
    color: var(--tx1);
    max-width: 76ch;
    text-wrap: pretty;
    /* Long unbreakable tokens (URLs, hashes) must break rather than force the
       whole turn wider than a narrow pane; `anywhere` also lets the block shrink
       to its container so the conversation never scrolls horizontally. */
    overflow-wrap: anywhere;
  }
  .t-prose :global(p) {
    margin: 0;
  }
  .t-prose :global(p + p) {
    margin-top: 8px;
  }
  .t-prose :global(strong) {
    color: var(--tx0);
    font-weight: 600;
  }
  .t-prose :global(em) {
    font-style: italic;
  }
  .t-prose :global(code) {
    font: 500 var(--fs-sm) var(--font-mono);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    padding: 0 4px;
    border-radius: 4px;
    color: var(--tx0);
    /* No `white-space: nowrap` here: a long inline token (path, hash) must be
       free to break with the inherited overflow-wrap, or it widens the whole
       pane past a narrow viewport (the conversation never scrolls sideways). */
  }
  /* Fenced code blocks: marked emits <pre><code>, and the inline-code rule above
     (nowrap, per-token border) is wrong for a block. Give the block the .t-code
     container language and reset the inner <code> to plain scrollable mono. */
  .t-prose :global(pre) {
    margin: 8px 0;
    padding: 9px 11px;
    background: var(--bg1);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow-x: auto;
    max-height: 260px;
    overflow-y: auto;
    tab-size: 2;
  }
  .t-prose :global(pre code) {
    background: none;
    border: 0;
    padding: 0;
    border-radius: 0;
    white-space: pre;
    font: 400 var(--fs-sm) / 1.65 var(--font-mono);
    color: var(--tx1);
  }
  .t-prose :global(ul) {
    margin: 4px 0;
    padding-left: 18px;
  }
  .t-prose :global(li) {
    margin: 3px 0;
  }
  .t-prose :global(.math) {
    font-family: var(--font-mono);
    font-style: italic;
    color: var(--tx0);
    background: color-mix(in oklab, var(--brand) 8%, transparent);
    padding: 0 4px;
    border-radius: 4px;
    /* No nowrap, same as inline code: a long span must break, not widen the pane. */
  }
</style>
