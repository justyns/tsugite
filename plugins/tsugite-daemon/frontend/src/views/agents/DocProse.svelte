<script lang="ts">
  // Thin wrapper around the shared chat Prose renderer that adds the
  // document-shaped tokens (headings, ordered lists, blockquote, table, links) -
  // matching how the files wiki renders a full document. `.t-prose` alone only
  // covers short chat-turn markup (paragraphs, code, lists); an agent's prompt
  // body and its resolved system message are full documents and want the rest of
  // the treatment too. Kept local to the agents view (not a `$lib` component)
  // since it's a two-call-site wrapper, not a reusable primitive.
  import Prose from '$lib/components/chatturns/Prose.svelte';

  let { content = '' }: { content?: string } = $props();
</script>

<div class="doc">
  <Prose {content} />
</div>

<style>
  /* Targets Prose's own `.t-prose` wrapper since Prose owns that class. Only adds
     what `.t-prose` doesn't already cover (paragraphs/code/lists/fences are fine). */
  .doc :global(.t-prose h1) {
    font: 600 var(--fs-2xl) / 1.25 var(--font-ui);
    letter-spacing: -0.01em;
    color: var(--tx0);
    margin: 16px 0 10px;
  }
  .doc :global(.t-prose h1:first-child) {
    margin-top: 0;
  }
  .doc :global(.t-prose h2) {
    font: 600 var(--fs-lg) / 1.3 var(--font-ui);
    color: var(--tx0);
    margin: 20px 0 7px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--bd0);
  }
  .doc :global(.t-prose h3) {
    font: 600 var(--fs-md) / 1.3 var(--font-ui);
    color: var(--tx0);
    margin: 16px 0 6px;
  }
  .doc :global(.t-prose ol) {
    margin: 4px 0;
    padding-left: 18px;
  }
  .doc :global(.t-prose blockquote) {
    border-left: 3px solid color-mix(in oklab, var(--st-info) 45%, transparent);
    background: color-mix(in oklab, var(--st-info) 8%, transparent);
    border-radius: var(--r-md);
    padding: 8px 12px;
    margin: 10px 0;
    color: var(--tx1);
  }
  .doc :global(.t-prose blockquote p) {
    margin: 0;
  }
  .doc :global(.t-prose blockquote table) {
    margin: 6px 0 0;
  }
  .doc :global(.t-prose table) {
    border-collapse: collapse;
    margin: 10px 0;
    font-size: var(--fs-sm);
  }
  .doc :global(.t-prose th) {
    text-align: left;
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--tx3);
    border-bottom: 1px solid var(--bd1);
    padding: 5px 12px 5px 0;
  }
  .doc :global(.t-prose td) {
    border-bottom: 1px solid var(--bd0);
    padding: 5px 12px 5px 0;
    color: var(--tx1);
  }
  .doc :global(.t-prose a) {
    color: var(--acc);
  }
</style>
