<script lang="ts">
  import type { Component } from 'svelte';
  import { TESTID } from '$lib/testids';

  // Auto-discover component demos: drop a `*.gallery.svelte` anywhere under
  // lib/components and it shows up here with zero shared-file edits. Lazy glob
  // (no `eager`) so demos are code-split out of the entry bundle and only
  // fetched when the gallery view mounts. Renders nothing when no demos exist.
  const modules = import.meta.glob<{ default: Component }>(
    '../../lib/components/**/*.gallery.svelte',
  );
  const demos = Object.entries(modules)
    .map(([path, load]) => ({
      name: path.split('/').pop()!.replace('.gallery.svelte', ''),
      promise: load(),
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
</script>

<section data-testid={TESTID.view('gallery')}>
  <h2>Gallery</h2>
  <div class="grid" data-testid={TESTID.gallery}>
    {#if demos.length === 0}
      <p>No component demos yet.</p>
    {/if}
    {#each demos as demo (demo.name)}
      <section class="demo">
        <h3>{demo.name}</h3>
        {#await demo.promise then mod}
          {@const Demo = mod.default}
          <Demo />
        {:catch err}
          <p>failed to load: {err.message}</p>
        {/await}
      </section>
    {/each}
  </div>
</section>

<style>
  .grid {
    display: grid;
    gap: var(--sp-5);
    margin-top: var(--sp-4);
  }
  .demo h3 {
    margin: 0 0 var(--sp-2);
    font-family: var(--font-mono);
    font-size: var(--fs-xs);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
</style>
