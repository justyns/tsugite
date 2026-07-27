<script lang="ts">
  // Acceptance-criteria tally: ✓pass ✕fail ·remaining. The glyphs (not just
  // colour) distinguish the three, and an aria-label spells the tally out.
  import type { AcCounts } from './jobModel';

  let { counts, showWord = false }: { counts: AcCounts; showWord?: boolean } = $props();

  const aria = $derived(
    [
      counts.pass ? `${counts.pass} passed` : '',
      counts.fail ? `${counts.fail} failed` : '',
      counts.remaining ? `${counts.remaining} remaining` : '',
    ]
      .filter(Boolean)
      .join(', ') || `${counts.total} criteria`,
  );
</script>

{#if counts.total > 0}
  <span class="crit" aria-label={`acceptance criteria: ${aria}`}>
    {#if counts.pass > 0}<span class="p">✓{counts.pass}</span>{/if}
    {#if counts.fail > 0}<span class="f">✕{counts.fail}</span>{/if}
    {#if counts.remaining > 0}<span class="r"
        >·{counts.remaining}{#if showWord}&nbsp;criteria{/if}</span
      >{/if}
  </span>
{/if}

<style>
  .crit {
    font: 500 var(--fs-2xs) var(--font-mono);
    display: inline-flex;
    gap: 6px;
  }
  .crit .p {
    color: var(--st-ok);
  }
  .crit .f {
    color: var(--st-err);
  }
  .crit .r {
    color: var(--tx3);
  }
</style>
