<script lang="ts">
  // Skeleton loading rows.
  // "Every list renders one of five things: rows, skeleton (loading), empty,
  // error, or permission. There is no blank div in this system." - reusable
  // standalone primitive; a varied-width shimmer, not a repeated block.
  let {
    rows = 4,
    label = 'Loading',
  }: {
    rows?: number;
    label?: string;
  } = $props();

  const WIDTH_PATTERN = [72, 88, 55, 80, 64, 92] as const;
  const widths = $derived(
    Array.from(
      { length: rows },
      (_, i) => WIDTH_PATTERN[i % WIDTH_PATTERN.length] ?? WIDTH_PATTERN[0],
    ),
  );
</script>

<div class="t-skel" role="status" aria-label={label}>
  {#each widths as w, i (i)}
    <i style="--w:{w}%" aria-hidden="true"></i>
  {/each}
</div>

<style>
  .t-skel {
    display: grid;
    gap: 9px;
    padding: 6px 0;
  }
  .t-skel i {
    display: block;
    width: var(--w, 100%);
    height: 11px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--bg3) 25%, var(--bg4) 45%, var(--bg3) 65%);
    background-size: 200% 100%;
    animation: tshimmer 1.4s linear infinite;
  }
  @keyframes tshimmer {
    to {
      background-position: -200% 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .t-skel i {
      animation-duration: 0.01ms;
    }
  }
</style>
