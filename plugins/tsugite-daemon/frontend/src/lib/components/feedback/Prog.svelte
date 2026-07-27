<script lang="ts">
  // Progress bar - determinate and indeterminate.
  // Omitting `value` renders the indeterminate sweep; passing it renders a real percentage.
  let {
    value,
    label,
  }: {
    value?: number;
    label: string;
  } = $props();

  const pct = $derived(value === undefined ? undefined : Math.min(100, Math.max(0, value)));
  const ariaNow = $derived(pct === undefined ? undefined : Math.round(pct));
</script>

<span
  class="t-prog"
  class:t-prog--ind={pct === undefined}
  role="progressbar"
  aria-label={label}
  aria-valuemin={pct === undefined ? undefined : 0}
  aria-valuemax={pct === undefined ? undefined : 100}
  aria-valuenow={ariaNow}
>
  <i style={pct === undefined ? undefined : `--w:${pct}%`}></i>
</span>

<style>
  .t-prog {
    height: 3px;
    background: var(--bg3);
    border-radius: var(--r-full);
    overflow: hidden;
    width: 100%;
  }
  .t-prog i {
    display: block;
    width: var(--w, 0%);
    height: 100%;
    background: var(--acc);
    border-radius: inherit;
    transition: width var(--t-3) var(--ease);
  }
  .t-prog--ind i {
    width: 38%;
    animation: tind 1.3s var(--ease) infinite;
  }
  @keyframes tind {
    0% {
      margin-left: -40%;
    }
    100% {
      margin-left: 104%;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .t-prog--ind i {
      animation-duration: 0.01ms;
    }
  }
</style>
