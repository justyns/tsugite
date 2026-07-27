<script lang="ts">
  // Inline usage meter (e.g. context window tokens): role="meter" bar + a
  // compact mono readout. `warn` styles the readout as a warning;
  // callers decide the threshold, this primitive only renders the state.
  let {
    value,
    max,
    min = 0,
    label,
    displayText,
    warn = false,
  }: {
    value: number;
    max: number;
    min?: number;
    label: string;
    displayText: string;
    warn?: boolean;
  } = $props();

  const pct = $derived(
    max > min ? Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100)) : 0,
  );
</script>

<span
  class="t-meter"
  class:is-warn={warn}
  role="meter"
  aria-valuenow={value}
  aria-valuemin={min}
  aria-valuemax={max}
  aria-valuetext={displayText}
  aria-label={label}
>
  <span class="bar"><i style="--w:{pct}%"></i></span>{displayText}
</span>

<style>
  .t-meter {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .t-meter .bar {
    width: 56px;
    height: 3px;
    background: var(--bg3);
    border-radius: var(--r-full);
    overflow: hidden;
  }
  .t-meter .bar i {
    display: block;
    height: 100%;
    width: var(--w, 0%);
    background: var(--tx3);
  }
  .t-meter.is-warn .bar i {
    background: var(--st-warn);
  }
</style>
