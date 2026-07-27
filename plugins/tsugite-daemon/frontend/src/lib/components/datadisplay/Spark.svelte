<script lang="ts">
  // CSS-only sparkline bar chart (e.g. last N schedule runs). Heights are
  // caller-computed pixels -
  // this primitive only maps status to color, it doesn't scale data itself.
  export type SparkStatus = 'ok' | 'fail' | 'skip';
  export type SparkPoint = { height: number; status?: SparkStatus };

  let { points, label }: { points: SparkPoint[]; label: string } = $props();
</script>

<span class="t-spark" role="img" aria-label={label}>
  {#each points as point, i (i)}
    <i
      style="--h:{point.height}px"
      class:f={point.status === 'fail'}
      class:s={point.status === 'skip'}
    ></i>
  {/each}
</span>

<style>
  .t-spark {
    display: inline-flex;
    gap: 2px;
    align-items: flex-end;
    height: 15px;
  }
  .t-spark i {
    height: var(--h, 0px);
    width: 3px;
    border-radius: 1px;
    background: var(--st-ok);
    opacity: 0.85;
  }
  .t-spark i.f {
    background: var(--st-err);
  }
  .t-spark i.s {
    background: var(--bg4);
  }
</style>
