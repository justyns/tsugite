<script module lang="ts">
  /** Effect runs per tab id keyed on the `params` object identity, so a test can
   *  tell whether a layout change handed a surface a new params object. */
  export const paramRuns = new Map<string, number>();
</script>

<script lang="ts">
  let { id, params }: { id: string; params: Record<string, string> } = $props();

  $effect(() => {
    void params;
    paramRuns.set(id, (paramRuns.get(id) ?? 0) + 1);
  });
</script>
