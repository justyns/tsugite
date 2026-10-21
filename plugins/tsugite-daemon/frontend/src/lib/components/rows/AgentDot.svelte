<script lang="ts">
  let { color, name }: { color: string | null; name?: string } = $props();
</script>

<span class="agent-dot" data-color={color} title={name} aria-hidden="true"></span>

<style>
  .agent-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--c, var(--tx3));
    flex: none;
    display: inline-block;
  }
  .agent-dot[data-color='blue'] {
    --c: var(--acc);
  }
  .agent-dot[data-color='green'] {
    --c: var(--st-ok);
  }
  .agent-dot[data-color='red'] {
    --c: var(--st-err);
  }
  .agent-dot[data-color='orange'] {
    --c: var(--st-warn);
  }
  .agent-dot[data-color='teal'] {
    --c: var(--st-verify);
  }
  .agent-dot[data-color='purple'] {
    --c: var(--hue-purple);
  }
  .agent-dot[data-color='yellow'] {
    --c: var(--hue-yellow);
  }
  .agent-dot[data-color='pink'] {
    --c: var(--hue-pink);
  }
</style>
