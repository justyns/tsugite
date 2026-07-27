<script lang="ts">
  // Dot.
  // The 8px compressed form of the state language for tight rows. Per the
  // state-language rule ("never signaled by color alone"), a Dot is either
  // paired with adjacent visible text (default: decorative, aria-hidden) or,
  // when used standalone, given a `label` for its own accessible name.
  import type { DotColor } from './dot-colors';

  let {
    color = 'mute',
    /** Animated halo - "live/streaming" activity. */
    pulse = false,
    /** Outlined instead of filled - "idle". */
    ring = false,
    /** Accessible name for a standalone dot with no adjacent visible text
     * (e.g. "idle"). Omit when visible text already sits next to it. */
    label,
  }: {
    color?: DotColor;
    pulse?: boolean;
    ring?: boolean;
    label?: string;
  } = $props();
</script>

<span
  class="t-dot"
  class:t-dot--ring={ring}
  class:t-dot--pulse={pulse}
  data-color={color}
  role={label ? 'img' : undefined}
  aria-label={label}
  aria-hidden={label ? undefined : true}
></span>

<style>
  /* ---- dots; the DotColor enum maps
     to the semantic state tokens here, one rule per color ---- */
  .t-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--c, var(--st-mute));
    flex: none;
    display: inline-block;
  }
  .t-dot[data-color='ok'] {
    --c: var(--st-ok);
  }
  .t-dot[data-color='verify'] {
    --c: var(--st-verify);
  }
  .t-dot[data-color='warn'] {
    --c: var(--st-warn);
  }
  .t-dot[data-color='err'] {
    --c: var(--st-err);
  }
  .t-dot[data-color='info'] {
    --c: var(--st-info);
  }
  .t-dot[data-color='queue'] {
    --c: var(--st-queue);
  }
  .t-dot[data-color='mute'] {
    --c: var(--st-mute);
  }
  .t-dot--ring {
    background: transparent;
    border: 1.5px solid var(--c, var(--st-mute));
  }
  .t-dot--pulse {
    animation: tpulse 1.6s var(--ease) infinite;
  }
  @keyframes tpulse {
    0%,
    100% {
      box-shadow: 0 0 0 0 color-mix(in oklab, var(--c, var(--acc)) 45%, transparent);
    }
    55% {
      box-shadow: 0 0 0 5px transparent;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .t-dot--pulse {
      animation: none;
    }
  }
</style>
