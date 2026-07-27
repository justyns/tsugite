<script lang="ts">
  // Braille spinner.
  // "The working block is the only spinner the chat ever shows" - always pair with text,
  // never render this alone (see Work.svelte). Frame text is decorative; aria-hidden.
  // Frames come from the shared startSpin() driver (same one Button/Pill use), so
  // the 96ms cadence, wrap, and reduced-motion glyph live in exactly one place.
  import { startSpin } from '$lib/components/buttons/spin';

  let { color }: { color?: string } = $props();

  let glyph = $state('⠋');
  $effect(() => startSpin((g) => (glyph = g)));
</script>

<span class="t-spin" style={color ? `--spin-c:${color}` : undefined} aria-hidden="true"
  >{glyph}</span
>

<style>
  .t-spin {
    font-family: var(--font-mono);
    font-weight: 600;
    display: inline-block;
    width: 1.1ch;
    line-height: 1;
    color: var(--spin-c, currentColor);
    flex: none;
  }
</style>
