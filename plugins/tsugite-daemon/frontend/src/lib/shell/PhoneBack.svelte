<script lang="ts">
  // Shared phone-drilldown back affordance. At phone width each workspace content
  // screen (a chat, a terminal, a file) leads its header with this control to
  // return to the rail/list; hidden on desktop/tablet, where the rail sits beside
  // the content. The lone left chevron reuses chev-r rotated (no chev-l in the set),
  // matching the rail collapse control.
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';

  let { onBack, label = 'Back' }: { onBack: () => void; label?: string } = $props();
</script>

<span class="phone-back">
  <Button
    variant="ghost"
    size="sm"
    iconOnly
    aria-label={label}
    data-testid={TESTID.phoneBack}
    onclick={onBack}
  >
    {#snippet icon()}<Icon name="chev-r" />{/snippet}
  </Button>
</span>

<style>
  .phone-back {
    display: none;
  }
  .phone-back :global(.ic) {
    rotate: 180deg;
  }
  @media (max-width: 640px) {
    .phone-back {
      display: inline-flex;
    }
  }
</style>
