<script lang="ts">
  // First-run / re-auth token entry, built on the error pane
  // (.t-pane--err). Shown whenever the auth store is gated - no stored token on
  // cold start, or a 401 that flipped `requireAuth`. Saving persists to
  // localStorage (tsugite_token) via the auth store, which clears the gate.
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { auth } from '$lib/stores/auth.svelte';
  import { TESTID } from '$lib/testids';

  let token = $state('');

  function connect(event?: Event) {
    event?.preventDefault();
    const value = token.trim();
    if (value) auth.save(value);
  }
</script>

<main class="gate" data-testid={TESTID.authGate}>
  <PaneState kind="error" title="Access token required">
    {#snippet icon()}<Icon name="lock" />{/snippet}
    {#snippet hint()}{auth.gateReason || 'Enter your tsugite daemon token to connect.'}{/snippet}
    {#snippet actions()}
      <form class="tokenform" onsubmit={connect}>
        <Input
          type="password"
          bind:value={token}
          placeholder="access token"
          ariaLabel="Access token"
          id={TESTID.tokenInput}
        />
        <Button variant="pri" type="submit" data-testid={TESTID.tokenConnect}>Connect</Button>
      </form>
    {/snippet}
  </PaneState>
</main>

<style>
  .gate {
    min-height: 100vh;
    display: grid;
    place-items: center;
    padding: var(--sp-5);
  }
  .tokenform {
    display: flex;
    gap: var(--sp-2);
    width: min(320px, 80vw);
  }
</style>
