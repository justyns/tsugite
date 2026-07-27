<script lang="ts">
  // Secrets store lock gate.
  // Only variant: locked, ready to submit a passphrase.
  // No busy/error props - those are speculative;
  // a real Secrets view can layer its own feedback around this callback.
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';

  let {
    onUnlock,
  }: {
    onUnlock?: (passphrase: string) => void;
  } = $props();

  const uid = $props.id();
  let passphrase = $state('');

  function submit(e: SubmitEvent) {
    e.preventDefault();
    onUnlock?.(passphrase);
  }
</script>

<form class="lock-card" onsubmit={submit}>
  <div class="lock-badge"><Icon name="lock" size={20} /></div>
  <h4>Store is locked</h4>
  <p class="sub">
    Encrypted at rest (age · x25519). A passphrase decrypts it into memory for the session only.
  </p>
  <div class="t-field">
    <label for="{uid}-passphrase">passphrase</label>
    <input
      id="{uid}-passphrase"
      class="t-input"
      type="password"
      placeholder="••••••••"
      autocomplete="off"
      required
      bind:value={passphrase}
    />
  </div>
  <Button variant="pri" type="submit">
    {#snippet icon()}<Icon name="lock" />{/snippet}Unlock store
  </Button>
</form>

<style>
  .lock-card {
    width: min(340px, 100%);
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-2);
    padding: 18px;
    display: grid;
    gap: 11px;
    justify-items: center;
    text-align: center;
  }
  .lock-badge {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: color-mix(in oklab, var(--acc) 15%, var(--bg2));
    color: var(--acc);
  }
  .lock-card h4 {
    margin: 0;
    font: 600 var(--fs-md) var(--font-ui);
  }
  .lock-card .sub {
    margin: 0;
    font-size: var(--fs-sm);
    color: var(--tx2);
    line-height: 1.5;
  }
  .lock-card .t-field {
    width: 100%;
  }
  /* :global(button) reaches the <button> that <Button> renders - it carries
     Button's scope hash, not this card's, so a plain descendant would miss it. */
  .lock-card > :global(button) {
    width: 100%;
  }

  /* t-field/t-input kept inline: the passphrase input needs `required` and
     `autocomplete="off"`, which the shared Input.svelte doesn't accept. */
  .t-field {
    display: grid;
    gap: 5px;
  }
  .t-field label {
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  .t-input {
    height: 28px;
    width: 100%;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 0 9px;
    color: var(--tx0);
    font: 400 var(--fs-md) var(--font-ui);
    transition:
      border-color var(--t-1),
      box-shadow var(--t-1);
  }
  .t-input::placeholder {
    color: var(--tx3);
  }
  .t-input:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
</style>
