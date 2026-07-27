<script lang="ts" module>
  export type SetSecretPayload = { name: string; value: string };
</script>

<script lang="ts">
  // Set-secret modal with add and rotate modes. Add-mode enables the name
  // field; rotate-mode locks it. Renders the `.t-modal` card only - no
  // `.t-scrim` backdrop - leaving the overlay/mount decision to whoever opens
  // the modal.
  import { untrack } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import { trapFocus } from '$lib/actions/trapFocus';

  let {
    mode,
    name = '',
    onCancel,
    onSave,
  }: {
    mode: 'add' | 'rotate';
    /** current secret name - required (and read-only) for mode="rotate" */
    name?: string;
    onCancel?: () => void;
    onSave?: (payload: SetSecretPayload) => void;
  } = $props();

  const uid = $props.id();
  // Seeded once from the prop, then locally editable - not kept in sync, so
  // read outside Svelte's reactive tracking on purpose.
  let nameValue = $state(untrack(() => name));
  let value = $state('');
  let showValue = $state(false);
  let nameEl = $state<HTMLInputElement>();
  let valueEl = $state<HTMLInputElement>();

  $effect(() => {
    (mode === 'add' ? nameEl : valueEl)?.focus();
  });

  function submit(e: SubmitEvent) {
    e.preventDefault();
    onSave?.({ name: mode === 'rotate' ? name : nameValue, value });
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      onCancel?.();
    }
  }
</script>

<div
  class="t-modal t-modal--wide"
  role="dialog"
  aria-modal="true"
  aria-labelledby="{uid}-title"
  tabindex="-1"
  onkeydown={handleKeydown}
  use:trapFocus
>
  <h3 id="{uid}-title">
    <Icon name="key" />
    {mode === 'rotate' ? `Rotate ${name}` : 'Add secret'}
  </h3>
  <form class="mform-shell" onsubmit={submit}>
    <div class="mform">
      <div class="t-field">
        <label for="{uid}-name">name</label>
        {#if mode === 'rotate'}
          <input id="{uid}-name" class="t-input mono" value={name} readonly />
          <span class="hint">read-only when rotating — the old value is destroyed on save</span>
        {:else}
          <input
            id="{uid}-name"
            class="t-input mono"
            placeholder="SECRET_NAME"
            required
            bind:value={nameValue}
            bind:this={nameEl}
          />
          <span class="hint">editable now — locked once the secret exists</span>
        {/if}
      </div>
      <div class="t-field">
        <label for="{uid}-value">value</label>
        <div class="value-row">
          <input
            id="{uid}-value"
            class="t-input mono"
            type={showValue ? 'text' : 'password'}
            placeholder="paste value…"
            autocomplete="off"
            required
            bind:value
            bind:this={valueEl}
          />
          <Button size="sm" aria-pressed={showValue} onclick={() => (showValue = !showValue)}>
            {showValue ? 'hide' : 'show'}
          </Button>
        </div>
        <span class="hint"
          >Saved to the configured secrets backend. Visible while you type — never read back
          afterward.</span
        >
      </div>
    </div>
    <div class="fx">
      <Button onclick={() => onCancel?.()}>Cancel</Button>
      <Button variant="pri" type="submit">
        {#snippet icon()}
          {#if mode === 'rotate'}<Icon name="retry" />{:else}<Icon name="check" />{/if}
        {/snippet}
        {mode === 'rotate' ? 'Rotate value' : 'Add secret'}
      </Button>
    </div>
  </form>
</div>

<style>
  .mono {
    font-family: var(--font-mono);
  }
  .t-modal {
    width: min(430px, 100%);
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-3);
    padding: 16px;
    display: grid;
    gap: 12px;
  }
  .t-modal h3 {
    margin: 0;
    font: 600 var(--fs-lg)/1.3 var(--font-ui);
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .t-modal h3 :global(.ic) {
    color: var(--acc);
  }
  .t-modal--wide {
    width: min(520px, 100%);
  }
  .mform-shell {
    display: contents;
  }
  .t-modal .mform {
    display: grid;
    gap: 11px;
    text-align: left;
  }
  .t-modal .mform .hint {
    font-size: var(--fs-2xs);
    color: var(--tx3);
    line-height: 1.5;
  }
  .t-modal .fx {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
  }
  .value-row {
    display: flex;
    gap: 6px;
  }
  .value-row .t-input {
    flex: 1;
  }

  /* t-field/t-input kept inline: the name input toggles readonly, and the
     value input swaps type + needs a bind:this focus ref (plus required and
     autocomplete) — neither of which the shared Field/Input model. */
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
