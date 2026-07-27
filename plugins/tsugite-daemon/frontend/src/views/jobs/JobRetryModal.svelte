<script lang="ts">
  // Retry-with-hint dialog. The backend accepts hint and/or model (400 without
  // at least one), plus reset_counter and fresh_workspace flags - all wired here.
  import Modal from '$lib/components/overlays/Modal.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Field from '$lib/components/inputs/Field.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Switch from '$lib/components/inputs/Switch.svelte';
  import { TESTID } from '$lib/testids';
  import type { JobRetryOpts } from '$lib/stores/jobs.svelte';

  let {
    open = false,
    prompt = '',
    onClose,
    onSubmit,
  }: {
    open?: boolean;
    /** The job's prompt, shown for context. */
    prompt?: string;
    onClose: () => void;
    onSubmit: (opts: JobRetryOpts) => void;
  } = $props();

  let hint = $state('');
  let model = $state('');
  let resetCounter = $state(false);
  let freshWorkspace = $state(false);

  const uid = $props.id();
  const hintId = `${uid}-hint`;
  const modelId = `${uid}-model`;
  const canSubmit = $derived(hint.trim() !== '' || model.trim() !== '');

  // Reset the form whenever the dialog reopens.
  $effect(() => {
    if (open) {
      hint = '';
      model = '';
      resetCounter = false;
      freshWorkspace = false;
    }
  });

  function submit() {
    if (!canSubmit) return;
    const opts: JobRetryOpts = {};
    if (hint.trim()) opts.hint = hint.trim();
    if (model.trim()) opts.model = model.trim();
    if (resetCounter) opts.resetCounter = true;
    if (freshWorkspace) opts.freshWorkspace = true;
    onSubmit(opts);
  }
</script>

<Modal {open} title="Retry job" onclose={onClose}>
  <div class="retry-form" data-testid={TESTID.jobRetryDrawer}>
    <p class="ctx">{prompt}</p>
    <Field
      id={hintId}
      label="hint"
      hint="Steer the worker's next attempt (used as its new prompt)."
    >
      {#snippet children(describedBy)}
        <textarea
          id={hintId}
          class="t-input"
          rows="3"
          placeholder="e.g. the disk check must also cover weekly snapshots older than 90d"
          aria-describedby={describedBy}
          data-testid={TESTID.jobRetryHint}
          data-autofocus
          bind:value={hint}></textarea>
      {/snippet}
    </Field>
    <Field
      id={modelId}
      label="model override"
      hint="Optional. Leave blank to keep the job's model."
    >
      {#snippet children(describedBy)}
        <Input
          id={modelId}
          mono
          bind:value={model}
          ariaDescribedby={describedBy}
          placeholder="claude_code:opus"
        />
      {/snippet}
    </Field>
    <div class="toggle">
      <Switch bind:checked={resetCounter} ariaLabel="Reset the attempt counter to 1" />
      <span class="tg-lb"
        >reset attempts to 1<span class="tg-h">otherwise continues the count</span></span
      >
    </div>
    <div class="toggle">
      <Switch bind:checked={freshWorkspace} ariaLabel="Retry in a fresh workspace checkout" />
      <span class="tg-lb"
        >fresh workspace<span class="tg-h">otherwise keeps the last checkout</span></span
      >
    </div>
  </div>
  {#snippet footer()}
    <Button variant="ghost" size="sm" onclick={onClose}>Cancel</Button>
    <Button
      variant="pri"
      size="sm"
      disabled={!canSubmit}
      data-testid={TESTID.jobRetrySubmit}
      onclick={submit}
    >
      Retry
    </Button>
  {/snippet}
</Modal>

<style>
  .retry-form {
    display: grid;
    gap: 12px;
    text-align: left;
  }
  .ctx {
    margin: 0;
    font: 400 var(--fs-sm) / 1.4 var(--font-ui);
    color: var(--tx2);
  }
  .t-input {
    width: 100%;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    color: var(--tx0);
    font: 400 var(--fs-md) var(--font-ui);
    height: auto;
    min-height: 34px;
    padding: 7px 9px;
    resize: none;
    line-height: 1.5;
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
  .toggle {
    display: flex;
    align-items: flex-start;
    gap: 9px;
  }
  .tg-lb {
    font: 500 var(--fs-sm) var(--font-ui);
    color: var(--tx1);
    display: grid;
    gap: 1px;
  }
  .tg-h {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
</style>
