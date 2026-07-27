<script lang="ts">
  // One-shot run launcher: a freeform prompt bound to a registered agent, opened
  // as a new chat surface. Agents declare no typed-input schema today, so the
  // launcher is always a single freeform prompt (plus an optional effort
  // override, which maps to the chat stream's reasoning_effort). Model override
  // is intentionally omitted - the chat send path carries no model field.
  import Modal from '$lib/components/overlays/Modal.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  let {
    open = false,
    agentName,
    agentDescription,
    defaultEffort,
    onLaunch,
    onClose,
  }: {
    open?: boolean;
    agentName: string;
    agentDescription?: string;
    /** The agent's declared reasoning_effort, shown as the pre-selected override. */
    defaultEffort?: string;
    onLaunch: (opts: { prompt: string; effort?: string }) => void;
    onClose: () => void;
  } = $props();

  const EFFORTS = ['default', 'low', 'medium', 'high'];

  let prompt = $state('');
  let effort = $state('default');

  // Reset the form each time the launcher opens for a (possibly different) agent.
  let wasOpen = false;
  $effect(() => {
    if (open && !wasOpen) {
      prompt = '';
      effort = defaultEffort && EFFORTS.includes(defaultEffort) ? defaultEffort : 'default';
    }
    wasOpen = open;
  });

  const canLaunch = $derived(prompt.trim().length > 0);
  const initial = $derived(agentName.slice(0, 1).toUpperCase());

  function launch() {
    if (!canLaunch) return;
    onLaunch({ prompt: prompt.trim(), effort: effort === 'default' ? undefined : effort });
  }

  function onPromptKeydown(e: KeyboardEvent) {
    // Cmd/Ctrl+Enter sends, matching the composer's send affordance.
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      launch();
    }
  }
</script>

<Modal {open} onclose={onClose} title={`Run ${agentName}`}>
  <div class="rl">
    <div class="rl-agent">
      <span class="rl-av" aria-hidden="true">{initial}</span>
      <div class="rl-meta">
        <div class="rl-name mono">{agentName}</div>
        {#if agentDescription}<div class="rl-desc">{agentDescription}</div>{/if}
      </div>
    </div>

    <label class="rl-field">
      <span class="rl-lbl">prompt <span class="req" aria-hidden="true">*</span></span>
      <textarea
        class="t-input"
        rows="4"
        placeholder="What should {agentName} do?"
        bind:value={prompt}
        onkeydown={onPromptKeydown}
        data-autofocus
        aria-label="Run prompt"></textarea>
    </label>

    <div class="rl-field rl-row">
      <span class="rl-lbl">effort</span>
      <Seg options={EFFORTS} bind:value={effort} ariaLabel="Reasoning effort override" />
      <span class="rl-hint"
        >{effort === 'default' ? "the agent's own setting" : 'override for this run'}</span
      >
    </div>
  </div>

  {#snippet footer()}
    <Button onclick={onClose}>Cancel</Button>
    <Button variant="pri" disabled={!canLaunch} onclick={launch}>
      {#snippet icon()}<Icon name="sparkle" />{/snippet}
      Start chat
    </Button>
  {/snippet}
</Modal>

<style>
  .rl {
    display: grid;
    gap: 13px;
    min-width: min(390px, 72vw);
  }
  .rl-agent {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 11px;
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    background: var(--bg1);
  }
  .rl-av {
    width: 28px;
    height: 28px;
    border-radius: var(--r-sm);
    display: grid;
    place-items: center;
    color: var(--on-acc);
    background: var(--acc);
    font: 700 var(--fs-sm) var(--font-ui);
    flex: none;
  }
  .rl-name {
    font: 600 var(--fs-sm) var(--font-mono);
    color: var(--tx0);
  }
  .rl-desc {
    font-size: var(--fs-xs);
    color: var(--tx2);
    margin-top: 1px;
  }
  .rl-field {
    display: grid;
    gap: 5px;
  }
  .rl-row {
    grid-auto-flow: column;
    grid-template-columns: auto auto 1fr;
    align-items: center;
    gap: 9px;
  }
  .rl-lbl {
    font: 600 var(--fs-xs) var(--font-ui);
    color: var(--tx1);
  }
  .req {
    color: var(--st-warn);
  }
  .rl-hint {
    font-size: var(--fs-2xs);
    color: var(--tx3);
  }
  /* Themed input treatment (no global rule exists). */
  .rl textarea.t-input {
    resize: vertical;
    width: 100%;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 6px 9px;
    color: var(--tx0);
    font: 400 var(--fs-md) / 1.5 var(--font-ui);
  }
  .rl textarea.t-input::placeholder {
    color: var(--tx3);
  }
  .rl textarea.t-input:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
  .mono {
    font-family: var(--font-mono);
  }
</style>
