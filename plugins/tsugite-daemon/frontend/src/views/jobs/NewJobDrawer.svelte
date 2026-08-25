<script lang="ts">
  // New-job composer. Emits the form; the view POSTs it to the generic command
  // dispatcher (POST /api/commands/job). No structured create
  // route exists yet: the response is free text with no job_id,
  // so the view reloads the list rather than auto-opening the new tile.
  import Drawer from '$lib/components/overlays/Drawer.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Field from '$lib/components/inputs/Field.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import { TESTID } from '$lib/testids';

  export interface NewJobForm {
    agent: string;
    prompt: string;
    acceptanceCriteria: string[];
    maxAttempts: number;
    executor: string;
    notifyWhen: string;
  }

  let {
    open = false,
    agents,
    executors,
    submitting = false,
    onClose,
    onSubmit,
  }: {
    open?: boolean;
    agents: string[];
    executors: string[];
    submitting?: boolean;
    onClose: () => void;
    onSubmit: (form: NewJobForm) => void;
  } = $props();

  const NOTIFY_LABELS = ['needs you', 'done', 'always', 'never'];
  // Display label -> backend notify_when value (cmd_job: done|stuck|errored|terminal|never).
  const NOTIFY_VALUE: Record<string, string> = {
    'needs you': 'stuck',
    done: 'done',
    always: 'terminal',
    never: 'never',
  };

  let prompt = $state('');
  let acs = $state<string[]>(['']);
  let maxAttempts = $state(3);
  let agent = $state('');
  let executor = $state('agent');
  let notify = $state('needs you');

  const promptId = $props.id();
  const showExecutor = $derived(executors.length > 1);

  // Seed agent once the roster arrives / the drawer opens.
  $effect(() => {
    if (open && !agent && agents.length) agent = agents[0]!;
  });

  const cleanAcs = $derived(acs.map((a) => a.trim()).filter(Boolean));
  const canSubmit = $derived(prompt.trim() !== '' && agent !== '' && !submitting);

  const preview = $derived.by(() => {
    const p = prompt.trim() || '…';
    const short = p.length > 56 ? `${p.slice(0, 56)}…` : p;
    const parts = [`/job "${short}"`, `--agent ${agent || '?'}`, `--max-attempts ${maxAttempts}`];
    if (executor !== 'agent') parts.push(`--executor ${executor}`);
    const nw = NOTIFY_VALUE[notify];
    if (nw && nw !== 'never') parts.push(`--notify-when ${nw}`);
    if (cleanAcs.length) parts.push(`--ac "${cleanAcs.join('|')}"`);
    return parts.join(' ');
  });

  function addAc() {
    acs = [...acs, ''];
  }
  function removeAc(i: number) {
    acs = acs.filter((_, idx) => idx !== i);
    if (acs.length === 0) acs = [''];
  }
  function step(delta: number) {
    maxAttempts = Math.min(9, Math.max(1, maxAttempts + delta));
  }
  function submit() {
    if (!canSubmit) return;
    onSubmit({
      agent,
      prompt: prompt.trim(),
      acceptanceCriteria: cleanAcs,
      maxAttempts,
      executor,
      notifyWhen: NOTIFY_VALUE[notify] ?? 'never',
    });
  }
  function onFormKeydown(e: KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      submit();
    }
  }
</script>

<Drawer {open} title="Spawn a job" label="New job" onclose={onClose}>
  {#snippet status()}
    <span class="newpill"><Icon name="jobs" size={11} />new</span>
  {/snippet}

  <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
  <div class="jf" data-testid={TESTID.newJobDrawer} onkeydown={onFormKeydown} role="form">
    <Field id={promptId} label="prompt">
      {#snippet children()}
        <textarea
          id={promptId}
          class="t-input"
          rows="3"
          placeholder="Describe the task…"
          data-testid={TESTID.newJobPrompt}
          bind:value={prompt}></textarea>
      {/snippet}
    </Field>

    <div class="t-field">
      <span class="lbl"
        >acceptance criteria <span class="sub">· the verifier grades each one</span></span
      >
      <div class="ac-edit">
        {#each acs as _, i (i)}
          <div class="ac-row">
            <span class="ac-hd"><Icon name="check" size={11} /></span>
            <Input
              bind:value={acs[i]}
              ariaLabel={`Criterion ${i + 1}`}
              placeholder="e.g. tests pass"
            />
            <Button
              variant="ghost"
              size="sm"
              iconOnly
              aria-label="Remove criterion"
              onclick={() => removeAc(i)}
            >
              {#snippet icon()}<Icon name="x" />{/snippet}
            </Button>
          </div>
        {/each}
      </div>
      <div>
        <Button variant="ghost" size="sm" data-testid={TESTID.newJobAcAdd} onclick={addAc}>
          {#snippet icon()}<Icon name="plus" />{/snippet}
          Add criterion
        </Button>
      </div>
    </div>

    <div class="setrow">
      <span class="lbl">agent</span>
      <span data-testid={TESTID.newJobAgent}>
        <Select
          options={agents.length ? agents : ['(no agents)']}
          bind:value={agent}
          ariaLabel="Host agent"
        />
      </span>
    </div>

    <div class="setrow">
      <span class="lbl">max attempts</span>
      <span class="stepper">
        <button type="button" aria-label="Decrease max attempts" onclick={() => step(-1)}>−</button>
        <span class="val" aria-live="polite">{maxAttempts}</span>
        <button type="button" aria-label="Increase max attempts" onclick={() => step(1)}>+</button>
        <span class="sub">retries before it goes stuck</span>
      </span>
    </div>

    {#if showExecutor}
      <div class="setrow">
        <span class="lbl">executor</span>
        <span data-testid={TESTID.newJobExecutor}>
          <Select options={executors} bind:value={executor} ariaLabel="Executor" />
        </span>
      </div>
    {/if}

    <div class="setrow">
      <span class="lbl">notify when</span>
      <Seg options={NOTIFY_LABELS} bind:value={notify} ariaLabel="Notify when" />
    </div>

    <div class="t-field">
      <span class="lbl">command preview</span>
      <pre class="cmd" data-testid={TESTID.newJobPreview}>{preview}</pre>
    </div>
  </div>

  {#snippet footer()}
    <Button
      variant="pri"
      size="sm"
      loading={submitting}
      disabled={!canSubmit}
      data-testid={TESTID.newJobSubmit}
      onclick={submit}
    >
      {#snippet icon()}<Icon name="play" />{/snippet}
      Spawn job
    </Button>
    <span class="kbd" aria-hidden="true">⌘⏎</span>
    <div class="grow"></div>
    <Button variant="ghost" size="sm" onclick={onClose}>Cancel</Button>
  {/snippet}
</Drawer>

<style>
  .jf {
    display: grid;
    gap: 14px;
  }
  .t-field {
    display: grid;
    gap: 5px;
  }
  .lbl,
  .t-field > .lbl {
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  .sub {
    text-transform: none;
    letter-spacing: 0;
    color: var(--tx3);
    font-weight: 400;
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
  .ac-edit {
    display: grid;
    gap: 6px;
  }
  .ac-row {
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .ac-hd {
    display: grid;
    place-items: center;
    color: var(--st-ok);
    flex: none;
  }
  .setrow {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .setrow .lbl {
    min-width: 92px;
  }
  .stepper {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font: 500 var(--fs-sm) var(--font-mono);
  }
  .stepper button {
    width: 24px;
    height: 24px;
    border-radius: var(--r-md);
    border: 1px solid var(--bd1);
    background: var(--bg2);
    color: var(--tx1);
    cursor: pointer;
    font-size: var(--fs-md);
    line-height: 1;
  }
  .stepper button:hover {
    border-color: var(--tx3);
    color: var(--tx0);
  }
  .stepper button:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .stepper .val {
    min-width: 14px;
    text-align: center;
    color: var(--tx0);
  }
  .cmd {
    margin: 0;
    background: var(--bg0);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 8px 10px;
    font: 400 var(--fs-xs) / 1.6 var(--font-mono);
    color: var(--tx2);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .newpill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 8px 0 7px;
    border-radius: var(--r-full);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    color: var(--st-queue);
    background: color-mix(in oklab, var(--st-queue) 13%, transparent);
    border: 1px solid color-mix(in oklab, var(--st-queue) 32%, transparent);
  }
  .kbd {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    padding: 2px 5px 3px;
    border: 1px solid var(--bd1);
    border-bottom-width: 2px;
    border-radius: 4px;
    background: var(--bg2);
    color: var(--tx2);
  }
  .grow {
    flex: 1;
  }
</style>
