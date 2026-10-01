<script lang="ts">
  import Modal from '$lib/components/overlays/Modal.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import ModelPicker from '../chats/ModelPicker.svelte';
  import { composerPrefill } from '../chats/composerPrefill.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { TESTID } from '$lib/testids';
  import { agentLabel, chatAgents, chatSettings } from './launcher';

  let {
    open = false,
    initialAgent = null,
    onOpen,
    onClose,
  }: {
    open?: boolean;
    /** Path of the agent file to preselect. */
    initialAgent?: string | null;
    onOpen: (sessionId: string) => void;
    onClose: () => void;
  } = $props();

  const MODES = ['chat', 'run once'];
  const EFFORTS = ['default', 'low', 'medium', 'high'];

  let mode = $state('chat');
  let agentPath = $state('');
  let model = $state<string | null>(null);
  let effort = $state('default');
  let message = $state('');
  let busy = $state(false);

  const pool = $derived(
    mode === 'chat' ? chatAgents(agentsMeta.agentFiles) : agentsMeta.agentFiles,
  );
  const labels = $derived(pool.map((f) => agentLabel(f, agentsMeta.agentFiles)));
  const agent = $derived(pool.find((f) => f.path === agentPath));
  const agentSel = $derived(agent ? agentLabel(agent, agentsMeta.agentFiles) : '');

  // Reset the form each time the launcher opens. The requested agent is applied
  // once its file is known (a cold load lists files after open), which also picks
  // the mode that agent supports.
  let wasOpen = false;
  let initialPending = false;
  $effect(() => {
    if (open && !wasOpen) {
      agentsMeta.loadAgentFiles().catch((err) => {
        toasts.push('err', 'Could not load agents', {
          body: err instanceof Error ? err.message : String(err),
        });
      });
      model = null;
      effort = 'default';
      message = '';
      busy = false;
      mode = 'chat';
      agentPath = '';
      initialPending = initialAgent !== null;
    }
    wasOpen = open;
  });

  $effect(() => {
    if (!open || !initialPending) return;
    const wanted = agentsMeta.agentFiles.find((f) => f.path === initialAgent);
    if (!wanted) return;
    initialPending = false;
    mode = wanted.chat ? 'chat' : 'run once';
    agentPath = wanted.path;
  });

  $effect(() => {
    if (open && !initialPending && !agent) {
      const fallback = pool.find((f) => f.default) ?? pool[0];
      if (fallback) agentPath = fallback.path;
    }
  });

  const canStart = $derived(!busy && !!agent && (mode === 'chat' || message.trim().length > 0));

  async function start() {
    if (!canStart || !agent) return;
    busy = true;
    try {
      if (mode === 'chat') {
        const id = await sessions.newSession({
          settings: chatSettings({ agent, model, effort }),
        });
        if (message.trim()) composerPrefill.request(id, message.trim(), true);
        await sessions.load();
        onOpen(id);
        onClose();
      } else {
        const id = await sessions.startRun({
          prompt: message.trim(),
          agentFile: agent.path,
          model: model || undefined,
          effort: effort === 'default' ? undefined : effort,
        });
        toasts.push('ok', 'Run started', {
          body: `${agent.name} · one-off run`,
          actionLabel: 'Open',
          onAction: () => onOpen(id),
        });
        onClose();
      }
    } catch (e) {
      toasts.push('err', mode === 'chat' ? 'Could not start chat' : 'Could not start run', {
        body: e instanceof Error ? e.message : String(e),
      });
    } finally {
      busy = false;
    }
  }

  function onMessageKeydown(e: KeyboardEvent) {
    // Cmd/Ctrl+Enter starts, matching the composer's send affordance.
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      void start();
    }
  }
</script>

<Modal {open} onclose={onClose} title={mode === 'chat' ? 'New chat' : 'Run agent'}>
  <div class="rl">
    <div class="rl-field rl-row" data-testid={TESTID.launcherMode}>
      <Seg options={MODES} bind:value={mode} ariaLabel="Launch mode" />
      <span class="rl-hint"
        >{mode === 'chat'
          ? 'a conversation with a chat agent'
          : 'one background run, any agent'}</span
      >
    </div>

    <div class="rl-field rl-row">
      <span class="rl-lbl">agent</span>
      {#key mode}
        <span data-testid={TESTID.launcherAgent}>
          <Select
            options={labels}
            value={agentSel}
            ariaLabel="Agent"
            onchange={(label) => {
              const i = labels.indexOf(label);
              if (i >= 0) agentPath = pool[i]!.path;
            }}
          />
        </span>
      {/key}
    </div>
    {#if agent?.description}<div class="rl-desc">{agent.description}</div>{/if}

    <div class="rl-field rl-row">
      <span class="rl-lbl">model</span>
      <ModelPicker sessionId={null} value={model} onPick={(id) => (model = id)} />
      <span class="rl-hint">{model ? 'for this session' : "the agent's own model"}</span>
    </div>

    <div class="rl-field rl-row">
      <span class="rl-lbl">effort</span>
      <Seg options={EFFORTS} bind:value={effort} ariaLabel="Reasoning effort override" />
    </div>

    <label class="rl-field">
      <span class="rl-lbl"
        >{mode === 'chat' ? 'starting message' : 'prompt'}
        {#if mode === 'run once'}<span class="req" aria-hidden="true">*</span>{/if}</span
      >
      <textarea
        class="t-input"
        rows="4"
        placeholder={mode === 'chat'
          ? 'Optional. Sent as the first message.'
          : 'What should the agent do?'}
        bind:value={message}
        onkeydown={onMessageKeydown}
        data-autofocus
        aria-label={mode === 'chat' ? 'Starting message' : 'Run prompt'}></textarea>
    </label>
  </div>

  {#snippet footer()}
    <Button onclick={onClose}>Cancel</Button>
    <Button
      variant="pri"
      disabled={!canStart}
      loading={busy}
      data-testid={TESTID.launcherStart}
      onclick={() => void start()}
    >
      {#snippet icon()}<Icon name="sparkle" />{/snippet}
      {mode === 'chat' ? 'Start chat' : 'Run'}
    </Button>
  {/snippet}
</Modal>

<style>
  .rl {
    display: grid;
    gap: 13px;
    min-width: min(390px, 72vw);
  }
  .rl-desc {
    font-size: var(--fs-xs);
    color: var(--tx2);
    margin-top: -6px;
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
</style>
