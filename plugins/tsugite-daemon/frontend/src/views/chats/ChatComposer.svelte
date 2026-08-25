<script lang="ts">
  // Composer surface (part 3): wraps the library Composer with a slash-command
  // menu (GET /api/commands), multipart file attach (POST .../upload), and
  // per-session draft persistence. Plain text sends go to the conversation
  // controller's chat stream; a `/command` line is dispatched to the command
  // endpoint instead (the chat route does not parse slashes). Reasoning effort
  // lives in the conversation header (ModelEffort) as a persisted setting.
  //
  // Three cohesive concerns live in co-located controllers: the slash menu
  // (slashCommands), client-context chips (contextItems), and file attach/paste
  // (fileAttach). This component keeps the send/draft glue and wires each
  // controller's loading effects and derived menus to the markup.
  import { untrack, tick } from 'svelte';
  import Composer from '$lib/components/composer/Composer.svelte';
  import Picker from '$lib/components/overlays/Picker.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import type { ContextItem } from '$lib/context/contextProviders';
  import { TESTID } from '$lib/testids';
  import { readDraft, writeDraft, clearDraft, readDraftStaged, writeDraftStaged } from './draft';
  import { composerPrefill } from './composerPrefill.svelte';
  import { contextAttach } from './contextAttach.svelte';
  import { SlashCommands } from './slashCommands.svelte';
  import { ContextItems } from './contextItems.svelte';
  import { FileAttach } from './fileAttach.svelte';

  /** What a delivered message carries besides its text: staged uploads and any
   *  attached/auto-attached client-context items (sent as metadata, not text). */
  interface SendExtras {
    uploadedFiles: { name: string }[];
    contextMetadata?: ContextItem[];
  }

  let {
    sessionId,
    streaming = false,
    busy = false,
    queuedMessages = [],
    restoreFailed = null,
    onSend,
    onStop,
    onQueue,
    onUnqueue,
    onCommandResult,
  }: {
    sessionId: string | null;
    streaming?: boolean;
    busy?: boolean;
    /** Messages parked for after the in-flight turn (rendered as removable chips). */
    queuedMessages?: string[];
    /** A send that failed before it took (409 busy, daemon down): restore this
     *  text into an empty composer so the message isn't lost. */
    restoreFailed?: { text: string; seq: number } | null;
    onSend: (text: string, opts: SendExtras) => void;
    onStop: () => void;
    onQueue?: (text: string, opts: SendExtras) => void;
    onUnqueue?: (index: number) => void;
    /** A slash-command finished: surface its result as an inline conversation echo
     *  (the controller's ephemeral localEcho channel) instead of a toast. */
    onCommandResult?: (
      command: string,
      output: string,
      ok: boolean,
      action?: { label: string; href: string },
    ) => void;
  } = $props();

  let value = $state('');

  // Client-context chips, the "add context" menu, and the @ reference sources.
  const context = new ContextItems({
    get sessionId() {
      return sessionId;
    },
  });
  // File attach + paste (re-encode + upload) and the large-paste chooser. A pasted
  // reference marker routes back through the context controller's attachRef.
  const attach = new FileAttach({
    get value() {
      return value;
    },
    set value(v) {
      value = v;
    },
    get sessionId() {
      return sessionId;
    },
    attachRef: (kind, id) => {
      void context.attachRef(kind, id);
    },
  });
  // Slash-command menu + dispatch; an argument pick sends through handleSend.
  const slash = new SlashCommands({
    get value() {
      return value;
    },
    set value(v) {
      value = v;
    },
    get sessionId() {
      return sessionId;
    },
    handleSend: (t) => handleSend(t),
    // A getter (not a captured value or fixed-arity wrapper) keeps the current
    // prop and lets dispatchCommand preserve its call arity: error/unknown echoes
    // pass three args, a success echo four.
    get onCommandResult() {
      return onCommandResult;
    },
  });

  // Load each controller's lists (best-effort - a missing feature just leaves the
  // menu empty). The server providers and workspace file list load once; effort
  // levels fetch when the argument in play wants them.
  $effect(() => {
    context.loadServerProviders();
    context.loadFileRefs();
    slash.loadCommands();
  });
  $effect(() => {
    slash.syncEffortLevels();
  });

  // A ⌘K command pick routes through here, so the palette and the inline `/` menu
  // share one execution path. Reactive on the prefill store (not mount-only), so it
  // fires whether this composer was already open or just mounted for the session the
  // palette navigated to. A run reuses dispatchCommand - but only once the command
  // list has loaded, so the name can resolve; a prefill just fills + focuses.
  $effect(() => {
    const req = composerPrefill.pending;
    if (!req || req.sessionId !== sessionId) return;
    if (req.run && slash.commands.length === 0) return;
    composerPrefill.consume(sessionId);
    untrack(() => {
      if (req.run) {
        void slash.dispatchCommand(req.text);
      } else {
        value = req.text;
        void tick().then(() => focus());
      }
    });
  });

  // Swap drafts when the open session changes: persist nothing here, just load.
  let draftKeyId: string | null = null;
  $effect(() => {
    if (sessionId !== draftKeyId) {
      draftKeyId = sessionId;
      value = readDraft(sessionId);
      const staged = readDraftStaged(sessionId);
      attach.attachments = staged.attachments;
      context.contextItems = staged.contextItems;
    }
  });

  // Persist staged attachments + context items (already uploaded/captured, so just
  // references) alongside the text draft, so a phone that sleeps and reloads the
  // PWA restores them, not just the words. Guarded to the loaded session so a
  // swap's reset can't write one session's staged items under another's key.
  $effect(() => {
    const staged = { attachments: attach.attachments, contextItems: context.contextItems };
    if (sessionId !== draftKeyId) return;
    writeDraftStaged(sessionId, staged);
  });

  // An "add to chat" action (or a reference paste routed to a target chat) pushes
  // its captured items here for this session's composer. Runs after the draft-swap
  // effect above so a fresh navigation's reset can't clobber the attached chips.
  $effect(() => {
    const req = contextAttach.pending;
    if (!req || req.sessionId !== sessionId) return;
    contextAttach.consume(sessionId);
    untrack(() => context.addContextItems(req.items));
  });

  // A failed send hands its text back - but never clobber something the user
  // has typed since. Keyed by seq so each failure restores at most once.
  let restoredSeq = 0;
  $effect(() => {
    const failed = restoreFailed;
    if (!failed || failed.seq === restoredSeq) return;
    restoredSeq = failed.seq;
    untrack(() => {
      if (value.trim()) return;
      value = failed.text;
      writeDraft(sessionId, failed.text);
    });
  });

  // While the large-paste chooser is open, Escape or a click outside it defaults
  // to inline (never discard the text); re-subscribes as the prompt opens/closes.
  $effect(() => attach.installPasteDismiss());

  type Deliver = (text: string, opts: SendExtras) => void;

  // A /command is side-band (it does not join the conversation and carries no
  // context), so it dispatches immediately; a plain message gathers any context
  // and is delivered - sent now, or queued for after the turn. Context rides as
  // structured metadata; the message text is never touched.
  async function submit(text: string, deliver: Deliver) {
    if (/^\s*\//.test(text)) {
      void slash.dispatchCommand(text);
    } else {
      const contextMetadata = await context.resolveContextMetadata();
      deliver(text, {
        uploadedFiles: attach.attachments.map((a) => ({ name: a.name })),
        ...(contextMetadata.length ? { contextMetadata } : {}),
      });
    }
    value = '';
    attach.attachments = [];
    context.contextItems = [];
    clearDraft(sessionId);
  }

  function handleSend(text: string) {
    void submit(text, onSend);
  }

  function handleQueue(text: string) {
    void submit(text, (t, opts) => onQueue?.(t, opts));
  }

  // Imperative entry for OS files dropped on the chat surface (Surface.svelte),
  // funneled through the same re-encode + upload path as the picker and paste.
  export function attachFiles(files: File[]) {
    void attach.upload(files);
  }

  let composerEl = $state<{ focus: () => void }>();
  // Imperative focus, forwarded to the library composer's textarea; Surface calls
  // this to auto-focus on chat navigation.
  export function focus() {
    composerEl?.focus();
  }
</script>

<div class="composer-host" data-testid={TESTID.chatComposer}>
  {#if queuedMessages.length > 0}
    <div class="queuedrow" aria-label="Queued messages">
      {#each queuedMessages as msg, i (i)}
        <span class="t-chip" title={msg}>
          <Icon name="clock" />
          <span class="qtext">{msg}</span>
          {#if onUnqueue}
            <button
              type="button"
              class="x"
              aria-label={`Remove queued message ${i + 1}`}
              onclick={() => onUnqueue?.(i)}
            >
              <Icon name="x" />
            </button>
          {/if}
        </span>
      {/each}
      <span class="qnote-inline">sends when this turn finishes</span>
    </div>
  {/if}
  {#if slash.slashOpen}
    <div class="slashpop" role="listbox" aria-label="Commands">
      {#each slash.slashMatches as cmd, i (cmd.name)}
        <button
          type="button"
          role="option"
          aria-selected={i === slash.slashActive}
          class:is-active={i === slash.slashActive}
          onmousedown={(e) => {
            e.preventDefault();
            slash.pickSlash(cmd);
          }}
        >
          /{cmd.name}<span class="d">{cmd.description}</span>
        </button>
      {/each}
    </div>
  {:else if slash.argOpen && slash.argChoices}
    <div class="slashpop" role="listbox" aria-label="Options">
      {#each slash.argChoices as choice, i (choice)}
        <button
          type="button"
          role="option"
          aria-selected={i === slash.argActive}
          class:is-active={i === slash.argActive}
          onmousedown={(e) => {
            e.preventDefault();
            slash.pickArgChoice(choice);
          }}
        >
          {choice}
        </button>
      {/each}
    </div>
  {/if}
  {#if attach.pastePrompt}
    <div class="pastebanner" role="group" aria-label="Large paste" bind:this={attach.pasteBannerEl}>
      <span class="pb-txt"
        >Large paste — {attach.pastePrompt.text.length.toLocaleString()} characters</span
      >
      <span class="pb-actions">
        <Button size="sm" variant="pri" onclick={attach.pasteAsFile}>
          {#snippet icon()}<Icon name="file" />{/snippet}Attach as .txt
        </Button>
        <Button size="sm" variant="ghost" onclick={attach.pasteInline}>Paste inline</Button>
      </span>
    </div>
  {/if}

  <Composer
    bind:this={composerEl}
    bind:value
    {streaming}
    queued={busy && !streaming}
    attachments={attach.attachments}
    contextItems={context.contextChips}
    contextMenu={context.contextMenu}
    refItems={context.refItems}
    refSources={context.refSources}
    onSend={handleSend}
    {onStop}
    onInput={slash.onInput}
    onAttach={attach.openFilePicker}
    onCamera={attach.openCamera}
    onPickContext={context.pickContext}
    onRequestChoices={context.requestChoices}
    onPickRef={context.pickRef}
    onRemoveAttachment={attach.removeAttachment}
    onRemoveContext={context.removeContext}
    hint={busy && !streaming ? 'queued — sends when this turn finishes' : undefined}
    onKeydown={slash.onComposerKeydown}
    onPaste={attach.onPaste}
    onQueue={onQueue ? handleQueue : undefined}
  />

  <!-- Generic attach: accept-less so it never filters out non-image files. -->
  <input
    bind:this={attach.fileInput}
    data-testid={TESTID.composerFileInput}
    type="file"
    multiple
    hidden
    aria-hidden="true"
    tabindex="-1"
    onchange={attach.onFilesChosen}
  />
  <!-- Phone camera: accept="image/*" + capture makes iOS export JPEG (which the
       client re-encode then downscales), sidestepping HEIC entirely. -->
  <input
    bind:this={attach.cameraInput}
    data-testid={TESTID.composerCameraInput}
    type="file"
    accept="image/*"
    capture="environment"
    hidden
    aria-hidden="true"
    tabindex="-1"
    onchange={attach.onFilesChosen}
  />

  {#if context.picker}
    <Picker
      items={context.picker.items}
      title={context.picker.item.label}
      onPick={context.pickFromPicker}
      onClose={() => (context.picker = null)}
    />
  {/if}
</div>

<style>
  .composer-host {
    position: relative;
    flex: none;
  }
  /* Queued-message chips ride above the composer, mirroring its attachment
     row (same .t-chip skin - the file/x icon sizing rationale applies here too). */
  .queuedrow {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    padding: 6px 12px 0;
    background: var(--bg1);
    border-top: 1px solid var(--bd0);
  }
  .queuedrow .t-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--st-queue);
    white-space: nowrap;
    max-width: 100%;
  }
  .queuedrow .qtext {
    min-width: 0;
    max-width: 38ch;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .queuedrow .x {
    cursor: pointer;
    color: var(--tx3);
    display: inline-flex;
    background: none;
    border: 0;
    padding: 0;
  }
  .queuedrow .x:hover {
    color: var(--st-err);
  }
  .qnote-inline {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  /* Large-paste chooser: rides above the composer like the queued row, offering
     attach-as-file vs paste-inline (dismissal defaults to inline). */
  .pastebanner {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    padding: 6px 12px 0;
    background: var(--bg1);
    border-top: 1px solid var(--bd0);
  }
  .pastebanner .pb-txt {
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx2);
  }
  .pastebanner .pb-actions {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
  }
  /* slashpop - floats above the composer's input row. */
  .slashpop {
    position: absolute;
    left: 12px;
    right: 12px;
    bottom: calc(100% - 6px);
    z-index: 40;
    display: flex;
    flex-direction: column;
    max-height: 240px;
    overflow-y: auto;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    padding: 4px;
  }
  .slashpop button {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 6px 9px;
    border: 0;
    border-radius: var(--r-sm);
    background: transparent;
    color: var(--tx0);
    font: 600 var(--fs-sm) var(--font-mono);
    text-align: left;
    cursor: pointer;
  }
  .slashpop button:hover,
  .slashpop button.is-active {
    background: var(--bg4);
  }
  .slashpop .d {
    font: 400 var(--fs-xs) var(--font-ui);
    color: var(--tx2);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
