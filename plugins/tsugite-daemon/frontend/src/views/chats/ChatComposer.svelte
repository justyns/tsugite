<script lang="ts" module>
  // Auto-attach capture failures notify once per provider per app session, not on
  // every send, so a denied permission with the setting on doesn't spam a toast
  // per message.
  const autoAttachWarned = new Set<string>();
</script>

<script lang="ts">
  // Composer surface (part 3): wraps the library Composer with a slash-command
  // menu (GET /api/commands), multipart file attach (POST .../upload), and
  // per-session draft persistence. Plain text sends go to the conversation
  // controller's chat stream; a `/command` line is dispatched to the command
  // endpoint instead (the chat route does not parse slashes). Reasoning effort
  // lives in the conversation header (ModelEffort) as a persisted setting.
  import { untrack, tick } from 'svelte';
  import Composer from '$lib/components/composer/Composer.svelte';
  import Picker from '$lib/components/overlays/Picker.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import type {
    Attachment,
    ContextMenuItem,
    ContextChoice,
    RefItem,
    RefSource,
  } from '$lib/components/composer/types';
  import { api } from '$lib/api/client';
  import { loadImageConfig } from '$lib/api/serverConfig';
  import { reencodeImage } from '$lib/media/imageEncode';
  import {
    contextProviders,
    contextProvider,
    type ContextItem,
  } from '$lib/context/contextProviders';
  import {
    fetchServerProviders,
    fetchServerChoices,
    captureServerContext,
    searchServerProvider,
    type ServerProvider,
  } from '$lib/context/serverProviders';
  import { sessions } from '$lib/stores/sessions.svelte';
  import { buildSessionRefs } from './sessionRefs';
  import { autoAttachStore } from '$lib/stores/autoAttach.svelte';
  import { auth } from '$lib/stores/auth.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { TESTID } from '$lib/testids';
  import { loadWorkspace } from '../files/load';
  import { readDraft, writeDraft, clearDraft, readDraftStaged, writeDraftStaged } from './draft';
  import { extractFiles } from './dropFiles';
  import { composerPrefill } from './composerPrefill.svelte';
  import { contextAttach } from './contextAttach.svelte';
  import { parseRefMarker } from './attachRecord';
  import { modelPickerRequest } from './modelPickerSignal.svelte';
  import { commandArgParam } from '$lib/shell/palette-sources';

  interface CommandParam {
    name: string;
    type: string;
    required: boolean;
    /** Rich-input hint: a dedicated control (`model`, `effort`) for this arg. */
    widget?: string;
    /** A fixed set of valid values, offered as an inline choices list. */
    choices?: string[];
  }
  interface Command {
    name: string;
    description: string;
    params: CommandParam[];
  }

  /** What a delivered message carries besides its text: staged uploads and any
   *  attached/auto-attached client-context items (sent as metadata, not text). */
  interface SendExtras {
    uploadedFiles: { name: string }[];
    contextMetadata?: ContextItem[];
  }

  let {
    agent,
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
    agent: string;
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
  let attachments = $state<Attachment[]>([]);
  // Manually-attached context items (chips). Provider key -> item.
  let contextItems = $state<ContextItem[]>([]);
  // Daemon-provided menu providers, loaded once (best-effort - empty if the daemon
  // lacks the feature). Client providers below always show regardless.
  let serverProviders = $state<ServerProvider[]>([]);
  $effect(() => {
    fetchServerProviders().then((ps) => (serverProviders = ps));
  });
  // Workspace files feeding the @ popover, loaded once per agent (best-effort).
  // Picking a file attaches it as a workspace-file context item rather than
  // inserting `@path` text (see pickRef).
  let fileRefs = $state<RefItem[]>([]);
  let refAgent: string | null = null;
  $effect(() => {
    const a = agent;
    if (!a || a === refAgent) return;
    refAgent = a;
    fileRefs = [];
    loadWorkspace(a)
      .then((ws) => {
        if (refAgent !== a) return;
        fileRefs = ws.entries
          .filter((e) => !e.is_dir)
          .map((e) => ({ id: e.path, kind: 'file' as const, label: e.path, group: 'Files' }));
      })
      .catch(() => {});
  });
  // The @ popover's built-in candidates: recent sessions first, then workspace
  // files (both grouped). Reactive to the sessions store and the open chat, so
  // the source stays current and never lists the chat you are in.
  const refItems = $derived<RefItem[]>([
    ...buildSessionRefs(sessions.ordered, sessionId),
    ...fileRefs,
  ]);
  // Prefix sources: any server provider that declared an autocomplete prefix. A
  // pick captures through that provider's key (carried on each result item).
  const refSources = $derived<RefSource[]>(
    serverProviders
      .filter((p) => p.autocompletePrefix)
      .map((p) => ({
        prefix: p.autocompletePrefix as string,
        label: p.label,
        search: async (q: string): Promise<RefItem[]> => {
          if (!sessionId) return [];
          const results = await searchServerProvider(p.key, sessionId, q);
          return results.map((r) => ({
            id: r.value,
            kind: 'plugin' as const,
            label: r.label,
            providerKey: p.key,
            group: p.label,
          }));
        },
      })),
  );
  // The generic Picker overlay, opened when a large-option (picker) provider is
  // chosen; it lists that provider's choices and captures the pick as a chip.
  let picker = $state<{ item: ContextMenuItem; items: ContextChoice[] } | null>(null);
  // The "add context" menu: the static client registry first, then any server
  // providers whose key doesn't collide with a client one (client wins, keeping
  // the keyed list unique). Client picks capture in the browser; server picks on
  // the daemon.
  const contextMenu = $derived<ContextMenuItem[]>([
    ...contextProviders.map((p) => ({
      key: p.key,
      label: p.label,
      icon: p.icon,
      kind: 'client' as const,
      hasChoices: false,
    })),
    ...serverProviders
      .filter((sp) => sp.inMenu && !contextProviders.some((cp) => cp.key === sp.key))
      .map((sp) => ({
        key: sp.key,
        label: sp.label,
        icon: sp.icon,
        kind: 'server' as const,
        hasChoices: sp.hasChoices,
        picker: sp.picker,
      })),
  ]);
  // Chips carry their provider's icon for the composer's attach row.
  const contextChips = $derived(
    contextItems.map((it) => ({ ...it, icon: contextProvider(it.key)?.icon })),
  );
  let commands = $state<Command[]>([]);
  let slashActive = $state(0);
  let argActive = $state(0);
  let fileInput = $state<HTMLInputElement>();
  let cameraInput = $state<HTMLInputElement>();

  // Load the command list once (best-effort - the menu just stays empty on 404).
  $effect(() => {
    void agent;
    api
      .get<{ commands: Command[] }>('/api/commands')
      .then((res) => (commands = res.commands))
      .catch(() => (commands = []));
  });

  // A ⌘K command pick routes through here, so the palette and the inline `/` menu
  // share one execution path. Reactive on the prefill store (not mount-only), so it
  // fires whether this composer was already open or just mounted for the session the
  // palette navigated to. A run reuses dispatchCommand - but only once the command
  // list has loaded, so the name can resolve; a prefill just fills + focuses.
  $effect(() => {
    const req = composerPrefill.pending;
    if (!req || req.sessionId !== sessionId) return;
    if (req.run && commands.length === 0) return;
    composerPrefill.consume(sessionId);
    untrack(() => {
      if (req.run) {
        void dispatchCommand(req.text);
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
      attachments = staged.attachments;
      contextItems = staged.contextItems;
    }
  });

  // Persist staged attachments + context items (already uploaded/captured, so just
  // references) alongside the text draft, so a phone that sleeps and reloads the
  // PWA restores them, not just the words. Guarded to the loaded session so a
  // swap's reset can't write one session's staged items under another's key.
  $effect(() => {
    const staged = { attachments, contextItems };
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
    untrack(() => addContextItems(req.items));
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

  // Slash menu: open while the value is a lone `/token` (before any argument);
  // Escape dismisses it until the next keystroke.
  let slashDismissed = $state(false);
  const slashQuery = $derived.by(() => {
    const m = /^\s*\/([^\s]*)$/.exec(value);
    return m ? (m[1] ?? '').toLowerCase() : null;
  });
  const slashMatches = $derived(
    slashQuery === null ? [] : commands.filter((c) => c.name.toLowerCase().startsWith(slashQuery)),
  );
  const slashOpen = $derived(!slashDismissed && slashMatches.length > 0);

  // Second stage: once the command is typed and a space begins its argument
  // (`/<cmd> <partial>`), resolve which command and which arg param is in play so
  // its `choices`/`widget` hint can drive an inline options list (kept to one line
  // - a slash command isn't multi-line).
  const argContext = $derived.by(() => {
    const m = /^\s*\/(\S+)[ \t]+([^\n]*)$/.exec(value);
    if (!m) return null;
    const [, name = '', partial = ''] = m;
    const cmd = commands.find((c) => c.name.toLowerCase() === name.toLowerCase());
    const arg = cmd && commandArgParam(cmd);
    return cmd && arg ? { cmd, arg, partial } : null;
  });

  // Effort levels are model-dependent, so they're fetched per session when the
  // current arg wants them (a `widget:"effort"` param). A null result (model has no
  // effort levels) leaves the arg a plain text field.
  let effortLevels = $state<string[] | null>(null);
  let effortFetchKey = '';
  $effect(() => {
    if (argContext?.arg.widget !== 'effort' || !sessionId) return;
    const key = `${agent}#${sessionId}`;
    if (key === effortFetchKey) return;
    effortFetchKey = key;
    effortLevels = null;
    api
      .get<{ supported_effort_levels: string[] | null }>(
        `/api/agents/${encodeURIComponent(agent)}/effort-levels?session_id=${encodeURIComponent(sessionId)}`,
      )
      .then((r) => {
        if (effortFetchKey === key) effortLevels = r.supported_effort_levels;
      })
      .catch(() => {});
  });

  // The choices to offer for the current argument, filtered by what's typed. A
  // `widget:"model"` arg has no inline list (picking the command opens the header
  // picker instead), so it's excluded here.
  const argChoices = $derived.by(() => {
    const ctx = argContext;
    if (!ctx || ctx.arg.widget === 'model') return null;
    const all = ctx.arg.choices ?? (ctx.arg.widget === 'effort' ? effortLevels : null);
    if (!all || all.length === 0) return null;
    const q = ctx.partial.trim().toLowerCase();
    return q ? all.filter((c) => c.toLowerCase().includes(q)) : all;
  });
  const argOpen = $derived(!slashDismissed && !slashOpen && !!argChoices && argChoices.length > 0);

  function onInput(next: string) {
    writeDraft(sessionId, next);
    slashActive = 0;
    argActive = 0;
    slashDismissed = false;
  }

  function pickSlash(cmd: Command) {
    // A model command routes to the header picker rather than a text field.
    if (commandArgParam(cmd)?.widget === 'model' && sessionId) {
      modelPickerRequest.request(sessionId);
      value = '';
      slashActive = 0;
      clearDraft(sessionId);
      return;
    }
    value = `/${cmd.name} `;
    slashActive = 0;
  }

  // Submit the command with the chosen argument value, running it through the same
  // dispatch (and inline echo) path as a typed-then-sent command.
  function pickArgChoice(choice: string) {
    const ctx = argContext;
    if (ctx) handleSend(`/${ctx.cmd.name} ${choice}`);
  }

  // Shared menu navigation for both stages: Tab/Enter picks the highlighted item,
  // arrows move the highlight (wrapping), Escape dismisses. Returns true when it
  // consumed the key so the composer doesn't also act on it.
  function navMenu<T>(
    e: KeyboardEvent,
    items: T[],
    active: number,
    setActive: (i: number) => void,
    choose: (item: T) => void,
  ): boolean {
    if (e.key === 'Tab' || e.key === 'Enter') {
      e.preventDefault();
      const it = items[active] ?? items[0];
      if (it !== undefined) choose(it);
      return true;
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((active + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length);
      return true;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      slashDismissed = true;
      return true;
    }
    return false;
  }

  function onComposerKeydown(e: KeyboardEvent): boolean {
    if (slashOpen)
      return navMenu(e, slashMatches, slashActive, (i) => (slashActive = i), pickSlash);
    if (argOpen && argChoices)
      return navMenu(e, argChoices, argActive, (i) => (argActive = i), pickArgChoice);
    return false;
  }

  async function dispatchCommand(line: string): Promise<void> {
    const trimmed = line.trim().slice(1);
    const gap = trimmed.indexOf(' ');
    const name = (gap === -1 ? trimmed : trimmed.slice(0, gap)).toLowerCase();
    const rest = gap === -1 ? '' : trimmed.slice(gap + 1).trim();
    // Echo under the exact line the user ran (Claude-Code style), args and all.
    const label = `/${name}${rest ? ` ${rest}` : ''}`;
    const cmd = commands.find((c) => c.name === name);
    if (!cmd) {
      onCommandResult?.(label, `Unknown command /${name}`, false);
      return;
    }
    const body: Record<string, unknown> = {};
    if (cmd.params.some((p) => p.name === 'user_id')) body.user_id = auth.userId;
    if (sessionId && cmd.params.some((p) => p.name === 'session_id')) body.session_id = sessionId;
    const primary = commandArgParam(cmd)?.name;
    if (primary && rest) body[primary] = rest;
    // The /job affordance moves from the toast's action button into the echo: a
    // link to the jobs board, matching the header's `#jobs` chip.
    const action = name === 'job' ? { label: 'Open jobs', href: '#jobs' } : undefined;
    try {
      const res = await api.post<{ result?: string }>(
        `/api/agents/${encodeURIComponent(agent)}/commands/${encodeURIComponent(name)}`,
        body,
      );
      onCommandResult?.(label, typeof res.result === 'string' ? res.result : '', true, action);
    } catch (err) {
      onCommandResult?.(label, err instanceof Error ? err.message : String(err), false);
    }
  }

  // Gather the context_metadata for an outgoing message: the manually-attached
  // chips, plus any provider whose auto-attach setting is on (best-effort
  // captured, a manual chip for the same provider winning). A capture failure
  // never blocks the send - it warns at most once per provider per session and
  // the message goes without that item.
  async function resolveContextMetadata(): Promise<ContextItem[]> {
    const items = [...contextItems];
    for (const p of contextProviders) {
      if (!p.autoAttachStoreKey) continue;
      if (items.some((i) => i.key === p.key)) continue;
      if (!autoAttachStore(p.autoAttachStoreKey).enabled) continue;
      const res = await p.capture();
      if ('value' in res) {
        items.push({ key: p.key, label: p.label, value: res.value });
      } else if (!autoAttachWarned.has(p.key)) {
        autoAttachWarned.add(p.key);
        toasts.push('warn', `${p.label} not attached`, { body: res.error.message });
      }
    }
    return items;
  }

  type Deliver = (text: string, opts: SendExtras) => void;

  // A /command is side-band (it does not join the conversation and carries no
  // context), so it dispatches immediately; a plain message gathers any context
  // and is delivered - sent now, or queued for after the turn. Context rides as
  // structured metadata; the message text is never touched.
  async function submit(text: string, deliver: Deliver) {
    if (/^\s*\//.test(text)) {
      void dispatchCommand(text);
    } else {
      const contextMetadata = await resolveContextMetadata();
      deliver(text, {
        uploadedFiles: attachments.map((a) => ({ name: a.name })),
        ...(contextMetadata.length ? { contextMetadata } : {}),
      });
    }
    value = '';
    attachments = [];
    contextItems = [];
    clearDraft(sessionId);
  }

  function handleSend(text: string) {
    void submit(text, onSend);
  }

  function handleQueue(text: string) {
    void submit(text, (t, opts) => onQueue?.(t, opts));
  }

  // Add (or refresh) captured chips, replacing any existing item that shares an
  // incoming key so a re-pick updates in place rather than duplicating.
  function addContextItems(incoming: ContextItem[]) {
    if (!incoming.length) return;
    const keys = new Set(incoming.map((i) => i.key));
    contextItems = [...contextItems.filter((i) => !keys.has(i.key)), ...incoming];
  }

  // Attach a record named by a pasted reference marker to this composer: capture
  // it on the daemon and chip it. An empty capture (stale/unknown id) toasts; an
  // error toasts. Shares addContextItems with the manual-add and add-to-chat paths.
  async function attachRef(kind: string, id: string) {
    if (!sessionId) return;
    try {
      const items = await captureServerContext(kind, sessionId, id);
      if (items.length) addContextItems(items);
      else toasts.push('warn', 'Nothing to attach');
    } catch (err) {
      toasts.push('err', 'Could not attach', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // Capture a daemon-side context item and chip it; a provider error toasts.
  // `arg` is a chosen submenu/picker value, or null for a no-choices provider.
  async function captureServerItem(key: string, label: string, arg: string | null) {
    if (!sessionId) return;
    try {
      addContextItems(await captureServerContext(key, sessionId, arg));
    } catch (err) {
      toasts.push('err', `${label} unavailable`, {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // Pick a provider from the "add context" menu. A client provider captures in the
  // browser; a server provider captures on the daemon (with `arg` = a chosen
  // submenu value, or null). A picker provider (large, searchable option set)
  // opens the Picker overlay instead. Captured items become chips; a failure toasts.
  async function pickContext(item: ContextMenuItem, arg: string | null = null) {
    if (item.kind === 'server') {
      if (!sessionId) return;
      if (item.picker && arg === null) {
        await openPicker(item);
        return;
      }
      await captureServerItem(item.key, item.label, arg);
      return;
    }
    const p = contextProvider(item.key);
    if (!p) return;
    const res = await p.capture();
    if ('value' in res) {
      addContextItems([{ key: p.key, label: p.label, value: res.value }]);
    } else {
      toasts.push(res.error.code === 'permission' ? 'warn' : 'err', `${p.label} unavailable`, {
        body: res.error.message,
      });
    }
  }

  // Open the Picker for a large-option provider: fetch its choices and show them
  // in a searchable overlay. Nothing to pick (empty or a failed load) simply
  // doesn't open, matching the inline-submenu behavior.
  async function openPicker(item: ContextMenuItem) {
    if (!sessionId) return;
    const choices = await fetchServerChoices(item.key, sessionId);
    if (choices.length === 0) return;
    picker = { item, items: choices };
  }

  function pickFromPicker(value: string) {
    const item = picker?.item;
    picker = null;
    if (item) void captureServerItem(item.key, item.label, value);
  }

  // A reference chosen in the @ popover attaches as a context chip. A file goes
  // through the workspace-file provider by its path (same as picking "Workspace
  // file" from the menu, and silent on an empty capture). A session or plugin
  // result attaches through its provider key (session's key IS `session`; a
  // plugin result carries its source provider key) with the ref id as the arg.
  function pickRef(ref: RefItem) {
    if (ref.kind === 'file') {
      void captureServerItem('file', ref.label, ref.label);
    } else {
      void attachRef(ref.providerKey ?? ref.kind, ref.id);
    }
  }

  // Load a server provider's submenu options, scoped to the active session.
  const requestChoices = (key: string): Promise<ContextChoice[]> =>
    sessionId ? fetchServerChoices(key, sessionId) : Promise.resolve([]);

  function removeContext(key: string) {
    contextItems = contextItems.filter((i) => i.key !== key);
  }

  function openFilePicker() {
    fileInput?.click();
  }

  function openCamera() {
    cameraInput?.click();
  }

  function onFilesChosen(e: Event) {
    const input = e.currentTarget as HTMLInputElement;
    const files = input.files ? Array.from(input.files) : [];
    input.value = '';
    void uploadChosen(files);
  }

  // A large text paste offers a choice instead of dumping a wall of text into the
  // draft. Thresholds mirror the legacy UI: large past 500 chars OR 11 lines.
  const PASTE_MAX_CHARS = 500;
  const PASTE_MAX_LINES = 11;

  let pastePrompt = $state<{ text: string; start: number; end: number } | null>(null);
  let pasteTa: HTMLTextAreaElement | null = null;
  let pasteBannerEl = $state<HTMLElement>();

  function isLargePaste(text: string): boolean {
    return text.length > PASTE_MAX_CHARS || text.split('\n').length > PASTE_MAX_LINES;
  }

  // A pasted screenshot (Firefox exposes it as an image File on clipboardData)
  // routes to attach; a large text paste opens the chooser; anything smaller is
  // left to the browser's native paste. Files always win over accompanying text.
  function onPaste(e: ClipboardEvent) {
    const files = extractFiles(e.clipboardData);
    if (files.length > 0) {
      e.preventDefault();
      void uploadChosen(files);
      return;
    }
    // A "copy reference" paste carries an html marker naming a record; attach that
    // record instead of pasting its text. A normal paste has no marker and falls
    // through to the unchanged image/large-text handling below.
    const ref = parseRefMarker(e.clipboardData?.getData('text/html') ?? '');
    if (ref) {
      e.preventDefault();
      void attachRef(ref.kind, ref.id);
      return;
    }
    const text = e.clipboardData?.getData('text/plain') ?? '';
    if (!isLargePaste(text)) return;
    const ta = e.target as HTMLTextAreaElement;
    e.preventDefault();
    pasteTa = ta;
    pastePrompt = {
      text,
      start: ta.selectionStart ?? value.length,
      end: ta.selectionEnd ?? value.length,
    };
  }

  function pasteFilename(): string {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    const ts = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    return `pasted-${ts}.txt`;
  }

  function resetPastePrompt() {
    pastePrompt = null;
    pasteTa = null;
  }

  function pasteAsFile() {
    const p = pastePrompt;
    if (!p) return;
    resetPastePrompt();
    void uploadChosen([new File([p.text], pasteFilename(), { type: 'text/plain' })]);
  }

  function pasteInline() {
    const p = pastePrompt;
    if (!p) return;
    const ta = pasteTa;
    resetPastePrompt();
    const next = value.slice(0, p.start) + p.text + value.slice(p.end);
    value = next;
    writeDraft(sessionId, next);
    const caret = p.start + p.text.length;
    void tick().then(() => ta?.setSelectionRange(caret, caret));
  }

  // While the chooser is open, Escape or a click outside it defaults to inline -
  // never discard the pasted text.
  $effect(() => {
    if (!pastePrompt) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        pasteInline();
      }
    };
    const onDown = (e: MouseEvent) => {
      if (pasteBannerEl && !pasteBannerEl.contains(e.target as Node)) pasteInline();
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('mousedown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('mousedown', onDown, true);
    };
  });

  // Imperative entry for OS files dropped on the chat surface (Surface.svelte),
  // funneled through the same re-encode + upload path as the picker and paste.
  export function attachFiles(files: File[]) {
    void uploadChosen(files);
  }

  let composerEl = $state<{ focus: () => void }>();
  // Imperative focus, forwarded to the library composer's textarea; Surface calls
  // this to auto-focus on chat navigation.
  export function focus() {
    composerEl?.focus();
  }

  async function uploadChosen(files: File[]) {
    if (!files.length) return;
    try {
      // Re-encode photos client-side (downscale + JPEG) before upload; non-images
      // and svg/gif pass through untouched. Config comes from /api/health.
      const cfg = await loadImageConfig();
      const processed = await Promise.all(files.map((f) => reencodeImage(f, cfg)));
      const res = await api.uploadFiles<{ files: { name: string; size?: number }[] }>(
        `/api/agents/${encodeURIComponent(agent)}/upload`,
        processed,
      );
      attachments = [
        ...attachments,
        ...res.files.map((f) => ({
          id: f.name,
          name: f.name,
          ...(f.size ? { size: `${Math.round(f.size / 1024)} KB` } : {}),
        })),
      ];
    } catch (err) {
      toasts.push('err', 'Upload failed', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  function removeAttachment(id: string) {
    attachments = attachments.filter((a) => a.id !== id);
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
  {#if slashOpen}
    <div class="slashpop" role="listbox" aria-label="Commands">
      {#each slashMatches as cmd, i (cmd.name)}
        <button
          type="button"
          role="option"
          aria-selected={i === slashActive}
          class:is-active={i === slashActive}
          onmousedown={(e) => {
            e.preventDefault();
            pickSlash(cmd);
          }}
        >
          /{cmd.name}<span class="d">{cmd.description}</span>
        </button>
      {/each}
    </div>
  {:else if argOpen && argChoices}
    <div class="slashpop" role="listbox" aria-label="Options">
      {#each argChoices as choice, i (choice)}
        <button
          type="button"
          role="option"
          aria-selected={i === argActive}
          class:is-active={i === argActive}
          onmousedown={(e) => {
            e.preventDefault();
            pickArgChoice(choice);
          }}
        >
          {choice}
        </button>
      {/each}
    </div>
  {/if}
  {#if pastePrompt}
    <div class="pastebanner" role="group" aria-label="Large paste" bind:this={pasteBannerEl}>
      <span class="pb-txt">Large paste — {pastePrompt.text.length.toLocaleString()} characters</span
      >
      <span class="pb-actions">
        <Button size="sm" variant="pri" onclick={pasteAsFile}>
          {#snippet icon()}<Icon name="file" />{/snippet}Attach as .txt
        </Button>
        <Button size="sm" variant="ghost" onclick={pasteInline}>Paste inline</Button>
      </span>
    </div>
  {/if}

  <Composer
    bind:this={composerEl}
    bind:value
    {agent}
    {streaming}
    queued={busy && !streaming}
    {attachments}
    contextItems={contextChips}
    {contextMenu}
    {refItems}
    {refSources}
    onSend={handleSend}
    {onStop}
    {onInput}
    onAttach={openFilePicker}
    onCamera={openCamera}
    onPickContext={pickContext}
    onRequestChoices={requestChoices}
    onPickRef={pickRef}
    onRemoveAttachment={removeAttachment}
    onRemoveContext={removeContext}
    hint={busy && !streaming ? 'queued — sends when this turn finishes' : undefined}
    onKeydown={onComposerKeydown}
    {onPaste}
    onQueue={onQueue ? handleQueue : undefined}
  />

  <!-- Generic attach: accept-less so it never filters out non-image files. -->
  <input
    bind:this={fileInput}
    data-testid={TESTID.composerFileInput}
    type="file"
    multiple
    hidden
    aria-hidden="true"
    tabindex="-1"
    onchange={onFilesChosen}
  />
  <!-- Phone camera: accept="image/*" + capture makes iOS export JPEG (which the
       client re-encode then downscales), sidestepping HEIC entirely. -->
  <input
    bind:this={cameraInput}
    data-testid={TESTID.composerCameraInput}
    type="file"
    accept="image/*"
    capture="environment"
    hidden
    aria-hidden="true"
    tabindex="-1"
    onchange={onFilesChosen}
  />

  {#if picker}
    <Picker
      items={picker.items}
      title={picker.item.label}
      onPick={pickFromPicker}
      onClose={() => (picker = null)}
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
