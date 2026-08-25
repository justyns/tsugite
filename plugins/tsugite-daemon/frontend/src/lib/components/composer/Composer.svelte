<script lang="ts" module>
  import type { RefKind } from './types';
  let _uid = 0;
  const nextUid = () => ++_uid;
  // Kinds that attach as a context chip (through onPickRef) instead of inserting
  // their `@label` text; the rest (chat/agent/terminal) insert inline.
  const ATTACH_KINDS = new Set<RefKind>(['file', 'session', 'plugin']);
  // Debounce a source search so a fast typist fires one request, not one per key.
  const SEARCH_DEBOUNCE_MS = 150;
  // How long after a send the Stop control ignores an activation that arrives as
  // the tail of a double-tap or double-click.
  export const SEND_GUARD_MS = 500;
</script>

<script lang="ts">
  import { tick, type Snippet } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import StagedStrip from './StagedStrip.svelte';
  import RefAutocomplete, { refNav } from './RefAutocomplete.svelte';
  import { parseRefToken } from './refToken';
  import { TESTID } from '$lib/testids';
  import type {
    Attachment,
    RefItem,
    RefSource,
    ContextChip,
    ContextMenuItem,
    ContextChoice,
  } from './types';

  let {
    value = $bindable(''),
    placeholder = 'message tsugite · / for commands, ⇧⏎ for newline',
    streaming = false,
    queued = false,
    rows = 2,
    hint,
    showKbd = false,
    attachments = [],
    contextItems = [],
    contextMenu = [],
    refItems = [],
    refSources = [],
    onSend,
    onStop,
    onQueue,
    onAttach,
    onCamera,
    onPickContext,
    onRequestChoices,
    onPickRef,
    onRemoveAttachment,
    onRemoveContext,
    onInput,
    onKeydown,
    onPaste,
    leading,
    pasteAffordance,
  }: {
    value?: string;
    placeholder?: string;
    /** Streaming turn in flight - flips the primary Send button to a danger Stop. */
    streaming?: boolean;
    /** Turn busy - the message will queue and send when the turn finishes. */
    queued?: boolean;
    rows?: number;
    /** Muted right-aligned note in the bottom row, e.g. `est. +1.2k tok`. */
    hint?: string;
    showKbd?: boolean;
    attachments?: Attachment[];
    /** Attached client-context items, shown as removable chips. Any present also
     *  lets an otherwise-empty composer send (a context-only message). */
    contextItems?: ContextChip[];
    /** Providers offered in the "add context" menu; an empty list hides the button. */
    contextMenu?: ContextMenuItem[];
    /** Candidate references for the @/# popover; the composer filters by the typed query. */
    refItems?: RefItem[];
    /** Prefix-scoped `@` autocomplete sources: typing `@<prefix> <query>` switches
     *  the popover to the matching source and (debounced) fetches its results. */
    refSources?: RefSource[];
    onSend?: (text: string) => void;
    onStop?: () => void;
    /** Offered while `streaming`: park the draft to send after this turn. Enter
     *  with a non-empty draft queues instead of stopping. */
    onQueue?: (text: string) => void;
    onAttach?: () => void;
    /** Phone-only camera-capture affordance; shown beside attach at ≤640px. */
    onCamera?: () => void;
    /** Commit a pick from the "add context" menu: a client/no-choices provider
     *  (arg omitted), or a submenu choice (arg = the chosen value). */
    onPickContext?: (item: ContextMenuItem, arg?: string | null) => void;
    /** Load a server provider's submenu options on demand (when it hasChoices). */
    onRequestChoices?: (key: string) => Promise<ContextChoice[]>;
    /** Commit a file reference from the @ popover: the host attaches it as a
     *  context item (a chip) rather than inserting `@path` text. Absent, or for a
     *  non-file ref, the ref inserts inline as text. */
    onPickRef?: (ref: RefItem) => void;
    onRemoveAttachment?: (id: string) => void;
    onRemoveContext?: (key: string) => void;
    onInput?: (value: string) => void;
    /** First look at textarea keydowns; return true to consume the key. */
    onKeydown?: (e: KeyboardEvent) => boolean | void;
    /** Raw paste on the textarea; the caller decides whether to consume it (e.g.
     *  route a pasted image to attach) or leave native text-paste behavior. */
    onPaste?: (e: ClipboardEvent) => void;
    /** Bottom-row leading controls (model/effort, meta chip). */
    leading?: Snippet;
    /** Large-paste affordance shown in the attach row (e.g. paste-collapsed-to-attachment chip). */
    pasteAffordance?: Snippet;
  } = $props();

  const uid = nextUid();
  const listId = `composer-mentions-${uid}`;
  const idBase = `composer-ref-${uid}`;

  let ta = $state<HTMLTextAreaElement>();
  let root: HTMLDivElement | undefined;

  // Auto-grow: the textarea tracks its content height (floor = the `rows`
  // baseline, ceiling = the CSS max-height; past the ceiling it scrolls).
  // Measured with the transient height:auto trick - synchronous, so nothing
  // paints between the reset and the re-apply - then expressed through the
  // --th custom-property bridge.
  let taH = $state<number | null>(null);
  $effect(() => {
    void value;
    const el = ta;
    if (!el) return;
    el.style.height = 'auto';
    taH = el.scrollHeight + 2;
    el.style.height = '';
  });

  const mention = $state({
    open: false,
    active: 0,
    start: 0,
    end: 0,
    query: '',
    trigger: '' as '@' | '#' | '',
  });

  const sourcePrefixes = $derived(refSources.map((s) => s.prefix));

  // Prefix routing: when the @ query reads `<prefix> <subquery>` for a known
  // source, scope the popover to it (first match wins) and carry the subquery.
  // Requires the prefix to be followed by a space, so `@jira` still filters the
  // built-in list and `@jira auth` scopes to the jira source.
  const activeSource = $derived.by(() => {
    if (!mention.open || refSources.length === 0) return null;
    const q = mention.query;
    for (const src of refSources) {
      if (q.startsWith(`${src.prefix} `)) {
        return { src, sub: q.slice(src.prefix.length + 1).trimStart() };
      }
    }
    return null;
  });

  // The active source's fetched results + in-flight state. `searchToken` is a
  // plain counter (deliberately not reactive): each query bump invalidates the
  // previous request, so an out-of-order response is dropped and only the latest
  // query's results land. The debounce timer is cleared on every query change.
  let searchToken = 0;
  let searchLoading = $state(false);
  let searchItems = $state<RefItem[]>([]);
  $effect(() => {
    const active = activeSource;
    if (!active) {
      searchLoading = false;
      searchItems = [];
      return;
    }
    const token = ++searchToken;
    const { src, sub } = active;
    searchLoading = true;
    const timer = setTimeout(() => {
      src
        .search(sub)
        .then((results) => {
          if (token === searchToken) {
            searchItems = results;
            searchLoading = false;
          }
        })
        .catch(() => {
          // Autocomplete must never break the composer: a failed search shows empty.
          if (token === searchToken) {
            searchItems = [];
            searchLoading = false;
          }
        });
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  });

  const visibleItems = $derived.by(() => {
    if (!mention.open) return [];
    // In a scoped source the daemon already filtered by the subquery, so its
    // results show as-is (never re-filtered by the whole `prefix query` string).
    if (activeSource) return searchItems;
    const q = mention.query.toLowerCase();
    if (!q) return refItems;
    return refItems.filter(
      (it) =>
        it.label.toLowerCase().includes(q) ||
        it.id.toLowerCase().includes(q) ||
        (it.detail?.toLowerCase().includes(q) ?? false),
    );
  });
  // A scoped source keeps the popover open through its loading + empty states so
  // it can show "Searching…" / "No matches"; the built-in list opens only with hits.
  const inSourceSearch = $derived(mention.open && activeSource !== null);
  const showPopover = $derived(mention.open && (visibleItems.length > 0 || inSourceSearch));
  const popoverStatus = $derived.by(() => {
    if (!inSourceSearch || searchItems.length > 0) return undefined;
    return searchLoading ? 'Searching…' : 'No matches';
  });
  // Any attached context alone is sendable - a context-only message is valid.
  const canSend = $derived(value.trim().length > 0 || contextItems.length > 0);

  // "Add context" menu (a small popover over the button); closes on pick or an
  // outside click. A server provider with choices swaps the provider list for a
  // submenu of its options rather than committing the pick immediately.
  let ctxMenuOpen = $state(false);
  let ctxAnchor = $state<HTMLDivElement>();
  let submenu = $state<{ item: ContextMenuItem; choices: ContextChoice[] } | null>(null);

  // A context chip shows only its short label; clicking it previews the full
  // value (a whole file or fetched page) in a modal rather than in the row.

  function closeCtxMenu() {
    ctxMenuOpen = false;
    submenu = null;
  }

  async function chooseMenuItem(opt: ContextMenuItem) {
    // A large-option (picker) provider skips the inline submenu: the host opens
    // the Picker overlay off the plain onPickContext commit below.
    if (opt.kind === 'server' && opt.hasChoices && !opt.picker) {
      const choices = (await onRequestChoices?.(opt.key)) ?? [];
      // No options to pick (empty set, or a failed load) - nothing to open.
      if (choices.length === 0) closeCtxMenu();
      else submenu = { item: opt, choices };
      return;
    }
    closeCtxMenu();
    onPickContext?.(opt);
  }

  function chooseSubmenu(choice: ContextChoice) {
    const item = submenu?.item;
    closeCtxMenu();
    if (item) onPickContext?.(item, choice.value);
  }

  function handleInput(e: Event) {
    const el = e.currentTarget as HTMLTextAreaElement;
    const caret = el.selectionStart ?? el.value.length;
    // Pass the source prefixes so a `@<prefix> <query>` token keeps its spaces.
    const token = parseRefToken(el.value, caret, sourcePrefixes);
    // Open for a static list OR when a prefix source could take over the query.
    if (token && (refItems.length > 0 || refSources.length > 0)) {
      mention.open = true;
      mention.query = token.query;
      mention.trigger = token.trigger;
      mention.start = token.start;
      mention.end = token.end;
      mention.active = 0;
    } else {
      mention.open = false;
    }
    onInput?.(el.value);
  }

  // Touch devices (phones) treat Enter as a newline and send with the button,
  // like every mobile chat app; a physical-keyboard device keeps Enter-to-send.
  // Keyed on the input type, not width, so a narrow desktop window still sends.
  let isTouch = $state(false);
  $effect(() => {
    const mq = window.matchMedia('(pointer: coarse)');
    const sync = () => (isTouch = mq.matches);
    sync();
    mq.addEventListener('change', sync);
    return () => mq.removeEventListener('change', sync);
  });

  let lastSubmitAt = 0;

  /** Send and Stop are the same control, so a repeated Enter or a double-click
   *  lands on Stop as soon as the turn starts. Escape names Stop on its own and
   *  so is never guarded. */
  function stopIfArmed() {
    if (Date.now() - lastSubmitAt >= SEND_GUARD_MS) onStop?.();
  }

  function handleKeydown(e: KeyboardEvent) {
    // The caller's own popover (e.g. the slash-command menu) gets first look;
    // a truthy return means it consumed the key.
    if (onKeydown?.(e)) return;
    if (showPopover) {
      const nav = refNav(e.key, mention.active, visibleItems.length);
      if (nav.handled) {
        e.preventDefault();
        mention.active = nav.activeIndex;
        const item = visibleItems[mention.active];
        if (nav.select && item) selectMention(item);
        else if (nav.close) closeMention();
        return;
      }
      // A source search shown with no selectable row yet (Searching… / No matches)
      // swallows Enter/Escape, so a pending search never accidentally sends and
      // Escape dismisses the popover instead.
      if (inSourceSearch && (e.key === 'Enter' || e.key === 'Escape')) {
        e.preventDefault();
        if (e.key === 'Escape') closeMention();
        return;
      }
    }
    if (e.key === 'Enter' && !e.shiftKey && !isTouch) {
      e.preventDefault();
      // Mid-turn Enter with a draft queues it; with nothing typed it stops,
      // unless the send that emptied the draft just happened.
      if (streaming) {
        if (canSend && onQueue) submit(onQueue);
        else stopIfArmed();
      } else submit(onSend);
      return;
    }
    if (e.key === 'Escape' && streaming && !showPopover) {
      e.preventDefault();
      onStop?.();
    }
  }

  function submit(cb?: (text: string) => void) {
    if (!canSend) return;
    cb?.(value.trim());
    value = '';
    lastSubmitAt = Date.now();
    mention.open = false;
  }

  async function selectMention(item: RefItem) {
    const before = value.slice(0, mention.start);
    const after = value.slice(mention.end);
    mention.open = false;
    if (onPickRef && ATTACH_KINDS.has(item.kind)) {
      // Convergence: a file/session/plugin ref attaches as a context chip on the
      // host, so drop the @query trigger and insert nothing in its place.
      value = `${before}${after}`;
      onPickRef(item);
      await tick();
      ta?.setSelectionRange(before.length, before.length);
    } else {
      value = `${before}${item.label} ${after}`;
      await tick();
      const pos = before.length + item.label.length + 1;
      ta?.setSelectionRange(pos, pos);
    }
    ta?.focus();
  }

  function closeMention() {
    mention.open = false;
    ta?.focus();
  }

  /** Imperative focus for callers that own navigation (e.g. auto-focus on landing
   *  in a chat); moves the caret into the textarea. */
  export function focus() {
    ta?.focus();
  }

  $effect(() => {
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (root && !root.contains(target)) mention.open = false;
      if (ctxAnchor && !ctxAnchor.contains(target)) closeCtxMenu();
    };
    window.addEventListener('mousedown', onDown);
    return () => window.removeEventListener('mousedown', onDown);
  });
</script>

<div class="composer" class:is-queued={queued} bind:this={root}>
  <div class="attrow">
    {@render pasteAffordance?.()}
    <StagedStrip {attachments} {contextItems} {onRemoveAttachment} {onRemoveContext} />
    <Button variant="ghost" size="sm" onclick={() => onAttach?.()}>
      {#snippet icon()}<Icon name="plus" />{/snippet}attach
    </Button>
    {#if onCamera}
      <span class="cam-only">
        <Button variant="ghost" size="sm" aria-label="Take a photo" onclick={() => onCamera?.()}>
          {#snippet icon()}<Icon name="camera" />{/snippet}photo
        </Button>
      </span>
    {/if}
    {#if contextMenu.length > 0}
      <div class="ctxwrap" bind:this={ctxAnchor}>
        <Button
          variant="ghost"
          size="sm"
          data-testid={TESTID.composerContext}
          aria-label="Add context"
          aria-haspopup="menu"
          aria-expanded={ctxMenuOpen}
          onclick={() => (ctxMenuOpen ? closeCtxMenu() : (ctxMenuOpen = true))}
        >
          {#snippet icon()}<Icon name="sparkle" />{/snippet}context
        </Button>
        {#if ctxMenuOpen && submenu}
          <div class="ctxmenu" role="menu" data-testid={TESTID.composerContextSubmenu}>
            <button type="button" class="ctxback" onclick={() => (submenu = null)}>
              <span class="chev" aria-hidden="true">‹</span>{submenu.item.label}
            </button>
            {#each submenu.choices as choice (choice.value)}
              <button
                type="button"
                role="menuitem"
                data-testid={TESTID.composerContextChoice(submenu.item.key, choice.value)}
                onclick={() => chooseSubmenu(choice)}
              >
                {choice.label}
              </button>
            {/each}
          </div>
        {:else if ctxMenuOpen}
          <div class="ctxmenu" role="menu" data-testid={TESTID.composerContextMenu}>
            {#each contextMenu as opt (opt.key)}
              <button
                type="button"
                role="menuitem"
                data-testid={TESTID.composerContextOption(opt.key)}
                aria-haspopup={opt.hasChoices ? 'menu' : undefined}
                onclick={() => chooseMenuItem(opt)}
              >
                <Icon name={opt.icon} />{opt.label}
                {#if opt.hasChoices}<span class="chev chev-r" aria-hidden="true">›</span>{/if}
              </button>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
    <span class="qnote">
      <Icon name="clock" />queued — sends when this turn finishes
    </span>
  </div>

  <div class="inwrap">
    <RefAutocomplete
      items={visibleItems}
      bind:activeIndex={mention.active}
      open={showPopover}
      floating
      {idBase}
      {listId}
      label="Reference suggestions"
      status={popoverStatus}
      onSelect={(item) => selectMention(item)}
    />
    <textarea
      class="t-input"
      bind:this={ta}
      bind:value
      {rows}
      style="--th:{taH == null ? 'auto' : `${taH}px`}"
      {placeholder}
      aria-label="Message"
      aria-autocomplete="list"
      aria-controls={listId}
      aria-activedescendant={showPopover ? `${idBase}-${mention.active}` : undefined}
      oninput={handleInput}
      onkeydown={handleKeydown}
      onpaste={onPaste}></textarea>
  </div>

  <div class="btmrow">
    {@render leading?.()}
    {#if hint}<span class="hint">{hint}</span>{/if}
    <div class="grow"></div>
    {#if streaming && onQueue}
      <Button
        size="sm"
        variant="ghost"
        data-act="queue"
        aria-label="Queue message for after this turn"
        disabled={!canSend}
        onclick={() => submit(onQueue)}
      >
        {#snippet icon()}<Icon name="clock" />{/snippet}Queue
      </Button>
    {/if}
    <Button
      size="sm"
      variant={streaming ? 'danger' : 'pri'}
      data-act={streaming ? 'stop' : 'send'}
      aria-label={streaming ? 'Stop streaming' : 'Send message'}
      onclick={() => (streaming ? stopIfArmed() : submit(onSend))}
    >
      {#snippet icon()}<Icon name={streaming ? 'stop' : 'send'} />{/snippet}{streaming
        ? 'Stop'
        : 'Send'}
    </Button>
  </div>
</div>

{#if showKbd}
  <div class="kbd-strip" aria-hidden="true">
    <span><span class="t-kbd">/</span> search</span>
    <span><span class="t-kbd">j</span><span class="t-kbd">k</span> sessions</span>
    <span><span class="t-kbd">⏎</span> send</span>
    <span><span class="t-kbd">⇧⏎</span> newline</span>
    <span><span class="t-kbd">esc</span> stop stream</span>
  </div>
{/if}

<style>
  /* composer */
  .composer {
    flex: none;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    padding: 9px 12px 8px;
    display: grid;
    gap: 7px;
    position: relative;
  }
  .attrow {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
  }
  .inwrap {
    position: relative;
  }
  .btmrow {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .hint {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .qnote {
    display: none;
    align-items: center;
    gap: 6px;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--st-queue);
  }
  .composer.is-queued .qnote {
    display: inline-flex;
  }
  /* Camera capture is a phone affordance (desktop has no environment camera and
     drag/drop + attach already cover it); hidden until the phone breakpoint. */
  .cam-only {
    display: none;
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  /* "Add context" menu: a small popover floated above its trigger button. */
  .ctxwrap {
    position: relative;
    display: inline-flex;
  }
  .ctxmenu {
    position: absolute;
    left: 0;
    bottom: calc(100% + 4px);
    z-index: 40;
    display: flex;
    flex-direction: column;
    min-width: 148px;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    padding: 4px;
  }
  .ctxmenu button {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 9px;
    border: 0;
    border-radius: var(--r-sm);
    background: transparent;
    color: var(--tx0);
    font: 500 var(--fs-sm) var(--font-ui);
    text-align: left;
    cursor: pointer;
  }
  .ctxmenu button:hover {
    background: var(--bg4);
  }
  /* A hasChoices row points into its submenu; the chevron sits at the row's end. */
  .chev-r {
    margin-left: auto;
  }
  .chev {
    color: var(--tx3);
    font: 600 var(--fs-md) var(--font-ui);
    line-height: 1;
  }
  /* Submenu header doubles as the back-to-providers affordance. */
  .ctxmenu .ctxback {
    color: var(--tx2);
    font-weight: 600;
  }

  /* kbd strip */
  .kbd-strip {
    display: flex;
    gap: 16px;
    align-items: center;
    padding: 5px 14px;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    flex: none;
    overflow-x: auto;
    white-space: nowrap;
  }
  .kbd-strip span {
    display: inline-flex;
    gap: 5px;
    align-items: center;
  }
  /* Shared primitives. t-btn was deduped onto the
     Button component (buttons/Button.svelte); t-kbd is global (tokens.css). Kept
     inline: t-input (a <textarea>, while the shared Input renders an <input>
     only); t-chip (its file/x icons render at 13px, but Chip forces its icon to
     10px / its × to 9px, which would shrink them). */
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
  textarea.t-input {
    /* Content-tracked height (--th bridge from the auto-grow effect), floored
       at the two-row baseline and capped so a huge paste scrolls instead of
       swallowing the conversation. */
    height: var(--th, auto);
    min-height: 34px;
    max-height: 38vh;
    overflow-y: auto;
    padding: 7px 9px;
    resize: none;
    line-height: 1.5;
    font-family: var(--font-ui);
  }

  /* Narrow: the composer sheds its hint + kbd strip. */
  @media (max-width: 640px) {
    .composer {
      padding: 7px 10px 6px;
      gap: 5px;
    }
    .hint,
    .kbd-strip {
      display: none;
    }
    .cam-only {
      display: inline-flex;
    }
  }
</style>
