<script lang="ts">
  // One unified strip of everything staged on the composer: uploaded files and
  // attached context items. Images preview as thumbnails (tap to enlarge), other
  // files and context show as chips. Past a threshold the extras collapse behind
  // a "+N more" toggle so the row stays compact. Attachments and context are one
  // payload type on the backend now; here they still read distinctly (a photo you
  // sent vs a session you referenced) but share one row.
  import Icon from '$lib/components/icon/Icon.svelte';
  import Modal from '$lib/components/overlays/Modal.svelte';
  import Scrim from '$lib/components/overlays/Scrim.svelte';
  import AttachmentThumb from '$lib/components/media/AttachmentThumb.svelte';
  import { TESTID } from '$lib/testids';
  import type { Attachment, ContextChip } from './types';

  let {
    agent,
    attachments = [],
    contextItems = [],
    onRemoveAttachment,
    onRemoveContext,
  }: {
    /** Owning agent, needed to load a staged image's bytes from its workspace.
     *  Absent (e.g. the component gallery) falls image attachments back to chips. */
    agent?: string;
    attachments?: Attachment[];
    contextItems?: ContextChip[];
    onRemoveAttachment?: (id: string) => void;
    onRemoveContext?: (key: string) => void;
  } = $props();

  const IMAGE_EXT = /\.(png|jpe?g|gif|webp|svg|avif|bmp)$/i;
  const isImage = (name: string): boolean => IMAGE_EXT.test(name);

  const MAX_VISIBLE = 6;
  let expanded = $state(false);

  type Row =
    { kind: 'att'; id: string; att: Attachment } | { kind: 'ctx'; id: string; ctx: ContextChip };
  const rows = $derived<Row[]>([
    ...attachments.map((a): Row => ({ kind: 'att', id: `att:${a.id}`, att: a })),
    ...contextItems.map((c): Row => ({ kind: 'ctx', id: `ctx:${c.key}`, ctx: c })),
  ]);
  const overflow = $derived(rows.length > MAX_VISIBLE);
  const visible = $derived(!overflow || expanded ? rows : rows.slice(0, MAX_VISIBLE));
  const hidden = $derived(rows.length - MAX_VISIBLE);

  let preview = $state<ContextChip | null>(null);
  let lightbox = $state<{ url: string; name: string } | null>(null);
</script>

{#each visible as row (row.id)}
  {#if row.kind === 'att'}
    {#if agent && isImage(row.att.name)}
      <span class="staged-thumb">
        <AttachmentThumb
          {agent}
          name={row.att.name}
          path={`uploads/${row.att.name}`}
          size={46}
          onopen={(url) => (lightbox = { url, name: row.att.name })}
        />
        {#if onRemoveAttachment}
          <button
            type="button"
            class="thumb-x"
            aria-label={`Remove attachment ${row.att.name}`}
            onclick={() => onRemoveAttachment?.(row.att.id)}
          >
            <Icon name="x" size={11} />
          </button>
        {/if}
      </span>
    {:else}
      <span class="t-chip">
        <Icon name="file" />
        {row.att.name}{row.att.size ? ` · ${row.att.size}` : ''}
        {#if onRemoveAttachment}
          <button
            type="button"
            class="x"
            aria-label={`Remove attachment ${row.att.name}`}
            onclick={() => onRemoveAttachment?.(row.att.id)}
          >
            <Icon name="x" />
          </button>
        {/if}
      </span>
    {/if}
  {:else}
    <span class="t-chip" data-testid={TESTID.composerContextChip(row.ctx.key)}>
      <Icon name={row.ctx.icon ?? 'pin'} />
      <button
        type="button"
        class="chip-label"
        title={row.ctx.label}
        aria-haspopup="dialog"
        onclick={() => (preview = row.ctx)}
      >
        {row.ctx.label}
      </button>
      {#if onRemoveContext}
        <button
          type="button"
          class="x"
          aria-label={`Remove ${row.ctx.label} context`}
          onclick={() => onRemoveContext?.(row.ctx.key)}
        >
          <Icon name="x" />
        </button>
      {/if}
    </span>
  {/if}
{/each}

{#if overflow}
  <button
    type="button"
    class="t-chip more"
    data-testid={TESTID.composerStagedMore}
    onclick={() => (expanded = !expanded)}
  >
    {expanded ? 'show less' : `+${hidden} more`}
  </button>
{/if}

{#if lightbox}
  <Scrim open onclose={() => (lightbox = null)}>
    <div class="lightbox" data-testid={TESTID.chatAttachmentLightbox}>
      <img src={lightbox.url} alt={lightbox.name} />
    </div>
  </Scrim>
{/if}

<Modal open={preview !== null} title={preview?.label ?? ''} onclose={() => (preview = null)}>
  <!-- data-autofocus lands focus inside the dialog (no other focusable here), so
       Esc closes it and focus restores to the chip. -->
  <pre class="ctx-preview" tabindex="-1" data-autofocus>{preview?.value ?? ''}</pre>
</Modal>

<style>
  .t-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
    cursor: default;
  }
  .t-chip .x {
    cursor: pointer;
    color: var(--tx3);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    background: none;
    border: 0;
    padding: 0;
  }
  .t-chip .x:hover {
    color: var(--st-err);
  }
  /* The chip's label truncates (its value can be a whole file); a wider one would
     shove the attach/context buttons off the row. Clicking previews the full value. */
  .t-chip .chip-label {
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    background: none;
    border: 0;
    padding: 0;
    color: inherit;
    font: inherit;
    cursor: pointer;
  }
  .t-chip .chip-label:hover {
    color: var(--tx0);
  }
  .more {
    cursor: pointer;
    color: var(--tx2);
  }
  .more:hover {
    color: var(--tx0);
  }
  .ctx-preview {
    margin: 0;
    max-height: 60vh;
    overflow: auto;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font: 400 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
  }

  .staged-thumb {
    position: relative;
    display: inline-flex;
  }
  .thumb-x {
    position: absolute;
    top: -5px;
    right: -5px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 17px;
    height: 17px;
    padding: 0;
    border: 1px solid var(--bd1);
    border-radius: 50%;
    background: var(--bg2);
    color: var(--tx2);
    cursor: pointer;
  }
  .thumb-x:hover {
    color: var(--st-err);
    border-color: var(--st-err);
  }

  .lightbox {
    display: flex;
    align-items: center;
    justify-content: center;
    max-width: min(92vw, 900px);
    max-height: 88vh;
  }
  .lightbox img {
    max-width: 100%;
    max-height: 88vh;
    object-fit: contain;
    border-radius: var(--r-md);
  }

  @media (max-width: 640px) {
    .t-chip .chip-label {
      max-width: 130px;
    }
  }
</style>
