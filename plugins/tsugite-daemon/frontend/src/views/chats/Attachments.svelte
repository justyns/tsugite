<script lang="ts">
  // Files/photos the person attached to a user message, rendered once under the
  // bubble: images as clickable thumbnails (click opens a full-image lightbox),
  // other files as compact chips that open the file in the workspace files view.
  import Scrim from '$lib/components/overlays/Scrim.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';
  import { navigate } from '$lib/router.svelte';
  import { trapFocus } from '$lib/actions/trapFocus';
  import AttachmentThumb from '$lib/components/media/AttachmentThumb.svelte';
  import type { TurnAttachment } from './turns';

  let {
    agent,
    attachments,
  }: {
    agent: string;
    attachments: TurnAttachment[];
  } = $props();

  // The recorder stores the content-type CLASS ("image"), not a mime type; accept
  // a mime-ish value too so a future recording shape still thumbnails correctly.
  const isImage = (a: TurnAttachment): boolean => a.type === 'image' || a.type.startsWith('image/');

  let lightbox = $state<{ att: TurnAttachment; url: string } | null>(null);
  let restoreTo: HTMLElement | null = null;
  let closeEl = $state<HTMLButtonElement | null>(null);

  function openLightbox(att: TurnAttachment, url: string): void {
    restoreTo = document.activeElement as HTMLElement | null;
    lightbox = { att, url };
  }
  function closeLightbox(): void {
    lightbox = null;
    restoreTo?.focus();
    restoreTo = null;
  }
  function onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.stopPropagation();
      closeLightbox();
    }
  }
  function openInFiles(path: string): void {
    closeLightbox();
    navigate('files', { agent, path });
  }

  // Land focus on the close control once the dialog renders.
  $effect(() => {
    if (lightbox) closeEl?.focus();
  });
</script>

<div class="attachments">
  {#each attachments as a, i (i)}
    {#if isImage(a)}
      <AttachmentThumb {agent} name={a.name} path={a.path} onopen={(url) => openLightbox(a, url)} />
    {:else}
      <button
        type="button"
        class="chip"
        data-testid={TESTID.chatAttachmentChip}
        title={a.name}
        onclick={() => openInFiles(a.path)}
      >
        <Icon name="file" size={13} />
        <span class="chip-name">{a.name}</span>
      </button>
    {/if}
  {/each}
</div>

<Scrim open={!!lightbox} onclose={closeLightbox}>
  {#if lightbox}
    <div
      class="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={lightbox.att.name}
      data-testid={TESTID.chatAttachmentLightbox}
      tabindex="-1"
      onkeydown={onKeydown}
      use:trapFocus
    >
      <div class="lb-bar">
        <span class="lb-name" title={lightbox.att.name}>{lightbox.att.name}</span>
        <button
          type="button"
          class="lb-btn"
          onclick={() => lightbox && openInFiles(lightbox.att.path)}
        >
          <Icon name="files" size={14} />open in files
        </button>
        <button
          type="button"
          class="lb-btn lb-btn--icon"
          bind:this={closeEl}
          aria-label="Close"
          onclick={closeLightbox}
        >
          <Icon name="x" size={16} />
        </button>
      </div>
      <img class="lb-img" src={lightbox.url} alt={lightbox.att.name} />
    </div>
  {/if}
</Scrim>

<style>
  .attachments {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    /* Never widen the bubble: the row wraps and each item is bounded. */
    max-width: 100%;
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 260px;
    padding: 5px 10px;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    color: var(--tx1);
    font: 500 var(--fs-xs) var(--font-mono);
    cursor: pointer;
  }
  .chip:hover {
    border-color: var(--acc);
    color: var(--tx0);
  }
  .chip-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .lightbox {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: min(92vw, 1100px);
    max-height: 90vh;
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-3);
    padding: 12px;
  }
  .lb-bar {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .lb-name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font: 600 var(--fs-sm) var(--font-mono);
    color: var(--tx1);
  }
  .lb-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: none;
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    color: var(--tx2);
    font: 500 var(--fs-xs) var(--font-mono);
    cursor: pointer;
    padding: 5px 10px;
  }
  .lb-btn:hover {
    color: var(--tx0);
    border-color: var(--acc);
  }
  .lb-btn--icon {
    padding: 5px;
  }
  .lb-img {
    /* Bounded to the dialog; a huge image scrolls inside its own box. */
    max-width: 100%;
    max-height: calc(90vh - 60px);
    object-fit: contain;
    overflow: auto;
    border-radius: var(--r-md);
  }
</style>
