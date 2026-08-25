<script lang="ts">
  // One image attachment's thumbnail. Loads the workspace bytes authenticated
  // (Bearer is header-only, so an <img src> can't reach the raw endpoint), wraps
  // them in an object URL, and revokes it on teardown so a long chat never leaks
  // URLs. A deleted upload resolves to a broken-file placeholder, never a throw.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';
  import { loadWorkspaceObjectURL } from '$lib/media/workspaceImage';

  let {
    name,
    path,
    size = 76,
    onopen,
  }: {
    name: string;
    path: string;
    /** Square edge in px. Sent-message thumbs use the default; the composer's
     *  staging strip renders smaller ones. */
    size?: number;
    /** Fired with the loaded object URL on click. The URL stays valid because
     *  this thumb keeps it alive until it unmounts (the lightbox renders over
     *  the still-mounted thumb). */
    onopen: (url: string) => void;
  } = $props();

  let url = $state<string | null>(null);
  let failed = $state(false);

  $effect(() => {
    let mine: string | null = null;
    let cancelled = false;
    loadWorkspaceObjectURL(path)
      .then((u) => {
        if (cancelled) {
          URL.revokeObjectURL(u);
          return;
        }
        mine = u;
        url = u;
      })
      .catch(() => {
        if (!cancelled) failed = true;
      });
    return () => {
      cancelled = true;
      if (mine) URL.revokeObjectURL(mine);
    };
  });
</script>

{#if failed}
  <span
    class="thumb thumb--broken"
    style={`width:${size}px;height:${size}px`}
    title={name}
    data-testid={TESTID.chatAttachmentImage}
  >
    <Icon name="alert" size={16} />
  </span>
{:else}
  <button
    type="button"
    class="thumb"
    class:is-loading={!url}
    style={`width:${size}px;height:${size}px`}
    data-testid={TESTID.chatAttachmentImage}
    disabled={!url}
    aria-label={`View ${name}`}
    onclick={() => url && onopen(url)}
  >
    {#if url}<img src={url} alt={name} />{/if}
  </button>
{/if}

<style>
  .thumb {
    width: 76px;
    height: 76px;
    padding: 0;
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    background: var(--bg1);
    overflow: hidden;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--tx3);
  }
  .thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .thumb.is-loading {
    cursor: default;
    animation: pulse 1.2s ease-in-out infinite;
  }
  .thumb:not(.is-loading):hover {
    border-color: var(--acc);
  }
  .thumb--broken {
    cursor: default;
  }

  @keyframes pulse {
    50% {
      opacity: 0.5;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .thumb.is-loading {
      animation: none;
    }
  }
</style>
