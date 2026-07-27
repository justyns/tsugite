<script lang="ts">
  // Search box (.t-search): icon + input[type=search] + a kbd hint chip
  // naming the global focus shortcut. When `shortcutKey` is set the hint is
  // wired to a real document-level listener, so the chip never lies about
  // what the key does. The listener steps aside whenever focus is already
  // in an editable field.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { pwmIgnore } from './pwmIgnore';
  import { isEditableTarget } from '$lib/dom';

  let {
    value = $bindable(''),
    placeholder = 'search…',
    ariaLabel,
    shortcutKey,
    disabled = false,
  }: {
    value?: string;
    placeholder?: string;
    ariaLabel: string;
    shortcutKey?: string;
    disabled?: boolean;
  } = $props();

  let inputEl: HTMLInputElement;

  $effect(() => {
    if (!shortcutKey) return;
    function onKeydown(e: KeyboardEvent) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key !== shortcutKey) return;
      if (isEditableTarget(e.target)) return;
      e.preventDefault();
      inputEl?.focus();
    }
    window.addEventListener('keydown', onKeydown);
    return () => window.removeEventListener('keydown', onKeydown);
  });
</script>

<div class="t-search">
  <Icon name="search" />
  <input
    bind:this={inputEl}
    bind:value
    type="search"
    class="t-input"
    {placeholder}
    aria-label={ariaLabel}
    {disabled}
    {...pwmIgnore}
  />
  {#if shortcutKey}
    <span class="t-kbd" aria-hidden="true">{shortcutKey}</span>
  {/if}
</div>

<style>
  .t-search {
    position: relative;
    display: flex;
    align-items: center;
  }
  .t-search :global(.ic) {
    position: absolute;
    left: 8px;
    color: var(--tx3);
    pointer-events: none;
  }
  .t-search .t-input {
    padding-left: 26px;
    padding-right: 30px;
  }
  .t-search .t-kbd {
    position: absolute;
    right: 7px;
    pointer-events: none;
  }
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
  /* type=search opts out of Chromium's password manager but pulls in the UA
     clear button; drop it so it can't sit under the kbd hint chip. */
  .t-input::-webkit-search-cancel-button,
  .t-input::-webkit-search-decoration {
    -webkit-appearance: none;
    appearance: none;
  }
  .t-input:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
  .t-input[disabled] {
    opacity: 0.5;
    pointer-events: none;
  }
</style>
