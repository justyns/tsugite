<script lang="ts">
  // Toast stack host.
  // "Bottom-right stack, role=status, 6s auto-dismiss paused on hover." Mount
  // this once (app shell); push from anywhere via the `toasts` store.
  import Toast from './Toast.svelte';
  import { toasts } from './toast-store.svelte';
</script>

<div class="t-toasts" role="status">
  {#each toasts.items as item (item.id)}
    <Toast
      variant={item.variant}
      title={item.title}
      body={item.body}
      icon={item.icon}
      actionLabel={item.actionLabel}
      onAction={item.onAction}
      sticky={item.sticky}
      onDismiss={() => toasts.dismiss(item.id)}
    />
  {/each}
</div>

<style>
  .t-toasts {
    position: fixed;
    bottom: 16px;
    right: 16px;
    z-index: 300;
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: min(340px, calc(100vw - 32px));
  }
</style>
