<script lang="ts">
  // Secrets: write-only name/value store. GET /api/secrets/ returns names only
  // (no GET-single-value route exists anywhere), so this view never holds or
  // renders a value after the moment it's typed into SetSecretModal. There is
  // no server-side unlock/lock concept and no reference-tracking data - see the
  // capability note in the empty state and the caption under the table rather
  // than a LockCard gate or fabricated "referenced by" chips.
  import { TESTID } from '$lib/testids';
  import { secrets } from '$lib/stores/secrets.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import SecTable, { type SecretRow } from '$lib/components/connstates/SecTable.svelte';
  import SetSecretModal, {
    type SetSecretPayload,
  } from '$lib/components/connstates/SetSecretModal.svelte';
  import Scrim from '$lib/components/overlays/Scrim.svelte';
  import Modal from '$lib/components/overlays/Modal.svelte';

  $effect(() => {
    secrets.load();
  });

  type SetModalState = { mode: 'add' } | { mode: 'rotate'; name: string };
  let setModal = $state<SetModalState | null>(null);
  let deleteTarget = $state<string | null>(null);

  // The backend lists names only (secrets.svelte.ts); `scope` is honestly
  // 'global' - the store is one flat namespace.
  const rows = $derived<SecretRow[]>(
    secrets.names.map((name) => ({ name, provenance: '', scope: 'global' })),
  );

  function messageOf(err: unknown): string {
    return err instanceof Error ? err.message : String(err);
  }

  function openAdd() {
    setModal = { mode: 'add' };
  }
  function openRotate(row: SecretRow) {
    setModal = { mode: 'rotate', name: row.name };
  }
  function closeSetModal() {
    setModal = null;
  }

  async function saveSecret(payload: SetSecretPayload) {
    const rotating = setModal?.mode === 'rotate';
    try {
      await secrets.set(payload.name, payload.value);
      toasts.push('ok', rotating ? `${payload.name} rotated` : `${payload.name} added`);
      setModal = null;
    } catch (err) {
      toasts.push('err', 'Save failed', { body: messageOf(err) });
    }
  }

  function requestDelete(row: SecretRow) {
    deleteTarget = row.name;
  }
  function cancelDelete() {
    deleteTarget = null;
  }
  async function confirmDelete() {
    const name = deleteTarget;
    if (!name) return;
    try {
      await secrets.remove(name);
      toasts.push('ok', `${name} deleted`);
      deleteTarget = null;
    } catch (err) {
      toasts.push('err', 'Delete failed', { body: messageOf(err) });
    }
  }
</script>

<section data-testid={TESTID.view('secrets')} aria-labelledby="secrets-h">
  <div class="head">
    <h2 id="secrets-h">Secrets</h2>
    <span class="grow"></span>
    <Button variant="pri" size="sm" data-testid={TESTID.secretsAdd} onclick={openAdd}>
      {#snippet icon()}<Icon name="plus" />{/snippet}
      Add secret
    </Button>
  </div>

  <div class="callout">
    <Icon name="alert" />
    <div>
      <b>Write-only.</b> Once saved, a value can never be displayed again — only rotated (overwritten)
      or deleted. Agents read secrets at run time; they are never rendered to the UI or written to logs.
    </div>
  </div>

  {#if secrets.error}
    <PaneState kind="error" title="Couldn't load secrets">
      {#snippet icon()}<Icon name="alert" />{/snippet}
      {#snippet hint()}<span class="mono">{secrets.error}</span>{/snippet}
      {#snippet actions()}
        <Button size="sm" onclick={() => secrets.load()}>
          {#snippet icon()}<Icon name="retry" />{/snippet}
          Retry
        </Button>
      {/snippet}
    </PaneState>
  {:else if secrets.loading && rows.length === 0}
    <PaneState kind="loading" />
  {:else if rows.length === 0}
    <PaneState kind="empty" title="No secrets stored">
      {#snippet icon()}<Icon name="key" />{/snippet}
      {#snippet hint()}
        Add one from the button above. If this daemon uses the default backend, secrets are read
        from the process environment only and never listed here.
      {/snippet}
    </PaneState>
  {:else}
    <SecTable {rows} onRotate={openRotate} onDelete={requestDelete} />
  {/if}
</section>

{#if setModal}
  <Scrim open={true} onclose={closeSetModal}>
    <SetSecretModal
      mode={setModal.mode}
      name={setModal.mode === 'rotate' ? setModal.name : ''}
      onCancel={closeSetModal}
      onSave={saveSecret}
    />
  </Scrim>
{/if}

<Modal
  open={deleteTarget !== null}
  tone="danger"
  title={deleteTarget ? `Delete ${deleteTarget}?` : 'Delete secret?'}
  onclose={cancelDelete}
>
  Any agent or skill reading <code>{deleteTarget}</code> at runtime gets nothing back after this.
  This cannot be undone.
  {#snippet footer()}
    <Button data-autofocus onclick={cancelDelete}>Cancel</Button>
    <Button variant="danger" onclick={confirmDelete}>Delete secret</Button>
  {/snippet}
</Modal>

<style>
  /* The app-view host clips overflow, so the view owns its scrolling (same
     finding as views/plugins). */
  section {
    display: grid;
    gap: var(--sp-3);
    padding: 14px 16px 26px;
    align-content: start;
    flex: 1;
    min-height: 0;
    overflow-y: auto;
  }
  .mono {
    font-family: var(--font-mono);
  }
  .head {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }
  .head h2 {
    margin: 0;
    font: 600 var(--fs-xl) var(--font-ui);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  /* .t-callout kept local: no shared Callout component owns this
     class across views. */
  .callout {
    display: flex;
    gap: 9px;
    align-items: flex-start;
    padding: 9px 12px;
    border: 1px solid color-mix(in oklab, var(--st-warn) 42%, var(--bd1));
    border-radius: var(--r-md);
    background: color-mix(in oklab, var(--st-warn) 8%, var(--bg1));
    font-size: var(--fs-sm);
    color: var(--tx2);
    line-height: 1.5;
    text-wrap: pretty;
  }
  .callout :global(.ic) {
    flex: none;
    margin-top: 1px;
    width: 14px;
    height: 14px;
    color: var(--st-warn);
  }
  .callout b {
    color: var(--tx1);
    font-weight: 600;
  }
</style>
