<script lang="ts">
  // Keyboard-shortcuts cheat sheet: the discoverable index of every shell chord.
  // Rides the shared Modal (focus-trap + Escape + scrim) and renders SHORTCUTS
  // grouped, each chord as .t-kbd chips - so it stays in lockstep with the one
  // source of truth in shortcuts.ts rather than duplicating the list.
  import Modal from '$lib/components/overlays/Modal.svelte';
  import { SHORTCUTS, keyLabel, type ShortcutGroup } from '$lib/shell/shortcuts';

  let { open = $bindable(false) }: { open?: boolean } = $props();

  const GROUP_ORDER: ShortcutGroup[] = ['Global', 'Navigation', 'Chat'];
  const grouped = GROUP_ORDER.map((group) => ({
    group,
    rows: SHORTCUTS.filter((s) => s.group === group),
  })).filter((section) => section.rows.length > 0);
</script>

<Modal {open} title="Keyboard shortcuts" onclose={() => (open = false)}>
  <div class="sc">
    {#each grouped as section (section.group)}
      <section class="sc-grp">
        <h4>{section.group}</h4>
        {#each section.rows as row (row.label)}
          <div class="sc-row">
            <span class="sc-lbl">{row.label}</span>
            <span class="sc-keys">
              {#each row.keys as k (k)}<span class="t-kbd">{keyLabel(k)}</span>{/each}
            </span>
          </div>
        {/each}
      </section>
    {/each}
  </div>
</Modal>

<style>
  .sc {
    display: grid;
    gap: 16px;
  }
  .sc-grp {
    display: grid;
    gap: 3px;
  }
  .sc-grp h4 {
    margin: 0 0 3px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .sc-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    min-height: 26px;
  }
  .sc-lbl {
    color: var(--tx1);
    font-size: var(--fs-md);
  }
  .sc-keys {
    display: inline-flex;
    gap: 4px;
    flex: none;
  }
</style>
