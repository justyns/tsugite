<script lang="ts">
  // Controlled enable/disable toggle. The shared <Switch> is bindable-only (it
  // flips its own `checked` on click and offers no change callback), which
  // doesn't fit a row whose source of truth is server state and whose click must
  // hit an endpoint - so this implements `.t-sw` with a real
  // `onToggle`. Same accessible shape: <button role="switch">, aria-checked, and
  // the thumb slides position (state never signalled by color alone).
  let {
    checked,
    ariaLabel,
    onToggle,
    testid,
  }: {
    checked: boolean;
    ariaLabel: string;
    onToggle: (next: boolean) => void;
    testid?: string;
  } = $props();
</script>

<button
  type="button"
  class="t-sw"
  role="switch"
  aria-checked={checked}
  aria-label={ariaLabel}
  data-testid={testid}
  onclick={(e) => {
    e.stopPropagation();
    onToggle(!checked);
  }}
></button>

<style>
  .t-sw {
    position: relative;
    width: 30px;
    height: 17px;
    border-radius: var(--r-full);
    background: var(--bg4);
    border: 1px solid var(--bd1);
    cursor: pointer;
    transition: background var(--t-2) var(--ease);
    flex: none;
    padding: 0;
  }
  .t-sw::after {
    content: '';
    position: absolute;
    top: 1.5px;
    left: 2px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--tx1);
    transition:
      translate var(--t-2) var(--ease),
      background var(--t-2);
  }
  .t-sw[aria-checked='true'] {
    background: color-mix(in oklab, var(--st-ok) 55%, var(--bg4));
    border-color: transparent;
  }
  .t-sw[aria-checked='true']::after {
    translate: 12px 0;
    background: var(--bg0);
  }
</style>
