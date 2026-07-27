<script lang="ts">
  // Static demo of every input/selector variant, side by side. No
  // switcher - every state is visible at once per the gallery contract.
  import Field from './Field.svelte';
  import Input from './Input.svelte';
  import Select from './Select.svelte';
  import Seg from './Seg.svelte';
  import Switch from './Switch.svelte';
  import SearchInput from './SearchInput.svelte';

  let searchValue = $state('');
  let focusedValue = $state('focused');
  let defaultValue = $state('');
  let disabledValue = $state('disabled');
  let webhookValue = $state('whsec_…truncated');
  let model = $state('sonnet-4.6');
  let effort = $state('med');
  let switchOn = $state(true);
  let switchOff = $state(false);
</script>

<section data-testid="gallery-inputs">
  <h3>Inputs · selectors</h3>

  <div class="demo-row">
    <div class="demo">
      <span class="lbl">search · with shortcut hint</span>
      <SearchInput
        bind:value={searchValue}
        ariaLabel="Search sessions"
        placeholder="search sessions…"
        shortcutKey="/"
      />
    </div>
  </div>

  <div class="demo-row">
    <div class="demo">
      <span class="lbl">focused</span>
      <Input bind:value={focusedValue} focused ariaLabel="Focused example" />
    </div>
    <div class="demo">
      <span class="lbl">default</span>
      <Input bind:value={defaultValue} placeholder="default" ariaLabel="Default example" />
    </div>
    <div class="demo">
      <span class="lbl">disabled</span>
      <Input bind:value={disabledValue} disabled ariaLabel="Disabled example" />
    </div>
  </div>

  <div class="demo-row">
    <div class="demo demo--narrow">
      <span class="lbl">field · invalid + error</span>
      <Field id="whk-demo" label="webhook secret" error="signature check failed on last delivery">
        {#snippet children(describedBy)}
          <Input
            id="whk-demo"
            bind:value={webhookValue}
            invalid
            mono
            ariaDescribedby={describedBy}
          />
        {/snippet}
      </Field>
    </div>
  </div>

  <div class="demo-row">
    <div class="demo">
      <span class="lbl">select</span>
      <Select
        bind:value={model}
        options={['sonnet-4.6', 'opus-4.6', 'haiku-4.5']}
        ariaLabel="Model"
      />
    </div>
    <div class="demo">
      <span class="lbl">seg · reasoning effort</span>
      <Seg bind:value={effort} options={['low', 'med', 'high']} ariaLabel="Reasoning effort" />
    </div>
    <div class="demo">
      <span class="lbl">switch · on</span>
      <Switch bind:checked={switchOn} ariaLabel="Enabled" />
    </div>
    <div class="demo">
      <span class="lbl">switch · off</span>
      <Switch bind:checked={switchOff} ariaLabel="Disabled" />
    </div>
  </div>
</section>

<style>
  section {
    display: grid;
    gap: var(--sp-4);
  }
  h3 {
    margin: 0;
    font: 600 var(--fs-xs) / 1 var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--tx2);
  }
  .demo-row {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: var(--sp-4);
  }
  .demo {
    display: grid;
    gap: var(--sp-1);
    min-width: 0;
  }
  .lbl {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
  }

  .demo--narrow {
    max-width: 280px;
  }
</style>
