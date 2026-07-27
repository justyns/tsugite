<script lang="ts">
  // Base text input (.t-input). States: default, focused (native :focus, or
  // `focused` to force the ring for e.g. a gallery), disabled, invalid. The
  // invalid ring alone never carries the full error signal - pair it with
  // Field's icon+text message via `ariaDescribedby`.
  let {
    value = $bindable(''),
    type = 'text',
    placeholder,
    disabled = false,
    invalid = false,
    focused = false,
    mono = false,
    id,
    ariaLabel,
    ariaDescribedby,
  }: {
    value?: string;
    type?: 'text' | 'password' | 'email' | 'search' | 'tel' | 'url' | 'number';
    placeholder?: string;
    disabled?: boolean;
    invalid?: boolean;
    focused?: boolean;
    mono?: boolean;
    id?: string;
    ariaLabel?: string;
    ariaDescribedby?: string;
  } = $props();
</script>

<input
  {id}
  {type}
  {placeholder}
  {disabled}
  bind:value
  class="t-input"
  class:is-invalid={invalid}
  class:is-focus={focused}
  class:mono
  aria-label={ariaLabel}
  aria-invalid={invalid ? 'true' : undefined}
  aria-describedby={ariaDescribedby}
/>

<style>
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
  .t-input:focus,
  .t-input.is-focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
  .t-input.is-invalid {
    border-color: var(--st-err);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--st-err) 18%, transparent);
  }
  .t-input[disabled] {
    opacity: 0.5;
    pointer-events: none;
  }
  .t-input.mono {
    font-family: var(--font-mono);
  }
</style>
