<script lang="ts">
  // MCP elicitation - server/plugin requests structured input.
  // Maps to `elicitation/create`: { message, requestedSchema } in →
  // { action: 'accept' | 'decline' | 'cancel', content } out. The three-button
  // contract is fixed; fields are schema-driven (string · number · boolean · enum).
  // Host-rendered - never sent to the model. Presentational + callback props.
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';

  type ElicitState = 'open' | 'submitted' | 'declined' | 'cancelled';

  type EnumOption = { value: string; label: string; hint?: string };
  type ElicitField =
    | {
        kind: 'string' | 'number';
        name: string;
        label: string;
        value?: string;
        required?: boolean;
        description?: string;
      }
    | {
        kind: 'boolean';
        name: string;
        label: string;
        value?: boolean;
        description?: string;
        hint?: string;
      }
    | {
        kind: 'enum';
        name: string;
        label: string;
        options: EnumOption[];
        value?: string;
        required?: boolean;
        description?: string;
      };

  let {
    source,
    blocking = true,
    message,
    fields = [],
    state: viewState = 'open',
    pillLabel = 'elicitation',
    onSubmit,
    onDecline,
    onCancel,
  }: {
    source: string;
    blocking?: boolean;
    message: string;
    fields?: ElicitField[];
    state?: ElicitState;
    pillLabel?: string;
    onSubmit?: (content: Record<string, unknown>) => void;
    onDecline?: () => void;
    onCancel?: () => void;
  } = $props();

  // Live form values, seeded once from the schema defaults. Booleans stay
  // boolean; string/number are edited as text and coerced on submit.
  const seed: Record<string, string | boolean> = {};
  // svelte-ignore state_referenced_locally -- seed captures the schema defaults once.
  for (const f of fields) {
    if (f.kind === 'boolean') seed[f.name] = f.value ?? false;
    else seed[f.name] = f.value ?? '';
  }
  const values = $state(seed);

  function coerce(f: ElicitField): unknown {
    const v = values[f.name];
    if (f.kind === 'boolean') return Boolean(v);
    if (f.kind === 'number') return v === '' || v == null ? undefined : Number(v);
    return v ?? '';
  }

  function submit() {
    const content: Record<string, unknown> = {};
    for (const f of fields) content[f.name] = coerce(f);
    onSubmit?.(content);
  }

  function toggle(name: string) {
    values[name] = !values[name];
  }
</script>

<div
  class="t-elicit"
  data-state={viewState}
  role="group"
  aria-label={`Input requested by ${source}`}
>
  <div class="hd">
    <Icon name="sparkle" />Input requested<span class="src"
      >· {source} · {blocking ? 'blocking' : 'non-blocking'}</span
    ><span class="t-pill" data-st="awaiting"
      ><Icon name="q" /><span class="ptxt">{pillLabel}</span></span
    >
  </div>
  <div class="msg">{message}</div>

  <div class="el-form">
    {#each fields as field (field.name)}
      <div class="el-f">
        {#if field.kind === 'enum'}
          <div class="cap" id={`el-lbl-${field.name}`}>
            {field.label}{#if field.required}<span class="req">*</span
              >{/if}{#if field.description}<span class="desc">{field.description}</span>{/if}
          </div>
          <div class="el-radio" role="radiogroup" aria-labelledby={`el-lbl-${field.name}`}>
            {#each field.options as opt (opt.value)}
              <label class="el-opt">
                <input
                  type="radio"
                  name={field.name}
                  value={opt.value}
                  checked={values[field.name] === opt.value}
                  onchange={() => (values[field.name] = opt.value)}
                />
                <span><span class="oh">{opt.label}</span> <span class="od">{opt.hint}</span></span>
              </label>
            {/each}
          </div>
        {:else if field.kind === 'boolean'}
          <div class="cap" id={`el-lbl-${field.name}`}>
            {field.label}{#if field.description}<span class="desc">{field.description}</span>{/if}
          </div>
          <div class="row row--switch">
            <button
              type="button"
              class="t-sw"
              role="switch"
              aria-checked={values[field.name] ? 'true' : 'false'}
              aria-labelledby={`el-lbl-${field.name}`}
              onclick={() => toggle(field.name)}
            ></button>
            {#if field.hint}<span class="hint">{field.hint}</span>{/if}
          </div>
        {:else}
          <label for={`el-${field.name}`}
            >{field.label}{#if field.required}<span class="req">*</span
              >{/if}{#if field.description}<span class="desc">{field.description}</span>{/if}</label
          >
          <input
            id={`el-${field.name}`}
            class="t-input mono"
            type={field.kind === 'number' ? 'number' : 'text'}
            value={String(values[field.name] ?? '')}
            oninput={(e) => (values[field.name] = e.currentTarget.value)}
          />
        {/if}
      </div>
    {/each}
  </div>

  <div class="fx">
    <Button variant="pri" size="sm" onclick={submit}>
      {#snippet icon()}<Icon name="check" />{/snippet}Submit
    </Button>
    <Button size="sm" onclick={() => onDecline?.()}>Decline</Button>
    <Button variant="ghost" size="sm" onclick={() => onCancel?.()}>Cancel</Button>
    <span class="schema-note"
      ><Icon name="lock" class="sn" />host-rendered · never sent to the model</span
    >
  </div>

  <div class="res r-ok">
    <Icon name="check" /><span class="res-txt">submitted · action: accept · content returned</span>
  </div>
  <div class="res r-no">
    <Icon name="x" /><span class="res-txt">declined · action: decline</span>
  </div>
  <div class="res r-cancel">
    <Icon name="x" /><span class="res-txt">cancelled · dismissed with no decision</span>
  </div>
</div>

<style>
  /* .t-btn deduped → buttons/Button.svelte. The .t-pill (data-st="awaiting" is
     the elicitation vocabulary, not a shared Pill PillState), .t-input (bound to
     the shared `values` object), and .t-sw (custom switch bound to `values` via
     aria-labelledby) stay inline — none fit their shared component's API. */
  .mono {
    font-family: var(--font-mono);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }
  .row--switch {
    gap: 8px;
  }
  .hint {
    font-size: var(--fs-xs);
    color: var(--tx3);
  }

  /* status pill */
  .t-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 8px 0 7px;
    border-radius: var(--r-full);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    letter-spacing: 0.02em;
    white-space: nowrap;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 13%, transparent);
    border: 1px solid color-mix(in oklab, var(--c) 32%, transparent);
  }
  .t-pill :global(.ic) {
    width: 11px;
    height: 11px;
  }
  .t-pill[data-st='awaiting'] {
    --c: var(--st-warn);
  }

  /* input */
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
  .t-input.mono {
    font-family: var(--font-mono);
  }
  .t-input::placeholder {
    color: var(--tx3);
  }
  .t-input:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }

  /* switch */
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

  /* ===== MCP elicitation (owned here) ===== */
  .t-elicit {
    border: 1px solid color-mix(in oklab, var(--st-info) 45%, var(--bd1));
    background: color-mix(in oklab, var(--st-info) 6%, var(--bg1));
    border-radius: var(--r-lg);
    padding: 12px 13px;
    display: grid;
    gap: 10px;
    max-width: 560px;
  }
  .t-elicit .hd {
    display: flex;
    align-items: center;
    gap: 7px;
    font: 600 var(--fs-sm) / 1.2 var(--font-ui);
    color: var(--st-info);
    flex-wrap: wrap;
  }
  .t-elicit .hd :global(.ic) {
    width: 14px;
    height: 14px;
  }
  .t-elicit .hd .src {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    font-weight: 400;
  }
  .t-elicit .hd .t-pill {
    margin-left: auto;
  }
  .t-elicit .msg {
    font-size: var(--fs-sm);
    color: var(--tx1);
    line-height: 1.5;
    text-wrap: pretty;
  }
  .t-elicit .el-form {
    display: grid;
    gap: 11px;
  }
  .t-elicit .el-f {
    display: grid;
    gap: 4px;
  }
  .t-elicit .el-f > label,
  .t-elicit .el-f > .cap {
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--tx2);
    display: flex;
    gap: 5px;
    align-items: center;
  }
  .t-elicit .el-f .req {
    color: var(--st-err);
    font-weight: 700;
  }
  .t-elicit .el-f .desc {
    font: 400 var(--fs-2xs) / 1.5 var(--font-ui);
    color: var(--tx3);
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
  }
  .t-elicit .el-radio {
    display: grid;
    gap: 4px;
  }
  .t-elicit .el-opt {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 6px 9px;
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    background: var(--bg1);
    cursor: pointer;
    font-size: var(--fs-sm);
    color: var(--tx1);
  }
  .t-elicit .el-opt:hover {
    border-color: var(--st-info);
  }
  .t-elicit .el-opt input {
    accent-color: var(--st-info);
    margin: 2px 0 0;
  }
  .t-elicit .el-opt .oh {
    font-weight: 600;
    color: var(--tx0);
  }
  .t-elicit .el-opt .od {
    font: 400 var(--fs-2xs) var(--font-ui);
    color: var(--tx3);
  }
  .t-elicit .fx {
    display: flex;
    gap: 7px;
    align-items: center;
    flex-wrap: wrap;
    border-top: 1px solid var(--bd0);
    padding-top: 10px;
  }
  .t-elicit .schema-note {
    margin-left: auto;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    display: inline-flex;
    gap: 5px;
    align-items: center;
  }
  .t-elicit .schema-note :global(.sn) {
    width: 10px;
    height: 10px;
  }
  .t-elicit .res {
    display: none;
    align-items: center;
    gap: 7px;
    font: 500 var(--fs-sm) var(--font-mono);
  }
  .t-elicit[data-state='submitted'] {
    border-color: color-mix(in oklab, var(--st-ok) 40%, transparent);
    background: color-mix(in oklab, var(--st-ok) 6%, transparent);
  }
  .t-elicit[data-state='submitted'] .hd {
    color: var(--st-ok);
  }
  .t-elicit[data-state='declined'],
  .t-elicit[data-state='cancelled'] {
    border-color: var(--bd1);
    background: var(--bg1);
  }
  .t-elicit[data-state='declined'] .hd,
  .t-elicit[data-state='cancelled'] .hd {
    color: var(--tx3);
  }
  .t-elicit:not([data-state='open']) :is(.el-form, .fx) {
    display: none;
  }
  .t-elicit[data-state='submitted'] .res.r-ok,
  .t-elicit[data-state='declined'] .res.r-no,
  .t-elicit[data-state='cancelled'] .res.r-cancel {
    display: flex;
  }
  .t-elicit .res.r-ok {
    color: var(--st-ok);
  }
  .t-elicit .res.r-no,
  .t-elicit .res.r-cancel {
    color: var(--tx3);
  }
</style>
