<script lang="ts">
  // Structured, read-only reflection of an agent file's parsed frontmatter. The
  // Markdown tab is the editable source of truth (there is no client YAML
  // serializer, so round-tripping edits back through YAML would mangle comments,
  // ordering, and Jinja in strings); this tab shows the parsed shape at a glance.
  // Values are THIS file's own declarations - fields inherited via `extends`
  // resolve at run time and show up in the Preview (resolved prompt) tab.
  import Chip from '$lib/components/buttons/Chip.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import DocProse from './DocProse.svelte';
  import type { AgentSummary } from './agentFrontmatter';

  let {
    summary,
    body,
  }: {
    summary: AgentSummary;
    body: string;
  } = $props();

  const hasSandbox = $derived(summary.sandbox && Object.keys(summary.sandbox).length > 0);
  const sandboxFlags = $derived(
    summary.sandbox
      ? Object.entries(summary.sandbox)
          .filter(([, v]) => v === true)
          .map(([k]) => k)
      : [],
  );
  const perms = $derived(
    [
      summary.visibility ? `visibility: ${summary.visibility}` : null,
      summary.spawnable === false ? 'not spawnable' : null,
      summary.disableHistory ? 'history off' : null,
    ].filter((x): x is string => x !== null),
  );
</script>

<div class="af" aria-label="Agent frontmatter">
  {#if summary.extends && summary.extends !== 'none'}
    <div class="af-note">
      <Icon name="fork" size={11} />
      Inherits from <span class="mono">{summary.extends}</span>. Fields not declared here resolve
      from the inheritance chain at run - see the Preview tab for the fully resolved prompt.
    </div>
  {/if}

  <div class="t-setrow">
    <span class="lbl">extends</span>
    <span class="ctl">
      {#if summary.extends && summary.extends !== 'none'}
        <Chip>{#snippet icon()}<Icon name="fork" size={11} />{/snippet}{summary.extends}</Chip>
      {:else}
        <span class="spec-note">none - standalone agent</span>
      {/if}
    </span>
  </div>

  <div class="t-setrow">
    <span class="lbl">model</span>
    <span class="ctl">
      {#if summary.model}
        <Chip><span class="mono">{summary.model}</span></Chip>
      {:else}
        <span class="spec-note">inherited / daemon default</span>
      {/if}
      {#if summary.effort}<Chip><span class="mono">effort: {summary.effort}</span></Chip>{/if}
    </span>
  </div>

  <div class="t-setrow">
    <span class="lbl">max_turns</span>
    <span class="ctl">
      {#if summary.maxTurns != null}
        <span class="mono val">{summary.maxTurns}</span>
      {:else}
        <span class="spec-note">inherited / default</span>
      {/if}
    </span>
  </div>

  <div class="t-setrow">
    <span class="lbl">permissions</span>
    <span class="ctl">
      {#if perms.length}
        {#each perms as p (p)}<Chip>{p}</Chip>{/each}
      {:else}
        <span class="spec-note">defaults - public, spawnable, history on</span>
      {/if}
      {#if summary.allowedSecrets.length}
        {#each summary.allowedSecrets as sec (sec)}
          <Chip
            >{#snippet icon()}<Icon name="key" size={11} />{/snippet}<span class="mono">{sec}</span
            ></Chip
          >
        {/each}
      {/if}
    </span>
    {#if summary.allowedSecrets.length}
      <p class="hint">secret access is allowlisted to the names above</p>
    {/if}
  </div>

  {#if hasSandbox}
    <div class="t-setrow">
      <span class="lbl">sandbox</span>
      <span class="ctl">
        <span class="sb-on"><Icon name="lock" size={11} />enabled</span>
        {#each sandboxFlags as flag (flag)}
          {#if flag !== 'enabled'}<Chip><span class="mono">{flag}</span></Chip>{/if}
        {/each}
      </span>
    </div>
  {/if}

  <div class="t-field">
    <span class="flabel">tools <span class="fnote">· capabilities exposed to the agent</span></span>
    {#if summary.tools.length}
      <div class="af-tools">
        {#each summary.tools as t (t.name)}
          <span class="af-chip" class:ns={t.namespace}>
            {#if t.namespace}<Icon name="plug" size={11} />{:else}<Icon
                name="tool"
                size={11}
              />{/if}
            {t.name}
          </span>
        {/each}
      </div>
      {#if summary.tools.some((t) => t.namespace)}
        <p class="hint">
          <span class="af-chip ns mini"><Icon name="plug" size={9} /></span> = an @namespace group expands
          to every tool in that category
        </p>
      {/if}
    {:else}
      <span class="spec-note">no tools - text-only agent</span>
    {/if}
  </div>

  {#if summary.attachments.length || summary.attachmentSpecs > 0}
    <div class="t-field">
      <span class="flabel"
        >attachments <span class="fnote">· files auto-added to context</span></span
      >
      <div class="af-tools">
        {#each summary.attachments as a (a)}
          <Chip
            >{#snippet icon()}<Icon name="file" size={11} />{/snippet}<span class="mono">{a}</span
            ></Chip
          >
        {/each}
        {#if summary.attachmentSpecs > 0}
          <Chip><span class="mono">+{summary.attachmentSpecs} indexed / assigned</span></Chip>
        {/if}
      </div>
    </div>
  {/if}

  {#if summary.autoLoadSkills.length}
    <div class="t-field">
      <span class="flabel">auto_load_skills</span>
      <div class="af-tools">
        {#each summary.autoLoadSkills as sk (sk)}
          <Chip>{#snippet icon()}<Icon name="skill" size={11} />{/snippet}{sk}</Chip>
        {/each}
      </div>
    </div>
  {/if}

  {#if summary.prefetch.length}
    <div class="t-field">
      <span class="flabel"
        >prefetch <span class="fnote">· read into context before the first turn</span></span
      >
      <div class="af-tools">
        {#each summary.prefetch as p, i (i)}
          <Chip>
            <span class="mono"
              >{p.tool ?? 'tool'}{#if p.assign}
                → {p.assign}{/if}</span
            >
          </Chip>
        {/each}
      </div>
    </div>
  {/if}

  {#if summary.runIf}
    <div class="t-setrow">
      <span class="lbl">run_if</span>
      <span class="ctl"><code class="mono">{summary.runIf}</code></span>
      <p class="hint">the agent is skipped when this expression is falsy</p>
    </div>
  {/if}

  {#if summary.extraKeys.length}
    <div class="t-field">
      <span class="flabel">other frontmatter</span>
      <div class="af-tools">
        {#each summary.extraKeys as k (k)}<Chip><span class="mono">{k}</span></Chip>{/each}
      </div>
    </div>
  {/if}

  {#if summary.instructions}
    <div class="t-field">
      <span class="flabel">instructions <span class="fnote">· frontmatter system block</span></span>
      <pre class="af-block mono">{summary.instructions}</pre>
    </div>
  {/if}

  <div class="t-field">
    <span class="flabel"
      >prompt body <span class="fnote">· markdown below the frontmatter</span></span
    >
    {#if body.trim()}
      <div class="af-body"><DocProse content={body} /></div>
    {:else}
      <span class="spec-note">empty</span>
    {/if}
  </div>
</div>

<style>
  /* Layout + utilities. The .t-* primitives come from the component library;
     these are agent-form-local. */
  .af {
    padding: 12px 14px 20px;
    overflow-y: auto;
    min-height: 0;
  }
  .af-note {
    display: flex;
    gap: 7px;
    align-items: flex-start;
    font: 400 var(--fs-xs) / 1.5 var(--font-ui);
    color: var(--tx2);
    background: color-mix(in oklab, var(--acc) 8%, var(--bg1));
    border: 1px solid color-mix(in oklab, var(--acc) 26%, transparent);
    border-radius: var(--r-md);
    padding: 8px 10px;
    margin-bottom: 10px;
  }
  .af-note :global(.ic) {
    color: var(--acc);
    flex: none;
    margin-top: 1px;
  }
  .t-setrow {
    display: grid;
    grid-template-columns: 128px minmax(0, 1fr);
    gap: 3px 12px;
    align-items: center;
    padding: 8px 0;
    border-top: 1px solid var(--bd0);
  }
  .t-setrow:first-of-type {
    border-top: 0;
  }
  .t-setrow > .lbl {
    font: 600 var(--fs-xs) var(--font-ui);
    color: var(--tx1);
  }
  .t-setrow > .ctl {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    flex-wrap: wrap;
  }
  .t-setrow > .hint,
  .t-field > .hint {
    grid-column: 1 / -1;
    margin: 2px 0 0;
    font: 400 var(--fs-2xs) / 1.5 var(--font-ui);
    color: var(--tx3);
  }
  .t-field {
    display: block;
    padding: 10px 0;
    border-top: 1px solid var(--bd0);
  }
  .flabel {
    display: block;
    margin-bottom: 6px;
    font: 600 var(--fs-xs) var(--font-ui);
    color: var(--tx1);
  }
  .fnote {
    color: var(--tx3);
    font-weight: 400;
  }
  .val {
    font: 600 var(--fs-sm) var(--font-mono);
    color: var(--tx0);
  }
  .af-tools {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    min-width: 0;
  }
  .af-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border: 1px solid var(--bd1);
    border-radius: var(--r-full);
    background: var(--bg1);
    color: var(--tx2);
    font: 500 var(--fs-xs) var(--font-mono);
  }
  .af-chip.ns {
    border-color: var(--acc);
    background: color-mix(in oklab, var(--acc) 12%, var(--bg1));
    color: var(--tx0);
  }
  .af-chip :global(.ic) {
    color: var(--tx3);
    flex: none;
  }
  .af-chip.ns :global(.ic) {
    color: var(--acc);
  }
  .af-chip.mini {
    padding: 2px;
    gap: 0;
    vertical-align: middle;
  }
  .sb-on {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: var(--r-full);
    font: 600 var(--fs-xs) var(--font-mono);
    color: var(--st-ok);
    border: 1px solid color-mix(in oklab, var(--st-ok) 40%, transparent);
    background: color-mix(in oklab, var(--st-ok) 10%, transparent);
  }
  .sb-on :global(.ic) {
    color: var(--st-ok);
    flex: none;
  }
  .af-block {
    margin: 0;
    max-height: 180px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
    font: 400 var(--fs-xs) / 1.6 var(--font-mono);
    color: var(--tx1);
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 9px 11px;
  }
  .af-body {
    border-left: 2px solid var(--bd1);
    padding-left: 12px;
  }
  code.mono {
    font: 400 var(--fs-xs) var(--font-mono);
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-sm);
    padding: 1px 6px;
    color: var(--tx1);
  }
  .mono {
    font-family: var(--font-mono);
  }
  .spec-note {
    font: 400 var(--fs-xs) var(--font-ui);
    color: var(--tx3);
  }
  @media (max-width: 640px) {
    .t-setrow {
      grid-template-columns: 1fr;
    }
  }
</style>
