<script lang="ts">
  // Gallery for the `mcp` group - every state/variant side by side, static.
  // Auto-discovered by the gallery view (import.meta.glob).
  import Elicit from './Elicit.svelte';
  import AppView from './AppView.svelte';
  import GenUI from './GenUI.svelte';
  import Entity from './Entity.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  // Each gallery instance gets a unique field-name suffix: radios keyed only by
  // `name` would otherwise form ONE cross-card group (browser keeps a single
  // one checked page-wide, and ids collide). Labels are unchanged.
  function elicitFields(suffix: string) {
    return [
      {
        kind: 'enum' as const,
        name: 'environment' + suffix,
        label: 'environment',
        required: true,
        description: 'enum',
        value: 'staging',
        options: [
          { value: 'staging', label: 'staging', hint: '— auto-verifies' },
          { value: 'production', label: 'production', hint: '— needs approval' },
        ],
      },
      {
        kind: 'string' as const,
        name: 'version' + suffix,
        label: 'version',
        required: true,
        description: 'string',
        value: 'refactor/sse-backoff',
      },
      {
        kind: 'boolean' as const,
        name: 'runSmokeTests' + suffix,
        label: 'run smoke tests',
        description: 'boolean',
        value: true,
        hint: 'gate promote on the smoke suite',
      },
    ];
  }

  const elicitMsg =
    'Confirm the deploy parameters. Fields come from the tool’s requestedSchema; the host renders them and returns { action, content }.';

  const genuiChoices = [
    'Roll forward now',
    'Wait for the job to finish',
    'Hold — decide after review',
  ];

  const entityStatuses = [
    { status: 'working' as const, statusLabel: 'In Progress' },
    { status: 'blocked' as const, statusLabel: 'Blocked' },
    { status: 'done' as const, statusLabel: 'Done' },
    { status: 'err' as const, statusLabel: 'Failed' },
  ];
</script>

<section data-testid="gallery-mcp">
  <!-- ===== MCP elicitation ===== -->
  <p class="lbl">Elicit — open · submitted · declined · cancelled</p>
  <div class="row-wrap">
    <div class="cell">
      <span class="tag">open</span>
      <Elicit
        source="deploy-server"
        message={elicitMsg}
        fields={elicitFields('-open')}
        state="open"
      />
    </div>
    <div class="cell">
      <span class="tag">submitted</span>
      <Elicit
        source="deploy-server"
        message={elicitMsg}
        fields={elicitFields('-sub')}
        state="submitted"
      />
    </div>
    <div class="cell">
      <span class="tag">declined</span>
      <Elicit
        source="deploy-server"
        message={elicitMsg}
        fields={elicitFields('-dec')}
        state="declined"
      />
    </div>
    <div class="cell">
      <span class="tag">cancelled</span>
      <Elicit
        source="deploy-server"
        message={elicitMsg}
        fields={elicitFields('-can')}
        state="cancelled"
      />
    </div>
  </div>

  <!-- ===== MCP App view ===== -->
  <p class="lbl">AppView — inline · fullscreen · pip · border-off · init lifecycle</p>
  <div class="row-wrap">
    <div class="cell">
      <span class="tag">inline · ready · bordered</span>
      <AppView name="Deploy Monitor" source="deploy-server" iconChar="D" iconColor="#1F8A5B">
        {@render deployBody()}
      </AppView>
    </div>
    <div class="cell cell--full">
      <span class="tag">fullscreen mode</span>
      <AppView
        name="Deploy Monitor"
        source="deploy-server"
        iconChar="D"
        iconColor="#1F8A5B"
        mode="fullscreen"
      >
        {@render deployBody()}
      </AppView>
    </div>
    <div class="cell">
      <span class="tag">pip mode</span>
      <AppView
        name="Deploy Monitor"
        source="deploy-server"
        iconChar="D"
        iconColor="#1F8A5B"
        mode="pip"
      >
        {@render deployBody()}
      </AppView>
    </div>
    <div class="cell">
      <span class="tag">border off</span>
      <AppView
        name="Deploy Monitor"
        source="deploy-server"
        iconChar="D"
        iconColor="#1F8A5B"
        border={false}
      >
        {@render deployBody()}
      </AppView>
    </div>
    <div class="cell">
      <span class="tag">init (handshake)</span>
      <AppView
        name="Deploy Monitor"
        source="deploy-server"
        iconChar="D"
        iconColor="#1F8A5B"
        life="init"
      >
        {@render deployBody()}
      </AppView>
    </div>
  </div>

  <!-- ===== Agent-generated inline UI ===== -->
  <p class="lbl">GenUI — open · resolved</p>
  <div class="row-wrap">
    <div class="cell">
      <span class="tag">open</span>
      <GenUI
        question="The retention job is still running. How do you want to sequence the deploy?"
        choices={genuiChoices}
      />
    </div>
    <div class="cell">
      <span class="tag">resolved (picked #2)</span>
      <GenUI
        question="The retention job is still running. How do you want to sequence the deploy?"
        choices={genuiChoices}
        selected={1}
      />
    </div>
  </div>

  <!-- ===== Entity auto-link + inline panel ===== -->
  <p class="lbl">Entity — chiplet status variants (dot + text)</p>
  <p class="prose">
    Tracked in{' '}
    {#each entityStatuses as e (e.status)}
      <Entity
        entityKey="JIRA-1234"
        status={e.status}
        statusLabel={e.statusLabel}
        title="SSE client never recovers after laptop sleep"
        assignee="you"
        priority="High"
        sprint="Sprint 24"
        via="Jira"
      />{' '}
    {/each}
  </p>

  <p class="lbl">Entity — detail popover (open)</p>
  <div class="pop-space">
    <Entity
      entityKey="JIRA-1234"
      status="working"
      statusLabel="In Progress"
      title="SSE client never recovers after laptop sleep"
      assignee="you"
      priority="High"
      sprint="Sprint 24"
      via="Jira"
      open
    />
  </div>

  <p class="lbl">Inline panel — plugin-contributed (composition of primitives)</p>
  <!-- plug-chatpanel is a static gallery composition: its .t-pill/.t-chip carry
       data-st vocabularies (running/done/pending) the shared Pill/Chip don't
       model, so the whole block stays inline as scaffolding. Icons use <Icon>. -->
  <div class="plug-chatpanel">
    <span class="via"
      ><span class="dot dot--jira"></span>rendered by Jira plugin · inline panel</span
    >
    <div class="row row--panel">
      <span class="av-ico av-ico--jira">J</span>
      <div class="panel-main">
        <div class="row row--head">
          <b class="mono panel-key">JIRA-1234</b>
          <span class="t-pill" data-st="running"
            ><Icon name="play" /><span class="ptxt">In Progress</span></span
          >
        </div>
        <div class="panel-title">SSE client never recovers after laptop sleep</div>
      </div>
    </div>
    <dl class="t-kv">
      <dt>assignee</dt>
      <dd>you</dd>
      <dt>priority</dt>
      <dd>High</dd>
      <dt>sprint</dt>
      <dd>Sprint 24</dd>
    </dl>
    <div class="row row--actions">
      <button type="button" class="t-btn t-btn--sm">Open in Jira</button>
      <button type="button" class="t-btn t-btn--sm t-btn--ghost">Move to Review</button>
    </div>
  </div>
</section>

{#snippet deployBody()}
  <div class="row row--split">
    <b class="avb-title">refactor/sse-backoff → staging</b>
    <span class="t-pill" data-st="running"
      ><Icon name="play" /><span class="ptxt">building</span></span
    >
  </div>
  <div
    class="t-prog t-prog--slim"
    role="meter"
    aria-valuenow="62"
    aria-valuemin="0"
    aria-valuemax="100"
    aria-label="Build 62%"
  >
    <i class="prog-62"></i>
  </div>
  <div class="row row--chips">
    <span class="t-chip" data-st="done"><Icon name="check" />build</span>
    <span class="t-chip" data-st="done"><Icon name="check" />test</span>
    <span class="t-chip" data-st="running"
      ><span class="t-spin" aria-hidden="true">⠸</span>push</span
    >
    <span class="t-chip" data-st="pending">promote</span>
  </div>
{/snippet}

<style>
  section {
    display: grid;
    gap: var(--sp-3);
  }
  .lbl {
    margin: var(--sp-4) 0 0;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .row-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-4);
    align-items: flex-start;
  }
  .cell {
    display: grid;
    gap: var(--sp-1);
    min-width: 0;
  }
  /* The fullscreen AppView demo needs a definite full-row host: its
     max-width:none only reads as "fills the panel" when the surrounding cell
     spans the row (mirrors the ref's definite-width .spec grid track). */
  .cell--full {
    flex-basis: 100%;
  }
  .tag {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .prose {
    font-size: var(--fs-md);
    line-height: 2;
    color: var(--tx1);
    max-width: 76ch;
  }
  .pop-space {
    padding-bottom: 180px;
  }

  /* Static app-body + inline-panel compositions: the t-pill/t-chip badges here
     carry data-st vocabularies (running/done/pending) the real Pill/Chip don't
     model, so they stay inline as gallery scaffolding. Icons use shared <Icon>. */
  .mono {
    font-family: var(--font-mono);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }
  .avb-title {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
  }
  .t-spin {
    font-family: var(--font-mono);
    font-weight: 600;
    display: inline-block;
    width: 1.1ch;
    line-height: 1;
    flex: none;
  }
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
  .t-pill[data-st='running'] {
    --c: var(--st-ok);
  }
  .t-prog {
    height: 3px;
    background: var(--bg3);
    border-radius: var(--r-full);
    overflow: hidden;
    width: 100%;
  }
  .t-prog i {
    display: block;
    height: 100%;
    background: var(--acc);
    border-radius: inherit;
  }
  .t-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
  }
  .t-chip :global(.ic) {
    width: 10px;
    height: 10px;
    color: var(--tx3);
  }
  .av-ico {
    width: 26px;
    height: 26px;
    border-radius: 6px;
    flex: none;
    display: grid;
    place-items: center;
    color: var(--on-brand);
    font: 700 11px system-ui;
  }
  .panel-key {
    font-size: var(--fs-sm);
    color: var(--tx0);
  }
  .panel-title {
    font: 500 var(--fs-md) / 1.4 var(--font-ui);
    color: var(--tx1);
  }
  .plug-chatpanel {
    position: relative;
    border: 1px dashed color-mix(in oklab, var(--acc) 38%, var(--bd1));
    border-radius: var(--r-lg);
    padding: 14px 11px 11px;
    display: grid;
    gap: 10px;
    max-width: 520px;
  }
  .plug-chatpanel > .via {
    position: absolute;
    top: -8px;
    left: 12px;
    background: var(--bg0);
    padding: 0 7px;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    display: inline-flex;
    gap: 5px;
    align-items: center;
  }
  .plug-chatpanel > .via .dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    flex: none;
  }
  .t-kv {
    margin: 0;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 5px 14px;
    font-size: var(--fs-sm);
    align-items: baseline;
  }
  .t-kv dt {
    color: var(--tx3);
    font: 500 var(--fs-2xs) / 1.7 var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .t-kv dd {
    margin: 0;
    color: var(--tx1);
    text-align: right;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .t-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    height: 28px;
    padding: 0 11px;
    border-radius: var(--r-md);
    border: 1px solid var(--bd1);
    background: var(--bg3);
    color: var(--tx0);
    font: 500 var(--fs-md) / 1 var(--font-ui);
    cursor: pointer;
    white-space: nowrap;
  }
  .t-btn:hover {
    background: var(--bg4);
    border-color: color-mix(in oklab, var(--bd1) 60%, var(--tx3));
  }
  .t-btn--sm {
    height: 23px;
    padding: 0 8px;
    font-size: var(--fs-sm);
    gap: 5px;
  }
  .t-btn--ghost {
    background: transparent;
    border-color: transparent;
    color: var(--tx1);
  }
  .t-btn--ghost:hover {
    background: var(--bg3);
    color: var(--tx0);
    border-color: transparent;
  }

  /* demo scaffolding (was inline styles): jira-brand accents + panel layout */
  .dot--jira,
  .av-ico--jira {
    background: #0052cc;
  }
  .row--panel {
    gap: 9px;
    align-items: flex-start;
  }
  .panel-main {
    min-width: 0;
    flex: 1;
    display: grid;
    gap: 3px;
  }
  .row--head {
    gap: 8px;
    flex-wrap: wrap;
  }
  .row--actions {
    gap: 6px;
  }
  .row--split {
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 6px;
  }
  .row--chips {
    gap: 6px;
    flex-wrap: wrap;
  }
  .t-prog--slim {
    height: 5px;
    margin: 9px 0;
  }
  .t-prog--slim .prog-62 {
    width: 62%;
  }
</style>
