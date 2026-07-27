<script lang="ts">
  import Palette from './Palette.svelte';
  import type { PaletteItem } from './palette-match';

  const items: PaletteItem[] = [
    {
      group: 'sessions',
      icon: 'chat',
      label: 'refactor: sse reconnect backoff',
      meta: 'code · now',
    },
    {
      group: 'sessions',
      icon: 'chat',
      label: 'ops: nightly backup failing on prune',
      meta: 'ops · 12m',
    },
    { group: 'sessions', icon: 'chat', label: 'research: local whisper models', meta: 'res · 2h' },
    { group: 'sessions', icon: 'chat', label: 'migrate schedules to cron v2', meta: 'code · 38m' },
    { group: 'jobs', icon: 'jobs', label: 'nightly backup prune policy', meta: 'awaiting input' },
    { group: 'jobs', icon: 'jobs', label: 'fix flaky sse reconnect test', meta: 'running' },
    { group: 'jobs', icon: 'jobs', label: 'update caddy to 2.10', meta: 'errored' },
    {
      group: 'terminals',
      icon: 'term',
      label: 'npm test -w @tsugite/sse --watch',
      meta: 'running',
    },
    { group: 'terminals', icon: 'term', label: 'tail -f ~/.tsugite/daemon.log', meta: 'running' },
    { group: 'schedules', icon: 'sched', label: 'nightly-backup', meta: 'errored · daily 03:00' },
    { group: 'schedules', icon: 'sched', label: 'inbox-triage', meta: 'every 15 min' },
    { group: 'agents', icon: 'agent', label: 'agents/ops-runner.md', meta: 'agent' },
    { group: 'agents', icon: 'agent', label: 'agents/code-worker.md', meta: 'agent' },
    { group: 'skills', icon: 'skill', label: 'pdf-extract', meta: '2 warnings' },
    { group: 'files', icon: 'files', label: 'src/lib/sse.ts', meta: 'workspace' },
    { group: 'actions', icon: 'plus', label: 'New session', meta: 'action', action: true },
    { group: 'actions', icon: 'plus', label: 'New job', meta: 'action', action: true },
    { group: 'actions', icon: 'term', label: 'Run command…', meta: 'action', action: true },
    {
      group: 'actions',
      icon: 'compress',
      label: 'Compact current session',
      meta: 'action',
      action: true,
    },
  ];

  const states: { label: string; query: string }[] = [
    { label: 'default · grouped (no query)', query: '' },
    { label: 'query “prune” · cross-surface + highlight', query: 'prune' },
    { label: 'query “npm” · pty subsequence', query: 'npm' },
    { label: 'query “new” · quick actions', query: 'new' },
    { label: 'no matches · empty hint', query: 'zzzzz' },
  ];
</script>

<section data-testid="gallery-palette">
  <div class="wrap">
    {#each states as s (s.label)}
      <div class="cell">
        <span class="lab">{s.label}</span>
        <Palette inline {items} initialQuery={s.query} />
      </div>
    {/each}
    <div class="cell note">
      <span class="lab">overlay + mobile sheet</span>
      <p>
        With <code>inline=&#123;false&#125;</code> the panel renders as a centered modal over a
        dimmed backdrop; under 640px it fills the screen as a full-screen sheet. Bind
        <code>open</code>
        (⌘K from the app shell) and pass <code>onSelect</code> to wire jumps and quick actions.
      </p>
    </div>
  </div>
</section>

<style>
  .wrap {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-5);
    align-items: flex-start;
  }
  .cell {
    display: grid;
    gap: var(--sp-2);
  }
  .lab {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .note {
    max-width: 320px;
  }
  .note p {
    margin: 0;
    font-size: var(--fs-sm);
    line-height: 1.6;
    color: var(--tx2);
  }
  .note code {
    font: 500 var(--fs-xs) var(--font-mono);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    padding: 0 4px;
    border-radius: 4px;
    color: var(--tx1);
  }
</style>
