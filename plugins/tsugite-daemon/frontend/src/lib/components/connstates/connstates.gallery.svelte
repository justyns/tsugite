<script lang="ts">
  // Static demo of every connstates state/variant, side by side. No
  // interaction - each variant is its own pre-configured instance.
  import Conn from './Conn.svelte';
  import StaleStamp from './StaleStamp.svelte';
  import PaneState from './PaneState.svelte';
  import LockCard from './LockCard.svelte';
  import SecTable from './SecTable.svelte';
  import type { SecretRow } from './SecTable.svelte';
  import SetSecretModal from './SetSecretModal.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  const noop = () => {};

  const secretRows: SecretRow[] = [
    {
      name: 'OPENAI_API_KEY',
      provenance: 'process env',
      scope: 'env',
    },
    {
      name: 'TAILSCALE_AUTHKEY',
      provenance: 'process env',
      scope: 'env',
    },
  ];

  const staleSince = Date.now() - 6_000;
</script>

<section data-testid="gallery-connstates">
  <h3>Conn</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">on</p>
      <div class="frame pad"><Conn state="on" /></div>
    </div>
    <div class="demo">
      <p class="lbl">re · with attempt count</p>
      <div class="frame pad"><Conn state="re" reconnectAttempt={2} /></div>
    </div>
    <div class="demo">
      <p class="lbl">off · with retry now</p>
      <div class="frame pad"><Conn state="off" onRetry={noop} /></div>
    </div>
  </div>

  <h3>StaleStamp</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">no timestamp</p>
      <div class="frame pad"><StaleStamp /></div>
    </div>
    <div class="demo">
      <p class="lbl">ticking · since 6s ago</p>
      <div class="frame pad"><StaleStamp since={staleSince} /></div>
    </div>
  </div>

  <h3>PaneState</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">empty</p>
      <div class="frame pad">
        <PaneState kind="empty" title="No jobs yet">
          {#snippet icon()}<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="4.5" /></svg>{/snippet}
          {#snippet hint()}Spawn one from any conversation with <span class="mono">/job</span
            >.{/snippet}
          {#snippet actions()}<Button size="sm">New job</Button>{/snippet}
        </PaneState>
      </div>
    </div>
    <div class="demo">
      <p class="lbl">loading</p>
      <div class="frame pad"><PaneState kind="loading" /></div>
    </div>
    <div class="demo">
      <p class="lbl">error</p>
      <div class="frame pad">
        <PaneState kind="error" title="Couldn't load run history">
          {#snippet icon()}<svg viewBox="0 0 16 16">
              <path d="M8 2.6L14.2 13H1.8z" /><path d="M8 6.6v2.8" /><circle
                cx="8"
                cy="11.4"
                r=".4"
                fill="currentColor"
                stroke="none"
              />
            </svg>{/snippet}
          {#snippet hint()}<span class="mono">GET /api/schedules/runs → 500</span>{/snippet}
          {#snippet actions()}
            <Button size="sm">
              {#snippet icon()}<Icon name="retry" />{/snippet}Retry
            </Button>
            <Button size="sm" variant="ghost">View daemon log</Button>
          {/snippet}
        </PaneState>
      </div>
    </div>
    <div class="demo">
      <p class="lbl">permission</p>
      <div class="frame pad">
        <PaneState kind="permission" title="Read-only mount">
          {#snippet icon()}<svg viewBox="0 0 16 16"
              ><rect x="4" y="7" width="8" height="6.5" rx="1" /><path
                d="M5.7 7V5.2a2.3 2.3 0 0 1 4.6 0V7"
              /></svg
            >{/snippet}
          {#snippet hint()}<span class="mono">/workspace/vendor</span> is mounted
            <span class="mono">ro</span> — editing disabled.{/snippet}
        </PaneState>
      </div>
    </div>
  </div>

  <h3>LockCard</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">locked</p>
      <div class="frame pad"><LockCard onUnlock={noop} /></div>
    </div>
  </div>

  <h3>SecTable</h3>
  <div class="demo-row">
    <div class="demo wide">
      <p class="lbl">rows</p>
      <div class="frame pad">
        <SecTable rows={secretRows} onRotate={noop} onDelete={noop} />
      </div>
    </div>
  </div>

  <h3>SetSecretModal</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">add</p>
      <div class="frame pad">
        <SetSecretModal mode="add" onCancel={noop} onSave={noop} />
      </div>
    </div>
    <div class="demo">
      <p class="lbl">rotate</p>
      <div class="frame pad">
        <SetSecretModal mode="rotate" name="OPENAI_API_KEY" onCancel={noop} onSave={noop} />
      </div>
    </div>
  </div>
</section>

<style>
  section {
    display: grid;
    gap: var(--sp-3);
  }
  h3 {
    margin: var(--sp-3) 0 0;
    font: 600 var(--fs-xs)/1 var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  h3:first-child {
    margin-top: 0;
  }
  .demo-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-4);
    align-items: flex-start;
  }
  .demo {
    display: grid;
    gap: 6px;
    min-width: 220px;
  }
  .demo.wide {
    min-width: min(600px, 100%);
    flex: 1;
  }
  .lbl {
    margin: 0;
    font: 400 var(--fs-2xs) var(--font-ui);
    color: var(--tx3);
  }
  .frame {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow: hidden;
  }
  .frame.pad {
    padding: var(--sp-3);
    background: var(--bg1);
  }

  /* The .mono spans in the snippets fed into PaneState above render outside this
     file's DOM (inside PaneState), so they carry no scope hash a scoped rule
     could match — hence :global. The buttons/icons in those snippets now use
     <Button>/<Icon>, which bring their own styles, so only .mono stays here. */
  :global(.mono) {
    font-family: var(--font-mono);
  }
</style>
