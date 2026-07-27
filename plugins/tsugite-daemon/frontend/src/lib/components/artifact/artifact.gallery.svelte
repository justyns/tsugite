<script lang="ts">
  import ArtifactPanel from './ArtifactPanel.svelte';
  import AnnPopover from './AnnPopover.svelte';
  import AnnThread from './AnnThread.svelte';
  import VerdictBar from './VerdictBar.svelte';
  import WikiLink from './WikiLink.svelte';
  import Backlinks from './Backlinks.svelte';
  import TagPill from './TagPill.svelte';

  let panelView = $state('rendered');
</script>

{#snippet docBody()}
  <div class="doc-md">
    <h2>Problem</h2>
    <p>
      Disk sits at <strong>84%</strong>, over target. The
      <span class="ann-hl">weeklies older than 90 days total 41&nbsp;GB</span> aren't covered by any prune
      rule.
    </p>
    <h2>Proposal</h2>
    <p>
      Add <code>weekly_max_age: 90d</code> and prune older weeklies nightly, keeping all 14 dailies.
    </p>
  </div>
{/snippet}

{#snippet diffBody()}
  <div class="doc-md">
    <pre><span class="diff-del">- keep_weekly: all</span>
<span class="diff-add">+ weekly_max_age: 90d</span></pre>
  </div>
{/snippet}

{#snippet rawBody()}
  <pre># Backup retention plan

Disk sits at 84%, over target...</pre>
{/snippet}

{#snippet jsonBody()}
  <pre>{`{ "kind": "plan", "sections": 6, "open_annotations": 2 }`}</pre>
{/snippet}

{#snippet editBody()}
  <textarea class="doc-edit" aria-label="Edit ops/runbook.md" spellcheck="false"
    ># Runbook

See [[sse-reconnect]] for the backoff table.</textarea
  >
{/snippet}

<section data-testid="gallery-artifact">
  <div class="ga">
    <!-- ============ ArtifactPanel ============ -->
    <div class="cell wide">
      <span class="lbl">ArtifactPanel · launch card (in chat)</span>
      <ArtifactPanel
        variant="launch"
        title="Backup retention plan"
        subtitle="plan · markdown · 6 sections · 2 open annotations"
        openLabel="Review"
      />
    </div>

    <div class="cell wide">
      <span class="lbl">ArtifactPanel · panel (mode switch + popover + verdict)</span>
      <div class="panelbox">
        <ArtifactPanel
          title="Backup retention plan"
          kind="plan"
          count="2 annotations · 2 open"
          bind:view={panelView}
          rendered={docBody}
          diff={diffBody}
          raw={rawBody}
          json={jsonBody}
        >
          {#snippet overlay()}
            <AnnPopover variant="art" open x={44} y={150} />
          {/snippet}
          {#snippet footer()}
            <VerdictBar state="pending" note="2 open annotations" />
          {/snippet}
        </ArtifactPanel>
      </div>
    </div>

    <div class="cell wide">
      <span class="lbl">ArtifactPanel · diff view</span>
      <div class="panelbox">
        <ArtifactPanel
          title="retention.yml"
          kind="diff"
          view="diff"
          views={['rendered', 'diff', 'raw', 'json']}
          rendered={docBody}
          diff={diffBody}
          raw={rawBody}
          json={jsonBody}
        />
      </div>
    </div>

    <div class="cell wide">
      <span class="lbl">ArtifactPanel · doc (rendered / raw / edit toggle)</span>
      <div class="panelbox">
        <ArtifactPanel
          title="ops/runbook.md"
          kind="doc"
          views={['rendered', 'raw', 'edit']}
          view="edit"
          rendered={docBody}
          raw={rawBody}
          edit={editBody}
        />
      </div>
    </div>

    <!-- ============ AnnPopover ============ -->
    <div class="cell">
      <span class="lbl">AnnPopover · menu skin</span>
      <div class="popbox"><AnnPopover open isStatic /></div>
    </div>
    <div class="cell">
      <span class="lbl">AnnPopover · art skin</span>
      <div class="popbox"><AnnPopover variant="art" open isStatic /></div>
    </div>

    <!-- ============ AnnThread ============ -->
    <div class="cell">
      <span class="lbl">AnnThread · open</span>
      <AnnThread
        author="you"
        anchor="cap total at 40 GB"
        when="2m"
        body="40 GB cap conflicts with the 90d weekly policy — pick one, don't stack both."
        status="open"
      />
    </div>
    <div class="cell">
      <span class="lbl">AnnThread · editing</span>
      <AnnThread
        author="you"
        anchor="cap total at 40 GB"
        when="now"
        body="40 GB cap conflicts with the 90d weekly policy."
        status="editing"
      />
    </div>
    <div class="cell">
      <span class="lbl">AnnThread · resolved</span>
      <AnnThread
        author="ada"
        anchor="prune nightly"
        when="1h"
        body="Confirmed — nightly prune keeps 14 dailies."
        status="resolved"
      />
    </div>

    <!-- ============ VerdictBar ============ -->
    <div class="cell wide">
      <span class="lbl">VerdictBar · pending / approved / changes</span>
      <div class="stack">
        <div class="ftbox"><VerdictBar state="pending" note="2 open annotations" /></div>
        <div class="ftbox"><VerdictBar state="approved" /></div>
        <div class="ftbox"><VerdictBar state="changes" /></div>
      </div>
    </div>

    <!-- ============ WikiLink ============ -->
    <div class="cell wide">
      <span class="lbl">WikiLink · resolved / missing</span>
      <p class="prose">
        Retention interacts with <WikiLink page="backup-retention" href="#" /> and the missing
        <WikiLink page="household-systems" missing href="#" /> page.
      </p>
    </div>

    <!-- ============ TagPill ============ -->
    <div class="cell wide">
      <span class="lbl">TagPill · link / removable</span>
      <div class="row">
        <TagPill tag="ops" href="#" />
        <TagPill tag="sse" href="#" />
        <TagPill tag="runbook" href="#" />
        <TagPill tag="draft" removable />
      </div>
    </div>

    <!-- ============ Backlinks ============ -->
    <div class="cell">
      <span class="lbl">Backlinks · with links</span>
      <Backlinks
        links={[
          { file: 'ops/runbook.md', snippet: '“…see sse-reconnect for the backoff table…”' },
          { file: 'notes/disk.md', snippet: '“…retention plan capped weeklies at 90d…”' },
        ]}
      />
    </div>
    <div class="cell">
      <span class="lbl">Backlinks · empty</span>
      <Backlinks links={[]} />
    </div>
  </div>
</section>

<style>
  .ga {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-4);
    align-items: flex-start;
  }
  .cell {
    display: grid;
    gap: 6px;
    align-content: start;
    min-width: 240px;
    max-width: 300px;
    flex: 1 1 240px;
  }
  .cell.wide {
    min-width: 320px;
    max-width: 560px;
    flex-basis: 340px;
  }
  .lbl {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .panelbox {
    height: 260px;
    display: flex;
    /* grid item in .cell: without this it pins to the json line's min-content
       and overflows the cell. .art-panel already carries min-width:0 downstream. */
    min-width: 0;
  }
  .panelbox :global(.art-panel) {
    flex: 1;
  }
  .popbox {
    padding: 10px;
    background: var(--bg2);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    display: flex;
  }
  .ftbox {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow: hidden;
  }
  .stack {
    display: grid;
    gap: 8px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .prose {
    margin: 0;
    font-size: var(--fs-md);
    line-height: 1.6;
    color: var(--tx1);
  }
  .doc-edit {
    width: 100%;
    min-height: 150px;
    resize: none;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 10px 12px;
    color: var(--tx0);
    font: 400 var(--fs-sm) / 1.6 var(--font-mono);
  }
  .doc-edit:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }

  .diff-del {
    color: var(--st-err);
  }
  .diff-add {
    color: var(--st-ok);
  }
</style>
