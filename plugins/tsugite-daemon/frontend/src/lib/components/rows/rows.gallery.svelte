<script lang="ts">
  import SessionRow from './SessionRow.svelte';
  import SpacesRow from './SpacesRow.svelte';
  import CheckItem from './CheckItem.svelte';

  const noop = () => {};
</script>

<section data-testid="gallery-rows">
  <h3>SessionRow — every state</h3>
  <div class="list">
    <figure>
      <figcaption>state: running</figcaption>
      <div class="frame">
        <SessionRow
          title="refactor: sse reconnect backoff"
          when="now"
          description="streaming a reply"
          state="running"
          sourceType="code"
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>state: thinking</figcaption>
      <div class="frame">
        <SessionRow
          title="plan: migrate history to sqlite"
          when="2m"
          description="reasoning about the migration"
          state="thinking"
          sourceType="code"
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>state: idle</figcaption>
      <div class="frame">
        <SessionRow
          title="chat: naming things"
          when="1d"
          description="tsugite means joinery"
          state="idle"
          sourceType="chat"
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>state: done</figcaption>
      <div class="frame">
        <SessionRow
          title="add jitter to backoff"
          when="3h"
          description="5/5 acceptance criteria passed"
          state="done"
          sourceType="code"
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>state: failed</figcaption>
      <div class="frame">
        <SessionRow
          title="ops: restore drill"
          when="6h"
          description="attempts exhausted — see log"
          state="failed"
          sourceType="ops"
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>state: needs-you</figcaption>
      <div class="frame">
        <SessionRow
          title="ops: nightly backup failing on prune"
          when="12m"
          description="job blocked on a retention question"
          state="needs-you"
          sourceType="ops"
          onSelect={noop}
        />
      </div>
    </figure>
  </div>

  <h3>SessionRow — variants</h3>
  <div class="list">
    <figure>
      <figcaption>variant: primary (currently open)</figcaption>
      <div class="frame">
        <SessionRow
          title="refactor: sse reconnect backoff"
          when="now"
          description="streaming a reply"
          state="running"
          sourceType="code"
          isActive
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>variant: pinned</figcaption>
      <div class="frame">
        <SessionRow
          title="research: local whisper models"
          when="2h"
          description="3 candidates benchmarked"
          state="idle"
          sourceType="research"
          isPinned
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>variant: unread</figcaption>
      <div class="frame">
        <SessionRow
          title="research: local whisper models"
          when="2h"
          description="3 candidates benchmarked"
          state="idle"
          sourceType="research"
          isUnread
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>variant: active job count</figcaption>
      <div class="frame">
        <SessionRow
          title="refactor: sse reconnect backoff"
          when="now"
          description="streaming a reply"
          state="running"
          sourceType="code"
          activeJobCount={1}
          onSelect={noop}
        />
      </div>
    </figure>
  </div>

  <h3>SpacesRow — per-space rollup states</h3>
  <div class="list">
    <figure>
      <figcaption>state: working</figcaption>
      <div class="frame">
        <SpacesRow
          title="refactor: sse reconnect backoff"
          who="odyn · sonnet-4.6"
          state="working"
          contextPct={3}
          contextTokens="34k"
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>state: blocked</figcaption>
      <div class="frame">
        <SpacesRow
          title="nightly backup prune policy"
          who="ops-runner · waiting 12m"
          state="blocked"
          contextPct={41}
          contextTokens="82k"
          contextWarn
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>state: idle</figcaption>
      <div class="frame">
        <SpacesRow
          title="research: local whisper models"
          who="odyn · opus-4.6"
          state="idle"
          contextPct={12}
          contextTokens="24k"
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>state: done</figcaption>
      <div class="frame">
        <SpacesRow
          title="add jitter to backoff"
          who="code-worker · 5/5 criteria"
          state="done"
          contextPct={9}
          contextTokens="18k"
          onSelect={noop}
        />
      </div>
    </figure>
    <figure>
      <figcaption>variant: primary (open in the multiplexer)</figcaption>
      <div class="frame">
        <SpacesRow
          title="refactor: sse reconnect backoff"
          who="odyn · sonnet-4.6"
          state="working"
          contextPct={3}
          contextTokens="34k"
          isActive
          onSelect={noop}
        />
      </div>
    </figure>
  </div>

  <h3>CheckItem — acceptance-criteria states</h3>
  <div class="list list--check">
    <figure>
      <figcaption>state: pending</figcaption>
      <div class="frame">
        <CheckItem label="restore test passes on staging" state="pending" />
      </div>
    </figure>
    <figure>
      <figcaption>state: active (being verified now)</figcaption>
      <div class="frame">
        <CheckItem label="no orphaned blobs in object store" state="active" />
      </div>
    </figure>
    <figure>
      <figcaption>state: pass</figcaption>
      <div class="frame">
        <CheckItem label="backup completes in under 10 minutes" state="pass" />
      </div>
    </figure>
    <figure>
      <figcaption>state: fail (verifier reason inline)</figcaption>
      <div class="frame">
        <CheckItem
          label="disk usage below 80% after prune"
          state="fail"
          note="verifier: 84% — prune kept 22 weeklies"
        />
      </div>
    </figure>
  </div>
</section>

<style>
  h3 {
    margin: var(--sp-5) 0 var(--sp-2);
    font: 600 var(--fs-xs) / 1 var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  h3:first-child {
    margin-top: 0;
  }
  .list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: var(--sp-4);
    align-items: start;
  }
  .list--check {
    display: block;
  }
  .list--check figure {
    margin-bottom: var(--sp-3);
  }
  figure {
    margin: 0;
    display: grid;
    gap: var(--sp-2);
  }
  figcaption {
    font: 500 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .frame {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow: hidden;
    background: var(--bg1);
  }
  .list--check .frame {
    padding: 0 var(--sp-3);
  }
</style>
