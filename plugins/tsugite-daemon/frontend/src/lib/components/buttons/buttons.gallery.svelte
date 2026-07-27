<script lang="ts">
  import Button from './Button.svelte';
  import Badge from './Badge.svelte';
  import Dot from './Dot.svelte';
  import Pill from './Pill.svelte';
  import Chip from './Chip.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { PILL_STATES } from './pill-state';
  import { DOT_COLORS } from './dot-colors';

  const BUTTON_STATES = ['default', 'hover', 'focus', 'loading', 'disabled'] as const;
</script>

{#snippet retryIcon()}
  <Icon name="retry" />
{/snippet}
{#snippet stopIcon()}
  <Icon name="stop" />
{/snippet}
{#snippet dotsIcon()}
  <Icon name="dots" />
{/snippet}
{#snippet jobsIcon()}
  <Icon name="jobs" />
{/snippet}
{#snippet termIcon()}
  <Icon name="term" />
{/snippet}
{#snippet chatIcon()}
  <Icon name="chat" />
{/snippet}
{#snippet fileIcon()}
  <Icon name="file" />
{/snippet}
{#snippet agentIcon()}
  <Icon name="agent" />
{/snippet}

<section data-testid="gallery-buttons">
  <div class="variant">
    <span class="vlabel">button · variant × state</span>
    <div class="stack">
      {#each BUTTON_STATES as bst (bst)}
        <figure data-state={bst === 'default' ? undefined : bst}>
          <figcaption>{bst}</figcaption>
          <div class="row wrap">
            <Button variant="pri" loading={bst === 'loading'} disabled={bst === 'disabled'}>
              Answer question
            </Button>
            <Button loading={bst === 'loading'} disabled={bst === 'disabled'}>Retry</Button>
            <Button variant="danger" loading={bst === 'loading'} disabled={bst === 'disabled'}>
              Cancel job
            </Button>
            <Button variant="ghost" loading={bst === 'loading'} disabled={bst === 'disabled'}>
              Dismiss
            </Button>
          </div>
        </figure>
      {/each}
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">button · size · icon</span>
    <div class="row wrap">
      <Button variant="pri" size="sm">Approve</Button>
      <Button size="sm" icon={retryIcon}>Retry</Button>
      <Button size="sm" variant="danger" icon={stopIcon}>Stop</Button>
      <Button size="sm" iconOnly variant="ghost" icon={dotsIcon} aria-label="More actions" />
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">badge · informational vs action (in context)</span>
    <div class="row wrap row--loose">
      <div class="t-nav">
        {@render jobsIcon()}<span class="lb">Jobs</span>
        <span class="bdg">
          <Badge label="3 jobs running">3</Badge><Badge variant="action" label="1 job needs you"
            >1</Badge
          >
        </span>
      </div>
      <div class="t-nav">
        {@render termIcon()}<span class="lb">Terminals</span>
        <span class="bdg"><Badge label="2 live terminals">2</Badge></span>
      </div>
      <div class="t-nav">
        {@render chatIcon()}<span class="lb">Chat</span>
        <span class="bdg"><Badge variant="dot" label="unread" /></span>
      </div>
    </div>
    <div class="stack-inline">
      <figure>
        <figcaption>informational</figcaption>
        <Badge>12</Badge>
      </figure>
      <figure>
        <figcaption>informational</figcaption>
        <Badge>9+</Badge>
      </figure>
      <figure>
        <figcaption>action-required</figcaption>
        <Badge variant="action">2</Badge>
      </figure>
      <figure>
        <figcaption>errored</figcaption>
        <Badge variant="err">1</Badge>
      </figure>
      <figure>
        <figcaption>unread (dot)</figcaption>
        <Badge variant="dot" label="unread" />
      </figure>
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">dot</span>
    <div class="stack-inline">
      <figure>
        <figcaption>active</figcaption>
        <Dot color="ok" />
      </figure>
      <figure>
        <figcaption>live/streaming</figcaption>
        <Dot color="info" pulse />
      </figure>
      <figure>
        <figcaption>attention</figcaption>
        <Dot color="warn" />
      </figure>
      <figure>
        <figcaption>idle</figcaption>
        <Dot ring label="idle" />
      </figure>
    </div>
    <div class="stack-inline">
      {#each DOT_COLORS as color (color)}
        <figure>
          <figcaption>{color}</figcaption>
          <Dot {color} label={color} />
        </figure>
      {/each}
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">pill · session states</span>
    <div class="row wrap">
      {#each PILL_STATES as pst (pst)}
        <Pill st={pst} />
      {/each}
    </div>
  </div>

  <div class="variant">
    <span class="vlabel">chip</span>
    <div class="stack-inline">
      <figure>
        <figcaption>plain</figcaption>
        <Chip>terminal_open</Chip>
      </figure>
      <figure>
        <figcaption>icon + label</figcaption>
        <Chip icon={agentIcon}>odyn</Chip>
      </figure>
      <figure>
        <figcaption>removable</figcaption>
        <Chip icon={fileIcon} removable removeLabel="Remove backoff.patch">backoff.patch</Chip>
      </figure>
      <figure>
        <figcaption>reference (dashed)</figcaption>
        <Chip variant="ref" icon={fileIcon} removable removeLabel="Remove file reference">
          file: sse-reconnect.md
        </Chip>
      </figure>
    </div>
  </div>
</section>

<style>
  section {
    display: grid;
    gap: var(--sp-5);
  }
  .variant {
    display: grid;
    gap: var(--sp-2);
  }
  .vlabel {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .stack {
    display: grid;
    gap: var(--sp-3);
  }
  .stack-inline {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: var(--sp-4);
  }
  figure {
    margin: 0;
    display: grid;
    gap: var(--sp-1);
    justify-items: start;
  }
  figcaption {
    font: 500 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }
  .wrap {
    flex-wrap: wrap;
  }

  /* ---- .t-nav - shown here for the badge-in-context demo
     (t-nav itself belongs to a sidebar/nav group, not this one). ---- */
  .t-nav {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 29px;
    padding: 0 8px;
    border-radius: var(--r-md);
    color: var(--tx1);
    font: 500 var(--fs-md) / 1 var(--font-ui);
  }
  .t-nav :global(.ic) {
    width: 14px;
    height: 14px;
    color: var(--tx3);
  }
  .t-nav .bdg {
    margin-left: auto;
    display: flex;
    gap: 4px;
    align-items: center;
  }

  .row--loose {
    gap: 14px;
  }
</style>
