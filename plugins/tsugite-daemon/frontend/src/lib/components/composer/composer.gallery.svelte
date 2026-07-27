<script lang="ts">
  import Composer from './Composer.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import RefAutocomplete from './RefAutocomplete.svelte';
  import type { RefItem } from './types';

  const noop = () => {};

  // Exactly as shown on the card: file + chat + agent, state carried in detail text.
  const cardItems: RefItem[] = [
    { id: 'f', kind: 'file', label: '@sse-reconnect.md', detail: 'kb/ops · modified' },
    { id: 'c', kind: 'chat', label: '@sse-reconnect-backoff', detail: 'chat · working' },
    { id: 'a', kind: 'agent', label: '@odyn', detail: 'agent · opus-4-8' },
  ];

  // Every git file-state glyph (letter + color + title).
  const gitItems: RefItem[] = [
    { id: 'm', kind: 'file', label: '@sse-reconnect.md', detail: 'kb/ops', git: 'm' },
    { id: 'a', kind: 'file', label: '@new-note.md', detail: 'kb/ops', git: 'a' },
    { id: 'u', kind: 'file', label: '@runbook.md', detail: 'kb/ops', git: 'u' },
    { id: 'd', kind: 'file', label: '@old-notes.md', detail: 'kb/ops', git: 'd' },
  ];

  const kindItems: RefItem[] = [
    { id: 'f', kind: 'file', label: '@sse-reconnect.md', detail: 'kb/ops · modified', git: 'm' },
    { id: 't', kind: 'terminal', label: '@npm test', detail: 'terminal · running' },
    { id: 'c', kind: 'chat', label: '@backup prune', detail: 'chat · idle' },
    { id: 'a', kind: 'agent', label: '@odyn', detail: 'agent · opus-4-8' },
  ];
</script>

<section data-testid="gallery-composer">
  <h3 class="g-h">Composer</h3>
  <div class="g-cases">
    <figure>
      <figcaption>default · type @ or # for the live popover</figcaption>
      <div class="g-frame">
        <Composer refItems={kindItems} onSend={noop} onStop={noop} onAttach={noop} />
      </div>
    </figure>

    <figure>
      <figcaption>attachment · leading model chip · hint</figcaption>
      <div class="g-frame">
        <Composer
          value="summarize the reconnect regression"
          hint="est. +1.2k tok"
          attachments={[{ id: '1', name: 'sleep-wake.har', size: '118 KB' }]}
          onSend={noop}
          onStop={noop}
          onAttach={noop}
          onRemoveAttachment={noop}
        >
          {#snippet leading()}
            <span class="t-chip mono">sonnet-4.6 · med effort</span>
          {/snippet}
        </Composer>
      </div>
    </figure>

    <figure>
      <figcaption>add-context menu · attached context chip</figcaption>
      <div class="g-frame">
        <Composer
          value="where am i"
          contextItems={[
            {
              key: 'location',
              label: 'Location',
              value: '37.77490, -122.41940 (±20m)',
              icon: 'pin',
            },
          ]}
          contextMenu={[{ key: 'location', label: 'Location', icon: 'pin' }]}
          onSend={noop}
          onStop={noop}
          onAttach={noop}
          onPickContext={noop}
          onRemoveContext={noop}
        />
      </div>
    </figure>

    <figure>
      <figcaption>streaming — Send becomes Stop</figcaption>
      <div class="g-frame">
        <Composer streaming value="add jitter to reconnect backoff" onSend={noop} onStop={noop} />
      </div>
    </figure>

    <figure>
      <figcaption>queued — sends when the turn finishes</figcaption>
      <div class="g-frame">
        <Composer queued value="and then run the suite" onSend={noop} onStop={noop} />
      </div>
    </figure>

    <figure>
      <figcaption>large-paste affordance slot · kbd strip</figcaption>
      <div class="g-frame">
        <Composer showKbd onSend={noop} onStop={noop} onAttach={noop}>
          {#snippet pasteAffordance()}
            <span class="t-chip mono">
              <Icon name="file" />pasted 4.2 KB → attachment
            </span>
          {/snippet}
        </Composer>
      </div>
    </figure>
  </div>

  <h3 class="g-h">RefAutocomplete — MentionPopover (@ / #)</h3>
  <p class="g-note">
    Live: type <code>@</code> or <code>#</code> in any composer above to open it.
  </p>
  <div class="g-cases g-cases--pop">
    <figure>
      <figcaption>as on the card · first row active</figcaption>
      <RefAutocomplete items={cardItems} onSelect={noop} />
    </figure>

    <figure>
      <figcaption>highlight on row 2</figcaption>
      <RefAutocomplete items={cardItems} activeIndex={1} onSelect={noop} />
    </figure>

    <figure>
      <figcaption>git file-state glyphs · M / A / ? / D</figcaption>
      <RefAutocomplete items={gitItems} onSelect={noop} />
    </figure>

    <figure>
      <figcaption>reference kinds · file · terminal · chat · agent</figcaption>
      <RefAutocomplete items={kindItems} onSelect={noop} />
    </figure>

    <figure>
      <figcaption>single result</figcaption>
      <RefAutocomplete items={[cardItems[0]!]} onSelect={noop} />
    </figure>
  </div>
</section>

<style>
  .g-h {
    margin: var(--sp-4) 0 var(--sp-2);
    font: 600 var(--fs-md) var(--font-ui);
    color: var(--tx1);
  }
  .g-note {
    margin: 0 0 var(--sp-3);
    font: 400 var(--fs-sm) var(--font-ui);
    color: var(--tx3);
  }
  .g-note code {
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
  }
  .g-cases {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-4);
    align-items: flex-start;
  }
  .g-cases--pop {
    align-items: flex-start;
  }
  figure {
    margin: 0;
    display: grid;
    gap: var(--sp-1);
  }
  figcaption {
    font: 500 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.04em;
    color: var(--tx3);
  }
  .g-frame {
    width: min(560px, 92vw);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    overflow: hidden;
    background: var(--bg1);
  }

  /* .t-chip kept inline (not swapped to the Chip component): it only fills the
     composer's leading / paste-affordance snippet slots here, and its file icon
     renders at 13px — Chip forces its icon to 10px, which would shrink it. */
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
  .mono {
    font-family: var(--font-mono);
  }
</style>
