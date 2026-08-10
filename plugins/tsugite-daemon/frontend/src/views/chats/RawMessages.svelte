<script lang="ts">
  // Debug overlay: the raw request messages the model saw and its raw response,
  // per turn, reconstructed on demand from the persisted event log. Wide, since raw
  // messages dwarf the context-meter popover this is reached from. Each block
  // scrolls inside its own frame and long tokens wrap, so nothing widens the page.
  import { onMount } from 'svelte';
  import RawOverlay from './RawOverlay.svelte';
  import { auth } from '$lib/stores/auth.svelte';
  import { fetchRawMessages, type RawMessage, type RawMessages, type RawTurn } from './rawMessages';

  let { agent, sessionId, onClose }: { agent: string; sessionId: string; onClose: () => void } =
    $props();

  let data = $state<RawMessages | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let copied = $state<string | null>(null);
  // Per-call collapse state (all closed on load): a call's request is the whole
  // conversation up to it, so auto-expanding any entry dumps the full chat.
  let openTurns = $state<boolean[]>([]);
  const anyOpen = $derived(openTurns.some(Boolean));
  function toggleAll(): void {
    const next = !anyOpen;
    openTurns = openTurns.map(() => next);
  }

  onMount(() => {
    void load();
  });

  async function load(): Promise<void> {
    try {
      data = await fetchRawMessages(agent, sessionId, auth.userId ?? undefined);
      openTurns = data?.turns.map(() => false) ?? [];
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  const contentText = (content: unknown): string =>
    typeof content === 'string' ? content : JSON.stringify(content, null, 2);

  const requestText = (request: RawMessage[]): string =>
    request.map((m) => `[${m.role}]\n${contentText(m.content)}`).join('\n\n');

  // A call's request is the whole conversation it saw, so consecutive calls repeat
  // everything before them. Show just what this call added, with the full prompt a
  // click away; after a compaction the prompt reset, so show it whole.
  const shownMessages = (t: RawTurn): RawMessage[] => (t.reset_before ? t.request : t.new_messages);
  const hasHiddenPrefix = (t: RawTurn): boolean =>
    !t.reset_before && t.request.length > t.new_messages.length;
  const reqLabel = (t: RawTurn): string =>
    t.reset_before
      ? 'request · reset after compaction'
      : hasHiddenPrefix(t)
        ? 'request · new this call'
        : 'request';

  async function copy(text: string, key: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(text);
      copied = key;
      setTimeout(() => {
        if (copied === key) copied = null;
      }, 1200);
    } catch {
      // Clipboard unavailable (insecure context / denied): the view still reads.
    }
  }
</script>

{#snippet copyBtn(text: string, key: string)}
  <button type="button" class="raw-copy" onclick={() => copy(text, key)}>
    {copied === key ? 'copied' : 'copy'}
  </button>
{/snippet}

<RawOverlay title="raw messages" {onClose}>
  {#snippet actions()}
    {#if data && data.turns.length > 0}
      <button type="button" class="raw-copy" onclick={toggleAll}>
        {anyOpen ? 'collapse all' : 'expand all'}
      </button>
    {/if}
  {/snippet}

  {#if loading}
    <p class="raw-note">loading…</p>
  {:else if error}
    <p class="raw-note">couldn't load raw messages: {error}</p>
  {:else if data}
    <section class="raw-sec">
      <div class="raw-sec-hd">
        <span>system prompt</span>
        {#if data.system_prompt}{@render copyBtn(data.system_prompt, 'sys')}{/if}
      </div>
      {#if data.system_prompt}
        <div class="raw-block"><pre>{data.system_prompt}</pre></div>
      {:else}
        <p class="raw-note">system prompt not shown</p>
      {/if}
    </section>

    {#if data.turns.length === 0}
      <p class="raw-note">no model turns recorded yet</p>
    {:else}
      <p class="raw-note raw-intro">
        Each entry is one model call, newest last. Its request is the whole conversation the model
        saw at that point; by default only what the call added is shown, with the full prompt one
        click away.
      </p>
    {/if}

    {#each data.turns as t, i (i)}
      <details class="raw-turn" bind:open={openTurns[i]}>
        <summary
          >call {t.index}{t.provider ? ` · ${t.provider}` : ''}{t.model
            ? ` · ${t.model}`
            : ''}</summary
        >
        <div class="raw-sub">
          <div class="raw-sec-hd">
            <span>{reqLabel(t)}</span>
            {@render copyBtn(requestText(t.request), `req-${i}`)}
          </div>
          {#if t.reset_before}
            <p class="raw-note">
              context was compacted just before this call; the prompt reset to the summary below.
            </p>
          {/if}
          {#if shownMessages(t).length === 0}
            <p class="raw-note">no new messages (identical prompt to the previous call)</p>
          {/if}
          {#each shownMessages(t) as m, j (j)}
            <div class="raw-msg">
              <span class="raw-role">{m.role}</span>
              <div class="raw-block"><pre>{contentText(m.content)}</pre></div>
            </div>
          {/each}
          {#if hasHiddenPrefix(t)}
            <details class="raw-full">
              <summary>full prompt · {t.request.length} messages</summary>
              {#each t.request as m, j (j)}
                <div class="raw-msg">
                  <span class="raw-role">{m.role}</span>
                  <div class="raw-block"><pre>{contentText(m.content)}</pre></div>
                </div>
              {/each}
            </details>
          {/if}
        </div>
        <div class="raw-sub">
          <div class="raw-sec-hd">
            <span>response</span>
            {#if t.response}{@render copyBtn(t.response.raw_content, `res-${i}`)}{/if}
          </div>
          {#if t.response}
            <div class="raw-block"><pre>{t.response.raw_content}</pre></div>
          {:else}
            <p class="raw-note">no response recorded</p>
          {/if}
        </div>
      </details>
    {/each}
  {/if}
</RawOverlay>

<style>
  .raw-note {
    margin: 0;
    color: var(--tx3);
    font-size: var(--fs-sm);
  }
  .raw-sec {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }
  .raw-sec-hd {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .raw-turn {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 10px 12px;
    background: var(--bg1);
  }
  .raw-turn > summary {
    cursor: pointer;
    font: 600 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
  }
  .raw-intro {
    margin: -6px 0 -2px;
    line-height: 1.5;
  }
  .raw-full {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 2px;
  }
  .raw-full > summary {
    cursor: pointer;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    padding: 2px 0;
  }
  .raw-full > summary:hover {
    color: var(--tx1);
  }
  .raw-sub {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 11px;
  }
  .raw-msg {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .raw-role {
    font: 600 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--tx2);
  }
  .raw-block {
    overflow: auto;
    max-height: 40vh;
    background: var(--bg0);
    border: 1px solid var(--bd0);
    border-radius: var(--r-sm);
  }
  .raw-block pre {
    margin: 0;
    padding: 8px 10px;
    font: 400 var(--fs-xs) / 1.5 var(--font-mono);
    color: var(--tx1);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
</style>
