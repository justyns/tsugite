<script lang="ts">
  // Inline question / approval block (`.t-ask`). Presentational:
  // it renders an ask_user prompt (yes_no / choice / text) and its answered
  // audit-trail state, and reports the user's answer through `onAnswer`. The
  // payload shape mirrors the daemon's ask_user event
  // ({ question, question_type, options }); resolution is supplied by the caller.
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import type { IconName } from '$lib/components/icon/icons';
  import { TESTID } from '$lib/testids';

  type QuestionType = 'yes_no' | 'choice' | 'text' | 'approval';
  type AskTone = 'approved' | 'denied';

  let {
    question,
    questionType = 'text',
    options = [],
    command,
    source,
    heading = 'Question',
    alert = false,
    affirmativeLabel = 'Yes',
    negativeLabel = 'No',
    alwaysLabel,
    submitLabel = 'Send',
    resolution = null,
    onAnswer,
    onAlways,
  }: {
    /** The question text; rendered as-is (plus `command` in mono, if given). */
    question: string;
    /** Which body to render. Mirrors the daemon's `question_type`. */
    questionType?: QuestionType;
    /** Options for `choice` mode. */
    options?: string[];
    /** Optional command highlighted in mono at the end of the question line. */
    command?: string;
    /** Muted session/source label shown after the heading. */
    source?: string;
    /** Header label. Defaults to "Question". */
    heading?: string;
    /** Use the alert glyph (permission gate) instead of the question glyph. */
    alert?: boolean;
    /** yes_no affirmative button label (emits `"yes"`). */
    affirmativeLabel?: string;
    /** yes_no negative button label (emits `"no"`). */
    negativeLabel?: string;
    /** When set, renders an opt-in "always allow" ghost button wired to `onAlways`. */
    alwaysLabel?: string;
    /** Submit-button label for choice / text modes. */
    submitLabel?: string;
    /** When set, the block is an inert answered record instead of a live prompt. */
    resolution?: { tone: AskTone; text: string } | null;
    /** Called with the chosen answer string. */
    onAnswer: (value: string) => void;
    /** Called when the "always allow" button is pressed. */
    onAlways?: () => void;
  } = $props();

  const uid = $props.id();

  let choice = $state('');
  let draft = $state('');

  const toneClass = $derived(
    resolution ? (resolution.tone === 'approved' ? 'is-approved' : 'is-denied') : '',
  );
  const headIcon = $derived<IconName>(
    resolution ? (resolution.tone === 'approved' ? 'check' : 'x') : alert ? 'alert' : 'q',
  );
  const canSubmitChoice = $derived(choice !== '');
  const canSubmitText = $derived(draft.trim() !== '');

  // Approval options arrive in a fixed order (Approve, Deny, optional "Always allow").
  // Approve leads as the primary action; a trailing "always" is de-emphasized as a
  // ghost so it is not fat-fingered; anything between stays a neutral secondary.
  function approvalVariant(i: number): 'pri' | 'ghost' | 'default' {
    if (i === 0) return 'pri';
    if (options.length > 2 && i === options.length - 1) return 'ghost';
    return 'default';
  }

  function submitText() {
    if (canSubmitText) onAnswer(draft);
  }
  function onTextKeydown(e: KeyboardEvent) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      submitText();
    }
  }
</script>

<div class="t-ask {toneClass}" role="group" aria-labelledby="{uid}-hd">
  <div class="hd">
    <Icon name={headIcon} />
    <span id="{uid}-hd">{heading}</span>
    {#if source}<span class="src">· {source}</span>{/if}
  </div>

  <div class="q" id="{uid}-q">
    {question}{#if command}&nbsp;<code>{command}</code>{/if}
  </div>

  {#if resolution}
    <div class="res">
      <Icon name={headIcon} />
      <span class="res-txt">{resolution.text}</span>
    </div>
  {:else if questionType === 'choice'}
    <div class="opts" role="radiogroup" aria-labelledby="{uid}-q">
      {#each options as opt (opt)}
        <label class="opt">
          <input type="radio" name="{uid}-choice" value={opt} bind:group={choice} />
          {opt}
        </label>
      {/each}
    </div>
    <div class="fx">
      <Button variant="pri" size="sm" disabled={!canSubmitChoice} onclick={() => onAnswer(choice)}>
        {submitLabel}
      </Button>
    </div>
  {:else if questionType === 'approval'}
    <div class="fx" data-testid={TESTID.askApproval}>
      {#each options as opt, i (opt)}
        <Button variant={approvalVariant(i)} size="sm" onclick={() => onAnswer(opt)}>
          {opt}
        </Button>
      {/each}
    </div>
  {:else if questionType === 'text'}
    <textarea
      class="t-input"
      aria-labelledby="{uid}-q"
      placeholder="type your answer…"
      bind:value={draft}
      onkeydown={onTextKeydown}></textarea>
    <div class="fx">
      <Button variant="pri" size="sm" disabled={!canSubmitText} onclick={submitText}>
        {submitLabel}
      </Button>
    </div>
  {:else}
    <div class="fx">
      <Button variant="pri" size="sm" onclick={() => onAnswer('yes')}>
        {affirmativeLabel}
      </Button>
      <Button size="sm" onclick={() => onAnswer('no')}>
        {negativeLabel}
      </Button>
      {#if alwaysLabel}
        <Button variant="ghost" size="sm" onclick={() => onAlways?.()}>
          {alwaysLabel}
        </Button>
      {/if}
    </div>
  {/if}
</div>

<style>
  /* ---- inline question / approval ---- */
  .t-ask {
    border: 1px solid color-mix(in oklab, var(--st-warn) 45%, transparent);
    background: color-mix(in oklab, var(--st-warn) 8%, transparent);
    border-radius: var(--r-lg);
    padding: 11px 13px;
    display: grid;
    gap: 9px;
    max-width: 600px;
  }
  .t-ask .hd {
    display: flex;
    align-items: center;
    gap: 7px;
    font: 600 var(--fs-sm) / 1 var(--font-ui);
    color: var(--st-warn);
  }
  .t-ask .hd .src {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    font-weight: 400;
  }
  .t-ask .q {
    font-size: var(--fs-md);
    color: var(--tx0);
    line-height: 1.5;
    text-wrap: pretty;
  }
  .t-ask .q code {
    font: 500 var(--fs-sm) var(--font-mono);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    padding: 1px 5px;
    border-radius: 4px;
  }
  .t-ask .opts {
    display: grid;
    gap: 4px;
  }
  .t-ask .opt {
    display: flex;
    gap: 8px;
    align-items: center;
    padding: 5px 8px;
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    background: var(--bg1);
    cursor: pointer;
    font-size: var(--fs-sm);
    color: var(--tx1);
  }
  .t-ask .opt:hover {
    border-color: var(--st-warn);
  }
  .t-ask .opt input {
    accent-color: var(--st-warn);
    margin: 0;
  }
  .t-ask .fx {
    display: flex;
    gap: 7px;
    align-items: center;
    flex-wrap: wrap;
  }
  /* Answered record: `.res` is only in the DOM when resolved, so it is shown
     directly (a display:none/reveal dance is unnecessary here). */
  .t-ask .res {
    display: flex;
    align-items: center;
    gap: 7px;
    font: 500 var(--fs-sm) var(--font-mono);
  }
  .t-ask.is-approved {
    border-color: color-mix(in oklab, var(--st-ok) 40%, transparent);
    background: color-mix(in oklab, var(--st-ok) 6%, transparent);
  }
  .t-ask.is-approved .hd {
    color: var(--st-ok);
  }
  .t-ask.is-denied {
    border-color: var(--bd1);
    background: var(--bg1);
  }
  .t-ask.is-denied .hd {
    color: var(--tx3);
  }
  .t-ask.is-approved .res {
    color: var(--st-ok);
  }
  .t-ask.is-denied .res {
    color: var(--tx3);
  }

  /* .t-input stays inline: this is a multi-line <textarea>, whereas the shared
     Input component renders an <input> only — a swap would change the element and
     its behavior. */
  .t-input {
    width: 100%;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    color: var(--tx0);
    font: 400 var(--fs-md) var(--font-ui);
    transition:
      border-color var(--t-1),
      box-shadow var(--t-1);
    height: auto;
    min-height: 34px;
    padding: 7px 9px;
    resize: none;
    line-height: 1.5;
  }
  .t-input::placeholder {
    color: var(--tx3);
  }
  .t-input:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
</style>
