<script lang="ts">
  import Ask from './Ask.svelte';

  const noop = () => {};
</script>

<section data-testid="gallery-ask">
  <div class="wrap">
    <figure>
      <figcaption>yes_no · approval (pending)</figcaption>
      <Ask
        heading="Approval required"
        alert
        source="session refactor: sse reconnect backoff"
        question="tsugite wants to run"
        command="git push origin refactor/sse-backoff"
        questionType="yes_no"
        affirmativeLabel="Approve & run"
        negativeLabel="Deny"
        alwaysLabel="Always allow git push"
        onAnswer={noop}
        onAlways={noop}
      />
    </figure>

    <figure>
      <figcaption>yes_no · plain (pending)</figcaption>
      <Ask
        source="chat: session cleanup"
        question="Delete the 3 merged worktrees now?"
        questionType="yes_no"
        onAnswer={noop}
      />
    </figure>

    <figure>
      <figcaption>choice (pending)</figcaption>
      <Ask
        source="research: pick embedding model"
        question="Which embedding model should we standardize on?"
        questionType="choice"
        options={['text-embedding-3-large', 'bge-large-en-v1.5', 'nomic-embed-text']}
        onAnswer={noop}
      />
    </figure>

    <figure>
      <figcaption>approval (pending)</figcaption>
      <Ask
        heading="Approval required"
        alert
        source="chat: research assistant"
        question="Fetch content from example.com?"
        questionType="approval"
        options={['Approve', 'Deny', 'Always allow']}
        onAnswer={noop}
      />
    </figure>

    <figure>
      <figcaption>text (pending)</figcaption>
      <Ask
        source="chat: name the release"
        question="What should we name this release?"
        questionType="text"
        onAnswer={noop}
      />
    </figure>

    <figure>
      <figcaption>resolved · approved (audit trail)</figcaption>
      <Ask
        heading="Approval required"
        source="session refactor: sse reconnect backoff"
        question="tsugite wants to run"
        command="git push origin refactor/sse-backoff"
        questionType="yes_no"
        resolution={{ tone: 'approved', text: 'approved · ran at 14:32 · exit 0' }}
        onAnswer={noop}
      />
    </figure>

    <figure>
      <figcaption>resolved · denied (audit trail)</figcaption>
      <Ask
        heading="Approval required"
        source="session refactor: sse reconnect backoff"
        question="tsugite wants to run"
        command="git push origin refactor/sse-backoff"
        questionType="yes_no"
        resolution={{ tone: 'denied', text: 'denied' }}
        onAnswer={noop}
      />
    </figure>

    <figure>
      <figcaption>resolved · answered (audit trail)</figcaption>
      <Ask
        source="chat: name the release"
        question="What should we name this release?"
        questionType="text"
        resolution={{ tone: 'approved', text: 'answered · “Backoff & Chill”' }}
        onAnswer={noop}
      />
    </figure>
  </div>
</section>

<style>
  .wrap {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: var(--sp-4);
    align-items: start;
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
</style>
