<script lang="ts">
  // Blinking token-stream caret.
  // "token caret" row - appended after in-flight streamed text (prose, code,
  // terminal output) to signal more tokens are still arriving. Purely
  // decorative: the streaming state itself is announced elsewhere (e.g. the
  // message body's aria-live/aria-busy), so this stays aria-hidden.
</script>

<span class="t-caret" aria-hidden="true"></span>

<style>
  .t-caret {
    display: inline-block;
    width: 7px;
    height: 13px;
    background: var(--acc);
    vertical-align: -2px;
    margin-left: 1px;
    animation: tblink 1.06s steps(2, start) infinite;
    border-radius: 1px;
  }
  @keyframes tblink {
    to {
      visibility: hidden;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .t-caret {
      animation: none;
      opacity: 0.65;
    }
  }
</style>
