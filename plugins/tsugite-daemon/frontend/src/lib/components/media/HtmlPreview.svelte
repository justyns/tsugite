<script lang="ts">
  // Rendered preview of an untrusted HTML document, shown by the file surface
  // and by the agent artifact pane. Isolation policy: $lib/media/htmlPreview.
  // `docPath` resolves relative asset references; content with no path on disk
  // passes '' and gets no inlining.
  import { files } from '$lib/stores/files.svelte';
  import { loadWorkspaceDataURL } from '$lib/media/workspaceImage';
  import {
    HTML_SANDBOX,
    buildSrcdoc,
    inlineAssets,
    loadInlineAssets,
    type AssetReaders,
  } from '$lib/media/htmlPreview';

  let {
    html,
    docPath = '',
    sessionId = null,
    title,
    testid,
  }: {
    html: string;
    /** Workspace path the document lives at, for resolving relative assets. */
    docPath?: string;
    /** Session whose workspace the assets live in; null for the daemon default. */
    sessionId?: string | null;
    /** Iframe accessible name - screen readers announce the frame by it. */
    title: string;
    testid: string;
  } = $props();

  const readers: AssetReaders = {
    readText: async (path) => (await files.read(path, sessionId)).content ?? '',
    readDataUri: (path) => loadWorkspaceDataURL(path, sessionId),
  };

  let srcdoc = $state('');

  // Assets load asynchronously, so paint the policy-wrapped document immediately
  // and swap in the inlined version when it settles. `stale` drops a late reply
  // whose document has already been replaced.
  $effect(() => {
    const source = html;
    const path = docPath;
    let stale = false;
    srcdoc = buildSrcdoc(source);
    if (path) {
      void loadInlineAssets(source, path, readers).then((assets) => {
        if (!stale && assets.size > 0) srcdoc = buildSrcdoc(inlineAssets(source, assets));
      });
    }
    return () => {
      stale = true;
    };
  });
</script>

<iframe class="html-frame" {title} sandbox={HTML_SANDBOX} {srcdoc} data-testid={testid}></iframe>

<style>
  .html-frame {
    width: 100%;
    height: 100%;
    min-height: 0;
    border: 0;
    /* The document brings its own colors and generally assumes a light page
       (coverage output, generated docs), and it cannot see our theme tokens. So
       pin the frame to the light color-scheme and let `Canvas` supply that
       scheme's page ground. */
    color-scheme: light;
    background: Canvas;
  }
</style>
