<script lang="ts">
  // Rendered preview of an untrusted HTML document, shown by the file surface
  // and by the agent artifact pane. Isolation policy: $lib/media/htmlPreview.
  // `docPath` resolves relative asset references; content with no path on disk
  // passes '' and gets no inlining.
  import { files } from '$lib/stores/files.svelte';
  import { loadWorkspaceDataURL } from '$lib/media/workspaceImage';
  import type { DocTheme } from '$lib/media/docTheme';
  import {
    HTML_SANDBOX,
    buildSrcdoc,
    inlineAssets,
    loadInlineAssets,
    type AssetReaders,
    type InlinedAsset,
  } from '$lib/media/htmlPreview';

  let {
    html,
    docPath = '',
    sessionId = null,
    docTheme = null,
    title,
    testid,
  }: {
    html: string;
    /** Workspace path the document lives at, for resolving relative assets. */
    docPath?: string;
    /** Session whose workspace the assets live in; null for the daemon default. */
    sessionId?: string | null;
    /** App theme to paint the document in, for generated content with no design
     *  of its own. Null leaves a document the colors its author gave it. */
    docTheme?: DocTheme | null;
    /** Iframe accessible name - screen readers announce the frame by it. */
    title: string;
    testid: string;
  } = $props();

  const readers: AssetReaders = {
    readText: async (path) => (await files.read(path, sessionId)).content ?? '',
    readDataUri: (path) => loadWorkspaceDataURL(path, sessionId),
  };

  let assets = $state(new Map<string, InlinedAsset>());

  // Assets load asynchronously, so the document paints without them and they
  // swap in when they settle. Keyed on the document alone, so re-theming a
  // mounted frame does not re-fetch them. `stale` drops a late reply whose
  // document has already been replaced.
  $effect(() => {
    const source = html;
    const path = docPath;
    assets = new Map();
    if (!path) return;
    let stale = false;
    void loadInlineAssets(source, path, readers).then((loaded) => {
      if (!stale) assets = loaded;
    });
    return () => {
      stale = true;
    };
  });

  // The theme sheet goes first, so the document's own styles override it.
  const srcdoc = $derived.by(() => {
    const doc = inlineAssets(html, assets);
    return buildSrcdoc(docTheme ? `<style>${docTheme.sheet}</style>${doc}` : doc);
  });
</script>

<iframe
  class="html-frame"
  style:color-scheme={docTheme?.scheme}
  {title}
  sandbox={HTML_SANDBOX}
  {srcdoc}
  data-testid={testid}
></iframe>

<style>
  .html-frame {
    width: 100%;
    height: 100%;
    min-height: 0;
    border: 0;
    /* An untouched document brings its own colors and generally assumes a light
       page (coverage output, generated docs), and it cannot see our theme
       tokens. So default the frame to the light color-scheme and let `Canvas`
       supply that scheme's page ground; a themed document overrides it. */
    color-scheme: light;
    background: Canvas;
  }
</style>
