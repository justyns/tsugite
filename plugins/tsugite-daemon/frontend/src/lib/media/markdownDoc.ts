/**
 * Wrap rendered markdown for display inside the sandboxed preview frame
 * (`$lib/media/htmlPreview`), which is where agent-supplied markdown renders so
 * its raw HTML can never touch the app's document.
 *
 * The colors come from the theme sheet `$lib/media/docTheme` injects ahead of
 * this one, so a generated document reads as part of the app on any theme. That
 * sheet also supplies the page ground, the body text colour and the link colour,
 * which is why no rule here sets them. The frame's CSP allows both through
 * `style-src 'unsafe-inline'`.
 *
 * The fonts stay literal stacks: the frame reaches no network, so the app's
 * webfonts would never load in it.
 */

const MARKDOWN_CSS = `
body {
  margin: 0;
  padding: 16px 20px 30px;
  max-width: 72ch;
  font: 15px/1.62 system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  overflow-wrap: break-word;
}
h1, h2, h3 { color: var(--tx0); line-height: 1.3; }
h1 { font-size: 1.6em; margin: 0 0 10px; }
h2 { font-size: 1.25em; margin: 20px 0 7px; padding-bottom: 4px; border-bottom: 1px solid var(--bd0); }
h3 { font-size: 1.05em; margin: 16px 0 6px; }
p { margin: 7px 0; }
ul, ol { margin: 6px 0; padding-left: 20px; }
li { margin: 3px 0; }
strong { color: var(--tx0); }
img { max-width: 100%; }
hr { border: 0; border-top: 1px solid var(--bd0); margin: 16px 0; }
code {
  font: 500 0.9em ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--bg3);
  border: 1px solid var(--bd0);
  border-radius: 4px;
  padding: 0 4px;
}
pre {
  background: var(--bg2);
  border: 1px solid var(--bd0);
  border-radius: 6px;
  padding: 10px 12px;
  overflow-x: auto;
}
pre code { background: none; border: 0; padding: 0; }
table { border-collapse: collapse; margin: 10px 0; font-size: 0.92em; }
th {
  text-align: left;
  color: var(--tx2);
  border-bottom: 1px solid var(--bd1);
  padding: 5px 12px 5px 0;
}
td { border-bottom: 1px solid var(--bd0); padding: 5px 12px 5px 0; }
blockquote {
  margin: 10px 0;
  padding: 8px 12px;
  border-left: 3px solid var(--acc);
  background: var(--bg2);
  border-radius: 6px;
}
blockquote p { margin: 0; }
.tsu-fm table { table-layout: fixed; width: 100%; }
.tsu-fm th { width: 14ch; vertical-align: top; }
.tsu-fm td { overflow-wrap: anywhere; }
.tsu-fm td pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
.wikilink { color: var(--brand); border-bottom: 1px dashed var(--brand); text-decoration: none; }
.wikilink.is-missing { color: var(--st-err); border-bottom-color: var(--st-err); }
.vh {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
}
`;

export function markdownDoc(html: string): string {
  return `<style>${MARKDOWN_CSS}</style>${html}`;
}
