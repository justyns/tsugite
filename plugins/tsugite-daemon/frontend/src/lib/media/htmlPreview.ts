/**
 * Rendered-HTML preview for workspace `.html` / `.htm` files.
 *
 * The document is untrusted: whatever an agent, a coverage run, or a doc
 * generator wrote to disk. Two independent layers contain it, and loosening
 * either one is a security change.
 *
 * 1. `HTML_SANDBOX` is the empty string, so the iframe is granted no sandbox
 *    capability at all. Without `allow-same-origin` the frame runs in an opaque
 *    origin: it cannot read the app's `localStorage` bearer token, reach
 *    `window.parent`, or call the daemon API as the user.
 * 2. `buildSrcdoc` wraps the document in a <head> carrying `HTML_CSP`, so the
 *    page reaches no network even if the sandbox were loosened.
 *    `default-src 'none'` plus three carve-outs: `style-src 'unsafe-inline'`
 *    (an inlined sheet lands as a <style> block), `img-src data:` and
 *    `font-src data:` (inlined assets are data: URIs). No `http(s):` source
 *    appears, so viewing a local file cannot phone home.
 *
 * <script> tags stay in the source; the raw view shows the file as it is on
 * disk, and the two layers above are what stop them running.
 *
 * Relative asset references resolve against the document's own directory, are
 * read through the authenticated workspace API, and are inlined by
 * `inlineAssets`. Anything else - an absolute URL, `//host/x`, `data:`,
 * `javascript:`, or a path above the workspace - resolves to null, is left as
 * written, and is blocked by the CSP.
 */

const HTML_EXT = /\.html?$/i;

export function isHtml(name: string): boolean {
  return HTML_EXT.test(name);
}

/** Empty sandbox token list = every sandbox capability denied. See the module
 *  comment; do not add tokens without re-reading it. */
export const HTML_SANDBOX = '';

/** Injected into every previewed document as a <meta http-equiv>. */
export const HTML_CSP = [
  "default-src 'none'",
  'img-src data:',
  "style-src 'unsafe-inline'",
  'font-src data:',
  "form-action 'none'",
  "base-uri 'none'",
].join('; ');

export interface AssetRef {
  /** The href/src exactly as written in the document - the key `inlineAssets`
   *  looks up, so callers must not normalize it. */
  href: string;
  kind: 'style' | 'image';
}

export type InlinedAsset = { kind: 'style'; css: string } | { kind: 'image'; dataUri: string };

/** <link> and <img> tags. Attributes are everything up to the closing `>`,
 *  trailing `/` included (parseAttrs ignores it). */
const ASSET_TAG = /<(link|img)\b([^>]*)>/gi;
const ATTR = /([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/g;

function parseAttrs(raw: string): [string, string][] {
  const out: [string, string][] = [];
  ATTR.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = ATTR.exec(raw)) !== null) {
    out.push([m[1]!.toLowerCase(), m[2] ?? m[3] ?? m[4] ?? '']);
  }
  return out;
}

function attrValue(attrs: [string, string][], name: string): string | undefined {
  return attrs.find(([k]) => k === name)?.[1];
}

/** The asset a <link>/<img> tag references, or null when it references none we
 *  care about (a preload/icon link, an <img> with no src). */
function assetRef(tag: string, attrs: [string, string][]): AssetRef | null {
  if (tag === 'img') {
    const src = attrValue(attrs, 'src');
    return src ? { href: src, kind: 'image' } : null;
  }
  const rel = (attrValue(attrs, 'rel') ?? '').toLowerCase().split(/\s+/);
  if (!rel.includes('stylesheet')) return null;
  const href = attrValue(attrs, 'href');
  return href ? { href, kind: 'style' } : null;
}

/** Every stylesheet <link> and <img> the document references, in source order,
 *  de-duplicated by (kind, href). Callers filter these through
 *  `resolveWorkspaceAsset` before fetching anything. */
export function collectAssetRefs(html: string): AssetRef[] {
  const out: AssetRef[] = [];
  const seen = new Set<string>();
  ASSET_TAG.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = ASSET_TAG.exec(html)) !== null) {
    const ref = assetRef(m[1]!.toLowerCase(), parseAttrs(m[2]!));
    if (!ref) continue;
    const key = `${ref.kind} ${ref.href}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(ref);
  }
  return out;
}

/**
 * Resolve one document-relative href to a workspace-relative path, or null when
 * it is not a workspace file we may read.
 *
 * Rejected (null): absolute URLs of any scheme, protocol-relative `//host/x`,
 * fragment-only links, empty hrefs, and anything climbing above the workspace
 * root. A leading `/` reads as workspace-root-relative, which is how a report
 * generated for a static server addresses its siblings.
 *
 * This is a client-side convenience only - the daemon re-validates every path it
 * is asked to read, so a bug here cannot widen what the server will serve.
 */
export function resolveWorkspaceAsset(docPath: string, href: string): string | null {
  const bare = href.split('#')[0]!.split('?')[0]!.trim();
  if (!bare || bare.startsWith('//') || /^[A-Za-z][A-Za-z0-9+.-]*:/.test(bare)) return null;

  const base = bare.startsWith('/') ? [] : docPath.split('/').slice(0, -1);
  const stack: string[] = [];
  for (const part of [...base, ...bare.split('/')]) {
    if (part === '' || part === '.') continue;
    if (part === '..') {
      if (stack.length === 0) return null; // climbs out of the workspace
      stack.pop();
      continue;
    }
    stack.push(part);
  }
  return stack.length ? stack.join('/') : null;
}

/** `</style` inside CSS would close the block we wrap it in. */
function escapeStyleBody(css: string): string {
  return css.replace(/<\/(style)/gi, '<\\/$1');
}

function replaceSrc(tag: string, dataUri: string): string {
  // Rewrites the src value in place rather than re-serializing the attribute
  // list, so entities in the untouched attributes (alt="a &amp; b") survive.
  return tag.replace(
    /(\bsrc\s*=\s*)(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+)/i,
    (_m, lead: string) => `${lead}"${dataUri.replace(/"/g, '%22')}"`,
  );
}

/**
 * Replace each asset reference the caller managed to read with its inline form:
 * a stylesheet <link> becomes a <style> block, an <img> keeps every attribute
 * but points at a data: URI. References missing from `resolved` are left exactly
 * as written - the CSP then blocks them.
 */
export function inlineAssets(html: string, resolved: Map<string, InlinedAsset>): string {
  if (resolved.size === 0) return html;
  return html.replace(ASSET_TAG, (whole, rawTag: string, rawAttrs: string) => {
    const tag = rawTag.toLowerCase();
    const ref = assetRef(tag, parseAttrs(rawAttrs));
    const asset = ref && resolved.get(ref.href);
    if (!asset || asset.kind !== ref!.kind) return whole;
    return asset.kind === 'style'
      ? `<style>${escapeStyleBody(asset.css)}</style>`
      : replaceSrc(whole, asset.dataUri);
  });
}

const CSP_META = `<meta http-equiv="Content-Security-Policy" content="${HTML_CSP}">`;

/**
 * Wrap a document for `srcdoc` inside a head that carries the CSP.
 *
 * The wrap is unconditional: only the parser can say where a real <head> is, so
 * searching the source for one lets a `<head>` written in a comment, an
 * attribute value, or script text take the policy somewhere it is never an
 * element, and a document that opens with content leaves it outside <head>,
 * where a CSP <meta> is ignored. Wrapping puts the policy first in a head the
 * parser built. The document's own <html>/<head>/<body> start tags are then
 * ignored, and its <style>/<link> still apply from the body.
 */
export function buildSrcdoc(html: string): string {
  return `<!doctype html><html><head>${CSP_META}</head><body>${html}</body></html>`;
}

export interface AssetReaders {
  /** Workspace-relative path -> text, for a stylesheet. */
  readText(path: string): Promise<string>;
  /** Workspace-relative path -> a `data:` URI, for an image. */
  readDataUri(path: string): Promise<string>;
}

/**
 * Read back the same-workspace assets a document references, ready for
 * `inlineAssets`. Anything that does not resolve to a workspace path is skipped
 * (never fetched), and an asset that fails to read is simply omitted - the CSP
 * blocks the original reference, so a missing file degrades to a missing image
 * rather than an error page.
 */
export async function loadInlineAssets(
  html: string,
  docPath: string,
  readers: AssetReaders,
  /** A generated report references a handful of assets; past this it is a
   *  runaway document, not a page. */
  limit = 40,
): Promise<Map<string, InlinedAsset>> {
  const wanted = collectAssetRefs(html)
    .map((ref) => ({ ref, path: resolveWorkspaceAsset(docPath, ref.href) }))
    .filter((a): a is { ref: AssetRef; path: string } => a.path !== null)
    .slice(0, limit);

  const loaded = await Promise.all(
    wanted.map(async ({ ref, path }): Promise<[string, InlinedAsset] | null> => {
      try {
        return ref.kind === 'style'
          ? [ref.href, { kind: 'style', css: await readers.readText(path) }]
          : [ref.href, { kind: 'image', dataUri: await readers.readDataUri(path) }];
      } catch {
        return null;
      }
    }),
  );
  return new Map(loaded.filter((e): e is [string, InlinedAsset] => e !== null));
}
