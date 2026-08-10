export interface MetaLink {
  key: string;
  href: string;
  label: string;
}

/** Link chips for a session's metadata: every value that is an http(s) URL, in key
 *  order. A bare string is labelled by its key (`"pr": "https://..."` -> `pr`); an
 *  agent that wants a better label writes `{"url": ..., "label": "PR #608"}`.
 *
 *  Metadata is agent-authored, so the scheme test keeps `javascript:` and `data:`
 *  out of the href. It runs on the trimmed value, which is also the value returned,
 *  so leading whitespace can't slip a scheme past it. */
export function metaLinks(metadata: Record<string, unknown> | undefined): MetaLink[] {
  const out: MetaLink[] = [];
  for (const [key, value] of Object.entries(metadata ?? {})) {
    const spec =
      typeof value === 'object' && value !== null
        ? (value as Record<string, unknown>)
        : { url: value };
    const href = typeof spec.url === 'string' ? spec.url.trim() : '';
    if (!/^https?:\/\//i.test(href)) continue;
    const label = typeof spec.label === 'string' ? spec.label.trim() : '';
    out.push({ key, href, label: label || key });
  }
  return out;
}
