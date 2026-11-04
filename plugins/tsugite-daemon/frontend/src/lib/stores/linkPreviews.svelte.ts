/**
 * Chat link previews from GET /api/link-preview. `lookup` runs inside a markdown
 * parse and never blocks. It returns what is loaded and fetches the rest. A link
 * whose fetch failed stays plain until reload, except that a timeout is asked about
 * once more after RETRY_MS.
 */
import { api, type ApiError } from '$lib/api/client';
import { pluginsMeta } from './pluginsMeta.svelte';

/** Milliseconds a preview is served before the next lookup refreshes it. */
const STALE_MS = 60_000;
/** Milliseconds before a link whose lookup timed out is asked about again. */
export const RETRY_MS = 5_000;

/** A resolver as GET /api/plugins lists it under `link_resolvers`. */
export interface LinkResolverPattern {
  plugin: string;
  name: string;
  hosts: string[];
  path: string | null;
}

interface CompiledResolver {
  hosts: string[];
  path: RegExp | null;
}

// Mirrors resolver matching in tsugite/link_previews.py. Change both together.

/** Drops a resolver whose path JavaScript cannot compile. */
export function compilePatterns(patterns: LinkResolverPattern[]): CompiledResolver[] {
  return patterns.flatMap((p) => {
    try {
      return [{ hosts: p.hosts, path: p.path === null ? null : new RegExp(p.path) }];
    } catch (err) {
      console.warn(`link resolver ${p.plugin}:${p.name} has a path JavaScript cannot compile`, err);
      return [];
    }
  });
}

export function matchResolver(compiled: CompiledResolver[], href: string): boolean {
  let url: URL;
  try {
    url = new URL(href);
  } catch {
    return false;
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;
  return compiled.some(
    (r) => r.hosts.includes(url.hostname) && (r.path?.test(url.pathname) ?? true),
  );
}

export interface LinkBadge {
  text: string;
  tone: 'ok' | 'warn' | 'err' | 'info' | 'mute';
}

/** A preview as the daemon normalizes it. `status` sets the chip's glyph and aria-label. */
export interface LinkPreview {
  label: string;
  title: string;
  url: string;
  status: LinkBadge | null;
  badges: LinkBadge[];
  fields: { label: string; value: string }[];
  /** A path under the resolver plugin's own UI assets, shown as a monochrome mask. */
  icon: string | null;
}

export class LinkPreviewStore {
  private readonly patterns: () => LinkResolverPattern[];
  private readonly clock: () => number;
  private settled = $state(0);
  private compiled: CompiledResolver[];
  private loaded = new Map<string, { preview: LinkPreview; at: number }>();
  private failed = new Set<string>();
  private inflight = new Set<string>();
  private retried = new Set<string>();

  constructor(patterns: () => LinkResolverPattern[], clock: () => number = Date.now) {
    this.patterns = patterns;
    this.clock = clock;
    this.compiled = $derived(compilePatterns(this.patterns()));
  }

  /** Changes when a preview arrives, a timed-out link is due again, or the resolver
   *  patterns change. */
  get version(): number {
    void this.compiled;
    return this.settled;
  }

  lookup(href: string): LinkPreview | undefined {
    const hit = this.loaded.get(href);
    const stale = !hit || this.clock() - hit.at > STALE_MS;
    if (stale && !this.inflight.has(href) && !this.failed.has(href)) {
      if (matchResolver(this.compiled, href)) void this.fetch(href);
    }
    return hit?.preview;
  }

  private async fetch(href: string): Promise<void> {
    this.inflight.add(href);
    try {
      const preview = await api.get<LinkPreview>(
        `/api/link-preview?url=${encodeURIComponent(href)}`,
      );
      this.loaded.set(href, { preview, at: this.clock() });
      this.settled++;
    } catch (err) {
      this.failed.add(href);
      if ((err as ApiError).status === 504 && !this.retried.has(href)) {
        this.retried.add(href);
        setTimeout(() => {
          this.failed.delete(href);
          this.settled++;
        }, RETRY_MS);
      }
    } finally {
      this.inflight.delete(href);
    }
  }
}

export const linkPreviews = new LinkPreviewStore(() => pluginsMeta.linkResolvers);
