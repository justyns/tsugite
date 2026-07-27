/**
 * Server-provided client config, read from the public /api/health payload.
 * Currently just the image re-encode knobs (see imageEncode.ts). Memoized for
 * the session — health is set at daemon start and doesn't hot-reload — and
 * falls back to defaults when health is unreachable.
 */
import { api } from './client';
import { DEFAULT_IMAGE_CONFIG, type ImageEncodeConfig } from '$lib/media/imageEncode';

interface HealthImages {
  images?: { max_edge?: number; quality?: number };
}

let cached: Promise<ImageEncodeConfig> | null = null;

export function loadImageConfig(): Promise<ImageEncodeConfig> {
  if (!cached) {
    cached = api
      .get<HealthImages>('/api/health')
      .then((h) => ({
        maxEdge: h.images?.max_edge ?? DEFAULT_IMAGE_CONFIG.maxEdge,
        quality: h.images?.quality ?? DEFAULT_IMAGE_CONFIG.quality,
      }))
      .catch(() => DEFAULT_IMAGE_CONFIG);
  }
  return cached;
}
