/**
 * Stale-while-revalidate cache: paint the last-known list from localStorage on a
 * cold load, then let the store refetch and re-cache. The old Alpine UI used the
 * same pattern under keys like `tsugite_sessions_${agent}`; this keeps the wire
 * key scheme but wraps the payload in a versioned envelope so a schema bump
 * self-invalidates instead of hydrating a stale shape.
 *
 * The parse/serialize halves are pure (node-unit-tested); readSwr/writeSwr are
 * the thin localStorage-guarded IO wrappers.
 */
import { readLocal, writeLocal } from '$lib/storage';

interface CacheEnvelope<T> {
  v: number;
  t: number;
  data: T;
}

export function serializeCache<T>(data: T, version: number, now: number = Date.now()): string {
  const envelope: CacheEnvelope<T> = { v: version, t: now, data };
  return JSON.stringify(envelope);
}

/** Returns the cached payload, or null when the entry is missing, corrupt, or
 *  from a superseded schema version. */
export function parseCache<T>(raw: string | null, version: number): T | null {
  if (!raw) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null) return null;
  const envelope = parsed as Partial<CacheEnvelope<T>>;
  if (envelope.v !== version) return null;
  return (envelope.data ?? null) as T | null;
}

export function readSwr<T>(key: string, version = 1): T | null {
  return parseCache<T>(readLocal(key), version);
}

export function writeSwr<T>(key: string, data: T, version = 1): void {
  writeLocal(key, serializeCache(data, version));
}
