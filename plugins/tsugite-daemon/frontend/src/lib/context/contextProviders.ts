/**
 * Client-context provider registry: the extension point behind the composer's
 * "add context" menu. Each provider captures a piece of device/browser state the
 * daemon can't see for itself (the browser is the only thing that knows where you
 * are, what timezone you're in, ...) and hands it back as a structured
 * {key,label,value} item. Those items ride the send as `context_metadata` (never
 * folded into the message text) and render in the conversation's context gutter.
 *
 * Adding a provider is one entry in `contextProviders`: give it a stable `key`
 * (also the context_metadata key and the chip/testid suffix), a `label`, an
 * `icon` (a name from $lib/components/icon), a `capture()` that never throws, and
 * - if it wants a Settings auto-attach toggle - an `autoAttachStoreKey` (the
 * localStorage key its AutoAttachStore persists under).
 */
import { getCurrentLocation, formatGeoFix } from '$lib/media/geolocation';
import type { IconName } from '$lib/components/icon/icons';

/** capture() outcome: a captured value, or a user-facing failure. `code` lets the
 *  caller pick a toast tone (e.g. a denied permission warns rather than errors). */
export type ContextCapture = { value: string } | { error: { code?: string; message: string } };

/** One structured context item attached to a message. */
export interface ContextItem {
  key: string;
  label: string;
  value: string;
}

export interface ContextProvider {
  /** Stable id: the context_metadata `key`, the chip testid suffix, the dedupe key. */
  key: string;
  /** Human label for the menu row, the chip, and the rendered gutter row. */
  label: string;
  /** Icon name ($lib/components/icon) for the menu row and the chip. */
  icon: IconName;
  /** Capture the current value best-effort; resolves an error rather than throwing. */
  capture(): Promise<ContextCapture>;
  /** localStorage key of this provider's auto-attach toggle, when it offers one. */
  autoAttachStoreKey?: string;
}

const location: ContextProvider = {
  key: 'location',
  label: 'Location',
  icon: 'pin',
  autoAttachStoreKey: 'tsugite_geo_autoattach',
  async capture(): Promise<ContextCapture> {
    const res = await getCurrentLocation();
    return res.ok
      ? { value: formatGeoFix(res.fix) }
      : { error: { code: res.error.code, message: res.error.message } };
  },
};

export const contextProviders: ContextProvider[] = [location];

export function contextProvider(key: string): ContextProvider | undefined {
  return contextProviders.find((p) => p.key === key);
}
