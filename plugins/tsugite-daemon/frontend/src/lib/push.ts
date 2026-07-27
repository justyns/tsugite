/**
 * Web-push subscribe/unsubscribe, ported from app.js. VAPID key fetch +
 * subscription round-trip through /api/push/*.
 */
import { api } from '$lib/api/client';

function urlBase64ToUint8Array(base64: string): Uint8Array<ArrayBuffer> {
  const normalized = base64.replace(/-/g, '+').replace(/_/g, '/');
  const raw = atob(normalized);
  const out = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

export async function subscribePush(): Promise<void> {
  const reg = await navigator.serviceWorker.ready;
  const { public_key } = await api.get<{ public_key: string }>('/api/push/vapid-key');
  const sub = await reg.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: urlBase64ToUint8Array(public_key),
  });
  await api.post('/api/push/subscribe', sub.toJSON());
}

export async function unsubscribePush(): Promise<void> {
  const reg = await navigator.serviceWorker.ready;
  const sub = await reg.pushManager.getSubscription();
  if (!sub) return;
  const endpoint = sub.endpoint;
  await sub.unsubscribe();
  await api.post('/api/push/unsubscribe', { endpoint });
}

/**
 * Toggle a subscription with a hard 10s timeout so the UI can never hang on a
 * stalled serviceWorker.ready / pushManager.subscribe. Returns the new state;
 * throws on failure (the caller surfaces it).
 */
export async function togglePush(currentlySubscribed: boolean): Promise<boolean> {
  const op = currentlySubscribed ? unsubscribePush() : subscribePush();
  let timer: ReturnType<typeof setTimeout>;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(
      () => reject(new Error('timed out after 10s - check notification permission and SW state')),
      10000,
    );
  });
  try {
    await Promise.race([op, timeout]);
    return !currentlySubscribed;
  } finally {
    clearTimeout(timer!);
  }
}
