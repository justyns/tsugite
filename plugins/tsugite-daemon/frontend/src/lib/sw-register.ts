/**
 * Service-worker registration + update flow, ported from app.js. sw.js stays
 * push-only (no precaching) - this only surfaces "a new version is waiting" via
 * onSwUpdateAvailable() and applies it on demand (SKIP_WAITING → controllerchange
 * → reload). Wiring the callback to reactive UI happens at the call site.
 */

let applyUpdateImpl: (() => void) | null = null;
let onUpdate: (() => void) | null = null;

/** Register a callback fired when a newly-installed SW is waiting to activate. */
export function onSwUpdateAvailable(cb: () => void): void {
  onUpdate = cb;
}

/** Activate a waiting update (or reload if there's nothing to skip to). */
export function applyUpdate(): void {
  if (applyUpdateImpl) applyUpdateImpl();
  else location.reload();
}

export function registerSW(): void {
  if (!('serviceWorker' in navigator)) return;
  navigator.serviceWorker
    .register('/sw.js')
    .then((reg) => {
      const notifyIfWaiting = (worker: ServiceWorker | null) => {
        if (worker && worker.state === 'installed' && navigator.serviceWorker.controller) {
          onUpdate?.();
        }
      };
      notifyIfWaiting(reg.waiting);
      reg.addEventListener('updatefound', () => {
        const nw = reg.installing;
        if (!nw) return;
        nw.addEventListener('statechange', () => notifyIfWaiting(nw));
      });

      let reloading = false;
      // controllerchange also fires when the FIRST SW claims a previously
      // uncontrolled page (every fresh profile's first visit) - that reload
      // killed in-flight chat streams. Only reload when an existing controller
      // is REPLACED, i.e. a real update.
      let hadController = Boolean(navigator.serviceWorker.controller);
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (!hadController) {
          hadController = true;
          return;
        }
        if (reloading) return;
        reloading = true;
        location.reload();
      });

      applyUpdateImpl = () => {
        const worker = reg.waiting || reg.installing;
        if (worker) worker.postMessage({ type: 'SKIP_WAITING' });
        else location.reload();
      };
    })
    .catch(() => {});
}
