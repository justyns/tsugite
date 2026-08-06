/**
 * Tiny hash router. `#view?a=1&b=2` → { view, params }. Pure parse/serialize
 * (parseHash/buildHash) are unit-tested; the reactive `router` $state object
 * mirrors the current hash, mutated in place - never a reassigned binding.
 * Window access is deferred to initRouter() so the pure parts import cleanly
 * under node (vitest).
 */

export type RouteParams = Record<string, string>;

export interface Route {
  view: string;
  params: RouteParams;
}

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, '');
  const q = raw.indexOf('?');
  const view = q === -1 ? raw : raw.slice(0, q);
  const query = q === -1 ? '' : raw.slice(q + 1);
  const params: RouteParams = {};
  new URLSearchParams(query).forEach((value, key) => {
    params[key] = value;
  });
  return { view, params };
}

export function buildHash(view: string, params?: RouteParams): string {
  const query = new URLSearchParams(params ?? {}).toString();
  return query ? `#${view}?${query}` : `#${view}`;
}

export const router = $state<Route>({ view: '', params: {} });

/** The route we were on before the current one, for back-affordance decisions
 *  (e.g. the phone chats drilldown pops to a preceding list rather than pushing). */
export const routeHistory = $state<{ prev: Route | null }>({ prev: null });

export function navigate(view: string, params?: RouteParams): void {
  location.hash = buildHash(view, params);
}

/** navigate() without a history entry - for params that track a live control
 *  (a search box), where every keystroke would otherwise become a Back step. */
export function replaceRoute(view: string, params?: RouteParams): void {
  location.replace(buildHash(view, params));
}

export function initRouter(): void {
  const sync = () => {
    const next = parseHash(location.hash);
    routeHistory.prev = { view: router.view, params: router.params };
    router.view = next.view;
    router.params = next.params;
  };
  sync();
  window.addEventListener('hashchange', sync);
}
