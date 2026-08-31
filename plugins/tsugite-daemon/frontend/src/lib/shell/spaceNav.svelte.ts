/**
 * Per-space nav selection: a space records the view it is on and returns there
 * when it becomes active again. Restoring goes through the hash router rather
 * than the shell store, so the URL and the store stay in agreement (App's route
 * effect does the store update). Returns a stop handle for its effect root.
 */
import { untrack } from 'svelte';
import { navigate, router } from '$lib/router.svelte';
import { spaces } from '$lib/stores/spaces.svelte';
import { views } from '../../views';

export function followSpaceNav(): () => void {
  return $effect.root(() => {
    $effect(() => {
      const view = router.view;
      untrack(() => {
        if (view) spaces.setNav(spaces.activeSpaceId, view);
      });
    });

    // Every path that swaps the active space - the switcher, a new or closed
    // space, the palette - lands here, so none of them need their own wiring.
    let current = spaces.activeSpaceId;
    $effect(() => {
      const id = spaces.activeSpaceId;
      untrack(() => {
        if (id === current) return;
        current = id;
        navigate(spaces.active.nav ?? views[0]!.id);
      });
    });
  });
}
