<script lang="ts">
  // Full-view host for a plugin surface that asked for a nav-rail entry. The nav
  // view id and the surface kind are the same `plugin/<name>/<kind>` string, so
  // this reads both off the route and hands the hash query through as the
  // surface's params - which makes #plugin/<name>/<kind>?path=a.md a deep link.
  import { router } from '$lib/router.svelte';
  import PluginSurface from './PluginSurface.svelte';
</script>

<PluginSurface kind={router.view} params={router.params} />
