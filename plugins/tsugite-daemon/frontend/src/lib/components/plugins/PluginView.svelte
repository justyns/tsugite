<script lang="ts">
  // Full-view host for a plugin surface with a nav-rail entry. The nav view id
  // and the surface kind are one string, so #plugin/<name>/<kind>?path=a.md
  // deep-links here.
  import { router } from '$lib/router.svelte';
  import PluginSurface from './PluginSurface.svelte';
</script>

<PluginSurface kind={router.view} params={router.params} />
