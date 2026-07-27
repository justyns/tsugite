import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-sans/700.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import '@fontsource/jetbrains-mono/600.css';
import '@fontsource/jetbrains-mono/700.css';
import '@fontsource/jetbrains-mono/400-italic.css';
import './styles/tokens.css';
import './styles/app.css';

import { mount } from 'svelte';
import App from './App.svelte';

const target = document.getElementById('app');
if (!target) throw new Error('#app mount target missing');
const app = mount(App, { target });

// A hashed chunk that 404s after a deploy (new build, no-cache SW) - hard reload
// to pick up the fresh index + asset graph.
window.addEventListener('vite:preloadError', () => location.reload());

// sw.js is push-only and useless in dev (Vite has no /sw.js); register only in
// prod builds.
if (!import.meta.env.DEV) {
  void import('$lib/sw-register').then((m) => m.registerSW());
}

export default app;
