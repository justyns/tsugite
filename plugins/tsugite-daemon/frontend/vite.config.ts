import { svelte } from '@sveltejs/vite-plugin-svelte';
import { playwright } from '@vitest/browser-playwright';
import { configDefaults, defineConfig } from 'vitest/config';

// base '/static/' in prod: Starlette mounts the built dist under /static.
// dev serves from '/' and proxies /api to a local daemon.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/static/' : '/',
  plugins: [svelte()],
  resolve: {
    alias: { $lib: '/src/lib' },
  },
  build: {
    outDir: '../tsugite_daemon/web',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        // Peel the stable third-party runtime (svelte, marked) into its own chunk:
        // it changes rarely, so app updates don't re-download it, and the app chunk
        // stays under the 500 kB warning. xterm is already a lazy chunk via its
        // dynamic import in the terminal view, so exclude it here to keep it lazy.
        manualChunks: (id: string) =>
          id.includes('node_modules') && !id.includes('@xterm') ? 'vendor' : undefined,
      },
    },
  },
  server: {
    proxy: {
      // TSU_API lets tooling point the dev server at an isolated daemon.
      '/api': { target: process.env.TSU_API ?? 'http://127.0.0.1:8374' },
    },
  },
  test: {
    // Two projects: pure logic runs fast in node; component interaction tests
    // (*.svelte.test.ts) run in a real headless chromium via Playwright.
    projects: [
      {
        extends: true,
        test: {
          name: 'unit',
          environment: 'node',
          include: ['src/**/*.test.ts'],
          exclude: [...configDefaults.exclude, 'src/**/*.svelte.test.ts'],
        },
      },
      {
        extends: true,
        // Tests must NEVER reach a real daemon: components fire incidental
        // fetches (slash commands, effort levels, session settings) that would
        // otherwise ride the inherited dev proxy into whatever daemon listens
        // on the default port - a production instance's logs filled with
        // tokenless 401s from test fixtures. Port 9 is a connect-only black
        // hole, chosen BECAUSE it's privileged: nothing here ever binds it, and
        // no rootless process can accidentally claim it either, so the connect
        // is guaranteed refused and fetches fail fast into the components'
        // catch paths. (An unprivileged port could be transiently occupied by
        // some other local service, which would silently receive test traffic.)
        server: {
          proxy: {
            '/api': { target: 'http://127.0.0.1:9' },
          },
        },
        test: {
          name: 'browser',
          include: ['src/**/*.svelte.test.ts'],
          // Serial file execution: the browser runner's RPC channel emits
          // phantom "Unknown event: response:*:ready" unhandled errors under
          // parallel multi-file load (tests all pass; exit code goes 1).
          fileParallelism: false,
          browser: {
            enabled: true,
            provider: playwright(),
            headless: true,
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
}));
