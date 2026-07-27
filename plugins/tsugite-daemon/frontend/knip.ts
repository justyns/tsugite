import type { KnipConfig } from 'knip';

// knip does not parse Svelte SFCs, so imports made only from a .svelte file
// would otherwise read as unused. This compiler emits each component's imports
// as plain statements for knip to resolve (knip's documented Svelte pattern).
// Two passes, each non-greedy and anchored on the specifier's closing quote so a
// single import becomes one valid statement (a greedy `import[^;]+` merges the
// dynamic imports in a Promise.all into unparseable junk and blinds knip to the
// whole file, e.g. TerminalCanvas's xterm/TermPill usage):
//   1. static/side-effect imports, only at statement position, so import-shaped
//      strings in gallery demo content aren't read as real (and reported as
//      unresolved) imports;
//   2. dynamic import() calls anywhere, e.g. `await import('x')`.
const svelteImports = (text: string) =>
  [
    ...[...text.matchAll(/^\s*import\b[^'"]*?['"][^'"]+['"]/gm)].map((m) => m[0].trim()),
    ...[...text.matchAll(/import\s*\(\s*['"][^'"]+['"]\s*\)/g)].map((m) => m[0]),
  ]
    .map((s) => `${s};`)
    .join('\n');

const config: KnipConfig = {
  entry: [
    // Dev-only component showcase: *.gallery.svelte demos are auto-discovered at
    // runtime via import.meta.glob (src/views/gallery/View.svelte), which knip
    // does not follow - list them as entries so their imports count as used.
    'src/**/*.gallery.svelte',
    // Vitest specs (unit + browser projects) - their imports are real usage.
    'src/**/*.test.ts',
    // Service worker: registered by the '/sw.js' string in lib/sw-register.ts,
    // never statically imported.
    'public/sw.js',
  ],
  project: ['src/**/*.{ts,svelte}'],
  paths: {
    // Mirrors the tsconfig / vite `$lib` alias.
    '$lib/*': ['./src/lib/*'],
  },
  compilers: { svelte: svelteImports },
  // Stores follow a frozen `export class X` + `export const x = new X()` pattern
  // (AGENTS.md Web UI): the class is referenced only by its own instance, and
  // cohesive type modules (e.g. the turns timeline vocabulary) reference their
  // members internally. Treat any export used within its own file as live so the
  // report means "truly unreachable", not "could drop the export keyword".
  ignoreExportsUsedInFile: true,
};

export default config;
