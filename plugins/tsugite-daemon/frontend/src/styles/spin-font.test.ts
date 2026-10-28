import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const src = fileURLToPath(new URL('..', import.meta.url));
const read = (rel: string) => readFileSync(`${src}${rel}`, 'utf8');

describe('spinner Braille font', () => {
  it('declares a Braille-only @font-face backed by a bundled woff2 and its license', () => {
    const face =
      /@font-face\s*{[^}]*'Tsugite Braille'[^}]*}/.exec(read('styles/app.css'))?.[0] ?? '';
    expect(face).toMatch(/unicode-range:\s*U\+2800-28FF\s*;/);
    const file = /url\('\.\.\/([^']+\.woff2)'\)/.exec(face)?.[1] ?? '';
    expect(file).toMatch(/\.woff2$/);
    expect(readFileSync(`${src}${file}`).subarray(0, 4).toString()).toBe('wOF2');
    expect(existsSync(`${src}${file.replace('.woff2', '.LICENSE.txt')}`)).toBe(true);
  });

  it('lists the Braille font first in the spinner font stack', () => {
    expect(read('styles/tokens.css')).toMatch(/--font-spin:\s*'Tsugite Braille',/);
  });

  it('uses the spinner font stack in every .t-spin rule', () => {
    const files = (readdirSync(src, { recursive: true }) as string[]).filter((f) =>
      f.endsWith('.svelte'),
    );
    const rules = files.flatMap((f) =>
      [...read(f).matchAll(/\.t-spin\s*{([^}]*)}/g)]
        .map((m) => m[1]!)
        .filter((body) => body.includes('font-family'))
        .map((body) => ({ f, body })),
    );
    expect(rules.length).toBeGreaterThan(0);
    for (const { f, body } of rules) {
      expect(body, f).toContain('font-family: var(--font-spin)');
    }
  });
});
