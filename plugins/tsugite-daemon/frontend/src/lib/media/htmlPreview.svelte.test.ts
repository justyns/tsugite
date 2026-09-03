/// <reference types="@vitest/browser/context" />
// Runs in the browser project (the `.svelte.test.ts` glob) because where the CSP
// <meta> lands is a parser question, and the node project has no DOMParser.
import { describe, expect, test } from 'vitest';
import { HTML_CSP, buildSrcdoc } from './htmlPreview';

const EVIL = 'http://127.0.0.1:9/evil.png';

/** Documents that put a `<head>` where the parser does not see one, plus a
 *  well-formed control. A CSP <meta> that is not the first element of a real
 *  <head> leaves the frame with no policy, and `sandbox=""` does not stop
 *  subresource loads. */
const DOCS: [label: string, html: string][] = [
  [
    'a well-formed document',
    `<html><head><title>t</title></head><body><img src="${EVIL}"></body></html>`,
  ],
  ['a <head> inside a comment', `<!-- <head> --><html><body><img src="${EVIL}"></body></html>`],
  ['content before the <head> tag', `<img src="${EVIL}"><head><title>t</title></head><body>x`],
  [
    'a <head> inside an attribute value',
    `<html><body><div data-x="<head>">z</div><img src="${EVIL}"></body></html>`,
  ],
  [
    'a <head> inside script text',
    `<html><body><script>var s = "<head>";</script><img src="${EVIL}"></body></html>`,
  ],
];

function parse(html: string): Document {
  return new DOMParser().parseFromString(buildSrcdoc(html), 'text/html');
}

describe('buildSrcdoc CSP placement', () => {
  for (const [label, html] of DOCS) {
    test(`the policy is the first element of <head> for ${label}`, () => {
      const doc = parse(html);
      const meta = doc.querySelector('meta[http-equiv="Content-Security-Policy" i]');

      expect(meta, 'the policy did not parse as an element').not.toBeNull();
      expect(meta!.getAttribute('content')).toBe(HTML_CSP);
      expect(meta!.parentElement, 'the policy landed outside <head>').toBe(doc.head);
      expect(doc.head.firstElementChild).toBe(meta);
    });
  }

  test('the document content survives the wrap', () => {
    const doc = parse('<html><body><h1>report</h1><p>hi</p></body></html>');
    expect(doc.querySelector('h1')!.textContent).toBe('report');
    expect(doc.querySelector('p')!.textContent).toBe('hi');
  });

  test('a <style> the document carries survives, in the body', () => {
    const doc = parse('<head><style>h1{font-weight:700}</style></head><h1>report</h1>');
    expect(doc.querySelector('style')).not.toBeNull();
    expect(doc.querySelector('h1')!.textContent).toBe('report');
  });
});
