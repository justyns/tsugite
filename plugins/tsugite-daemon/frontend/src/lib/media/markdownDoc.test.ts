import { describe, expect, test } from 'vitest';
import { markdownDoc } from './markdownDoc';

describe('markdownDoc', () => {
  test('puts the stylesheet ahead of the content', () => {
    const doc = markdownDoc('<h1>Report</h1>');

    expect(doc.startsWith('<style>')).toBe(true);
    expect(doc.indexOf('</style>')).toBeLessThan(doc.indexOf('<h1>Report</h1>'));
    expect(doc).toContain('<h1>Report</h1>');
  });

  test('styles what renderMarkdown emits', () => {
    const doc = markdownDoc('');

    for (const selector of ['h1', 'pre', 'table', 'blockquote', 'img', '.tsu-fm', '.wikilink']) {
      expect(doc).toContain(selector);
    }
  });
});
