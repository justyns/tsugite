const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { isShellFile, originOf, tokenFor } = require('../src/urls');

const shell = (f) => pathToFileURL(path.join(__dirname, '..', 'src', f)).href;

test('only the two shell pages are shell files', () => {
  assert.ok(isShellFile(shell('error.html')));
  assert.ok(isShellFile(`${shell('remote.html')}?x=1#y`));
  assert.equal(isShellFile(shell('main.js')), false);
  assert.equal(isShellFile(shell('error.html.evil')), false);
  assert.equal(isShellFile('https://example.com/error.html'), false);
});

test('malformed urls are denied instead of throwing', () => {
  assert.equal(isShellFile('not a url'), false);
  assert.equal(isShellFile(undefined), false);
  assert.equal(originOf('not a url'), null);
  assert.equal(originOf(undefined), null);
  assert.equal(originOf('http://127.0.0.1:9/x'), 'http://127.0.0.1:9');
});

test('tokenFor gives the token only to a page on the target origin', () => {
  const target = { origin: 'http://127.0.0.1:8374', token: 'tok' };
  assert.equal(tokenFor(target, 'http://127.0.0.1:8374/chat?x=1'), 'tok');
  for (const sender of ['http://127.0.0.1:8375/', 'https://127.0.0.1:8374/', shell('error.html'), 'about:blank', undefined]) {
    assert.equal(tokenFor(target, sender), '', String(sender));
  }
  assert.equal(tokenFor({ origin: '', token: 'tok' }, 'about:blank'), '');
  assert.equal(tokenFor({ origin: '', token: 'tok' }, undefined), '');
});
