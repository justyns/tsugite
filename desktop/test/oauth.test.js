const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { forwardCallback, loopbackCallback, watchSignin } = require('../src/oauth');

const CALLBACK = 'http://127.0.0.1/api/plugins/mcp/callback';

test('loopbackCallback gives the query of the registered redirect and null for anything else', () => {
  assert.equal(loopbackCallback(`${CALLBACK}?code=c&state=s`), '?code=c&state=s');
  assert.equal(loopbackCallback(`http://127.0.0.1:8080/api/plugins/mcp/callback?code=c`), '?code=c');
  for (const url of [
    'https://127.0.0.1/api/plugins/mcp/callback?code=c',
    'http://localhost/api/plugins/mcp/callback?code=c',
    'http://127.0.0.1/other?code=c',
    'https://example.test/authorize?state=s',
    'not a url',
  ]) {
    assert.equal(loopbackCallback(url), null, url);
  }
});

test('watchSignin stops a navigation to the loopback redirect and forwards its query', () => {
  for (const event of ['will-navigate', 'will-redirect']) {
    const contents = new EventEmitter();
    const forwarded = [];
    watchSignin(contents, (search) => forwarded.push(search));

    let prevented = 0;
    contents.emit(event, { url: 'https://example.test/consent', preventDefault: () => prevented++ });
    assert.deepEqual([prevented, forwarded], [0, []], event);
    contents.emit(event, { url: `${CALLBACK}?code=c&state=s`, preventDefault: () => prevented++ });
    assert.deepEqual([prevented, forwarded], [1, ['?code=c&state=s']], event);
  }
});

test('forwardCallback calls the daemon callback and throws when it rejects', async () => {
  const calls = [];
  const reply = (status) => async (url) => (calls.push(url), { ok: status < 400, status });

  await forwardCallback('http://daemon:8374', '?code=c&state=s', reply(200));
  assert.deepEqual(calls, ['http://daemon:8374/api/plugins/mcp/callback?code=c&state=s']);
  await assert.rejects(forwardCallback('http://daemon:8374', '?state=x', reply(400)), /rejected the sign-in \(400\)/);
});
