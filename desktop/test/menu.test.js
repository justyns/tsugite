const test = require('node:test');
const assert = require('node:assert/strict');
const { daemonItems } = require('../src/menu');

const build = (o) => {
  const used = [];
  const items = daemonItems({ mode: 'local', remoteUrl: '', hasRemote: false, onUse: (m) => used.push(m), onConnect: () => used.push('connect'), ...o });
  return { items, used };
};

test('without a saved remote only the local daemon can be picked', () => {
  const { items } = build();
  assert.deepEqual(items.map((i) => i.label), ['Local Daemon', 'Remote Daemon', 'Connect to Remote Daemon...']);
  assert.equal(items[0].checked, true);
  assert.equal(items[1].enabled, false);
});

test('a saved remote shows its host and switches without the form', () => {
  const { items, used } = build({ mode: 'remote', remoteUrl: 'https://tsugite.example.com', hasRemote: true });
  assert.equal(items[1].label, 'Remote Daemon (tsugite.example.com)');
  assert.equal(items[1].checked, true);
  assert.equal(items[0].checked, false);
  assert.equal(items[2].label, 'Change Remote Daemon...');
  items[0].click();
  items[1].click();
  items[2].click();
  assert.deepEqual(used, ['local', 'remote', 'connect']);
});
