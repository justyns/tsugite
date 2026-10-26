const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { Settings } = require('../src/settings');
const { tmp } = require('./helpers');

const reverse = (b) => Buffer.from(Buffer.from(b).reverse());
function storage({ available = true, prefix = 'v11', reEncrypt = false } = {}) {
  const calls = { encrypt: 0 };
  return {
    calls,
    isAsyncEncryptionAvailable: async () => available,
    encryptStringAsync: async (s) => {
      calls.encrypt++;
      return Buffer.concat([Buffer.from(prefix, 'latin1'), reverse(s)]);
    },
    decryptStringAsync: async (b) => ({ result: reverse(b.subarray(3)).toString(), shouldReEncrypt: reEncrypt }),
  };
}

test('remote token is stored encrypted, never in settings.json', async () => {
  const dir = tmp();
  const s = new Settings(dir, storage(), 'linux');
  await s.setRemote('https://tsugite.example.com/some/path', 'tsu_secret');
  assert.equal(s.data.mode, 'remote');
  assert.equal(s.data.remoteUrl, 'https://tsugite.example.com');
  assert.ok(!fs.readFileSync(s.file, 'utf8').includes('tsu_secret'));
  assert.ok(!fs.readFileSync(s.remoteTokenFile).includes('tsu_secret'));
  assert.equal(await new Settings(dir, storage(), 'linux').remoteToken(), 'tsu_secret');
});

test('remote mode refuses to store a token without OS encryption', async () => {
  const s = new Settings(tmp(), storage({ available: false }), 'linux');
  await assert.rejects(s.setRemote('https://x.example', 't'), /unavailable/);
  assert.equal(s.data.mode, 'local');
  assert.equal(fs.existsSync(s.remoteTokenFile), false);
});

test('on Linux a token encrypted with the built-in fallback key is refused', async () => {
  const s = new Settings(tmp(), storage({ prefix: 'v10' }), 'linux');
  await assert.rejects(s.setRemote('https://x.example', 't'), /No Secret Service/);
  assert.equal(s.data.mode, 'local');
  assert.equal(fs.existsSync(s.remoteTokenFile), false);
});

test('on macOS the v10 prefix is accepted', async () => {
  const s = new Settings(tmp(), storage({ prefix: 'v10' }), 'darwin');
  await s.setRemote('https://x.example', 't');
  assert.equal(s.data.mode, 'remote');
});

test('a token the OS asks to re-encrypt is written back', async () => {
  const dir = tmp();
  await new Settings(dir, storage(), 'linux').setRemote('https://x.example', 'tsu_secret');
  const st = storage({ reEncrypt: true });
  const s = new Settings(dir, st, 'linux');
  assert.equal(await s.remoteToken(), 'tsu_secret');
  assert.equal(st.calls.encrypt, 1);
});

test('storing a remote token tightens an existing token file to 0600', async () => {
  const dir = tmp();
  const s = new Settings(dir, storage(), 'linux');
  fs.writeFileSync(s.remoteTokenFile, 'old', { mode: 0o644 });
  fs.chmodSync(s.remoteTokenFile, 0o644);
  await s.setRemote('https://tsugite.example.com', 'tsu_secret');
  assert.equal(fs.statSync(s.remoteTokenFile).mode & 0o777, 0o600);
});

test('hasRemote needs both a saved URL and a token file', async () => {
  const dir = tmp();
  const s = new Settings(dir, storage(), 'linux');
  assert.equal(s.hasRemote(), false);
  await s.setRemote('https://x.example', 't');
  assert.equal(s.hasRemote(), true);
  s.update({ mode: 'local' });
  assert.equal(new Settings(dir, storage(), 'linux').hasRemote(), true);
  fs.rmSync(s.remoteTokenFile);
  assert.equal(s.hasRemote(), false);
});
test('plain http is accepted only for loopback hosts', async () => {
  for (const url of ['http://localhost:8374', 'http://127.0.0.1:8374', 'http://127.4.5.6', 'http://[::1]:9']) {
    const s = new Settings(tmp(), storage(), 'linux');
    await s.setRemote(url, 't');
    assert.equal(s.data.remoteUrl, new URL(url).origin);
  }
  for (const url of ['http://192.168.1.5:8374', 'http://tsugite.example.com', 'http://127.0.0.1.example.com']) {
    const s = new Settings(tmp(), storage(), 'linux');
    await assert.rejects(s.setRemote(url, 't'), /https:\/\//);
    assert.equal(fs.existsSync(s.remoteTokenFile), false);
  }
});

test('a write that dies midway leaves the previous settings.json readable', () => {
  const dir = tmp();
  new Settings(dir, storage(), 'linux').update({ mode: 'remote' });
  const s = new Settings(dir, storage(), 'linux');
  const write = fs.writeFileSync;
  fs.writeFileSync = (file, data, ...rest) => {
    write(file, String(data).slice(0, 5), ...rest);
    throw new Error('disk died');
  };
  try {
    assert.throws(() => s.update({ remoteUrl: 'https://x.example' }), /disk died/);
  } finally {
    fs.writeFileSync = write;
  }
  assert.equal(new Settings(dir, storage(), 'linux').data.mode, 'remote');
});

test('the local token is stored encrypted in local-token.bin with mode 0600', async () => {
  const dir = tmp();
  const s = new Settings(dir, storage(), 'linux');
  assert.equal(await s.localTokenStore().read(), '');
  await s.localTokenStore().write('tsu_local');
  assert.equal(fs.statSync(s.localTokenFile).mode & 0o777, 0o600);
  assert.ok(!fs.readFileSync(s.localTokenFile).includes('tsu_local'));
  assert.equal(await new Settings(dir, storage(), 'linux').localTokenStore().read(), 'tsu_local');
});

test('a local token the OS asks to re-encrypt is written back', async () => {
  const dir = tmp();
  await new Settings(dir, storage(), 'linux').localTokenStore().write('tsu_local');
  const st = storage({ reEncrypt: true });
  assert.equal(await new Settings(dir, st, 'linux').localTokenStore().read(), 'tsu_local');
  assert.equal(st.calls.encrypt, 1);
});

for (const [why, st] of [['unavailable', storage({ available: false })], ['the Linux fallback key', storage({ prefix: 'v10' })]]) {
  test(`with ${why} the local token stays in memory for the run`, async () => {
    const s = new Settings(tmp(), st, 'linux');
    const notes = [];
    const store = s.localTokenStore((msg) => notes.push(msg));
    await store.write('tsu_local');
    assert.equal(await store.read(), 'tsu_local');
    assert.equal(fs.existsSync(s.localTokenFile), false);
    assert.equal(notes.length, 1);
  });
}

test('an old plaintext local-token file is removed', () => {
  const dir = tmp();
  fs.writeFileSync(path.join(dir, 'local-token'), 'tsu_plain');
  new Settings(dir, storage(), 'linux');
  assert.equal(fs.existsSync(path.join(dir, 'local-token')), false);
});
