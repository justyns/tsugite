const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const net = require('node:net');
const path = require('node:path');
const { desktopDirs, ensureConfig, readPort, isPortFree } = require('../src/config');
const { tmp } = require('./helpers');

test('first run writes daemon.yaml with XDG data paths and creates the workspace', async () => {
  const dir = tmp();
  const { configPath, port, workspaceDir } = await ensureConfig({ userData: dir, home: dir, env: {} });
  const data = path.join(dir, '.local', 'share', 'tsugite');
  const text = fs.readFileSync(configPath, 'utf8');
  assert.ok(text.includes(`state_dir: ${JSON.stringify(path.join(data, 'desktop'))}`));
  assert.ok(text.includes(`default_workspace_dir: ${JSON.stringify(path.join(data, 'workspaces', 'desktop'))}`));
  assert.ok(text.includes('http:\n  enabled: true\n  host: 127.0.0.1\n'));
  assert.equal(readPort(configPath), port);
  assert.ok(port > 0);
  assert.equal(workspaceDir, path.join(data, 'workspaces', 'desktop'));
  assert.ok(fs.statSync(workspaceDir).isDirectory());
  assert.ok(fs.statSync(path.join(data, 'desktop')).isDirectory());
  assert.equal(fs.existsSync(path.join(dir, 'Tsugite')), false);
});

test('XDG_DATA_HOME moves the state dir and workspace', () => {
  const dirs = desktopDirs({ env: { XDG_DATA_HOME: '/x/data' }, home: '/home/u' });
  assert.deepEqual(dirs, { stateDir: '/x/data/tsugite/desktop', workspaceDir: '/x/data/tsugite/workspaces/desktop' });
});
test('later runs keep an edited config and its free port', async () => {
  const dir = tmp();
  const first = await ensureConfig({ userData: dir, home: dir, env: {} });
  fs.appendFileSync(first.configPath, 'log_level: debug\n');
  const second = await ensureConfig({ userData: dir, home: dir, env: {} });
  assert.equal(second.port, first.port);
  assert.match(fs.readFileSync(first.configPath, 'utf8'), /log_level: debug/);
});

test('a taken port is replaced in the config', async () => {
  const dir = tmp();
  const first = await ensureConfig({ userData: dir, home: dir, env: {} });
  const blocker = net.createServer().listen(first.port, '127.0.0.1');
  await new Promise((r) => blocker.once('listening', r));
  try {
    assert.equal(await isPortFree(first.port), false);
    const second = await ensureConfig({ userData: dir, home: dir, env: {} });
    assert.notEqual(second.port, first.port);
    assert.equal(readPort(second.configPath), second.port);
    assert.equal(second.url, `http://127.0.0.1:${second.port}`);
  } finally {
    blocker.close();
  }
});

const SECRETS = (stateDir) =>
  `plugins:\n  secrets:\n    provider: sqlite\n    path: ${JSON.stringify(path.join(stateDir, 'secrets.db'))}\n    keychain: true\n    keychain_account: desktop\n`;

test('first run on macOS keeps secrets in sqlite with the key in the keychain', async () => {
  const dir = tmp();
  const { configPath } = await ensureConfig({ userData: dir, home: dir, env: {}, platform: 'darwin' });
  const { stateDir } = desktopDirs({ env: {}, home: dir });
  assert.ok(fs.readFileSync(configPath, 'utf8').endsWith(SECRETS(stateDir)));
});

test('Linux gets the keychain secrets section only when secret-tool is on PATH', async () => {
  const without = tmp();
  const a = await ensureConfig({ userData: without, home: without, env: { PATH: without }, platform: 'linux' });
  assert.doesNotMatch(fs.readFileSync(a.configPath, 'utf8'), /secrets:/);

  const dir = tmp();
  const bin = path.join(dir, 'bin');
  fs.mkdirSync(bin);
  fs.writeFileSync(path.join(bin, 'secret-tool'), '', { mode: 0o755 });
  const b = await ensureConfig({ userData: dir, home: dir, env: { PATH: bin }, platform: 'linux' });
  assert.match(fs.readFileSync(b.configPath, 'utf8'), /keychain: true/);
});

test('an existing config without a secrets section gets one appended', async () => {
  const dir = tmp();
  const { configPath } = await ensureConfig({ userData: dir, home: dir, env: {}, platform: 'linux' });
  fs.appendFileSync(configPath, 'log_level: debug\n');
  await ensureConfig({ userData: dir, home: dir, env: {}, platform: 'darwin' });
  const text = fs.readFileSync(configPath, 'utf8');
  assert.match(text, /log_level: debug\n/);
  assert.ok(text.endsWith(SECRETS(desktopDirs({ env: {}, home: dir }).stateDir)));
});

test('an existing plugins section gets secrets added under it', async () => {
  const dir = tmp();
  const { configPath } = await ensureConfig({ userData: dir, home: dir, env: {}, platform: 'linux' });
  fs.appendFileSync(configPath, 'plugins:\n  other:\n    enabled: true\n');
  await ensureConfig({ userData: dir, home: dir, env: {}, platform: 'darwin' });
  const text = fs.readFileSync(configPath, 'utf8');
  assert.equal(text.match(/^plugins:/gm).length, 1);
  assert.match(text, /^plugins:\n  secrets:\n    provider: sqlite\n[\s\S]*  other:\n    enabled: true\n/m);
});

test('a plugins line with an inline value stops instead of adding a second plugins key', async () => {
  const dir = tmp();
  const { configPath } = await ensureConfig({ userData: dir, home: dir, env: {}, platform: 'linux' });
  fs.appendFileSync(configPath, 'plugins: {}\n');
  await assert.rejects(ensureConfig({ userData: dir, home: dir, env: {}, platform: 'darwin' }), /plugins/);
  assert.equal(fs.readFileSync(configPath, 'utf8').match(/^plugins:/gm).length, 1);
});

test('a user-edited secrets section is kept', async () => {
  const dir = tmp();
  const { configPath } = await ensureConfig({ userData: dir, home: dir, env: {}, platform: 'linux' });
  fs.appendFileSync(configPath, 'plugins:\n  secrets:\n    provider: env\n');
  await ensureConfig({ userData: dir, home: dir, env: {}, platform: 'darwin' });
  const text = fs.readFileSync(configPath, 'utf8');
  assert.match(text, /provider: env\n$/);
  assert.doesNotMatch(text, /sqlite/);
});
