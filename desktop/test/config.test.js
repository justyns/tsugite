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
