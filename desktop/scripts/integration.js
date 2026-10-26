// Runs the supervision modules against the real daemon (dev runtime) with HOME and XDG dirs in a temp dir.
// UV_CACHE_DIR and UV_PROJECT_ENVIRONMENT from the caller are passed through so uv does not re-download.
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { ensureConfig } = require('../src/config');
const { Daemon } = require('../src/daemon');
const { RotatingLog } = require('../src/log');
const { resolveRuntime } = require('../src/runtime');
const { ensureLocalToken, tokenWorks } = require('../src/token');
const { alive } = require('../test/helpers');

(async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'tsugite-desktop-it-'));
  const userData = path.join(home, 'userData');
  fs.mkdirSync(userData);
  const env = {
    ...process.env,
    HOME: home,
    XDG_CONFIG_HOME: path.join(home, '.config'),
    XDG_DATA_HOME: path.join(home, '.local', 'share'),
    XDG_STATE_HOME: path.join(home, '.local', 'state'),
    XDG_CACHE_HOME: path.join(home, '.cache'),
  };
  const runtime = resolveRuntime({ env: process.env });
  console.log('runtime:', runtime.source);
  const { configPath, url } = await ensureConfig({ userData, home, env });
  const log = new RotatingLog(path.join(userData, 'logs', 'daemon.log'));
  const daemon = new Daemon({
    runtime, configPath, url, log, env, cwd: userData,
    pidFile: path.join(userData, 'daemon.pid'),
    healthTimeoutMs: 180000,
  });
  try {
    await daemon.start();
  } catch (err) {
    console.error(err.message, '\n', log.tail());
    process.exit(1);
  }
  const pid = daemon.child.pid;
  console.log('health ok at', url, 'pid', pid);

  const health = await (await fetch(`${url}/api/health`)).json();
  assert.equal(health.status, 'ok');
  const root = await fetch(`${url}/`);
  assert.equal(root.status, 200, `GET / returned ${root.status}`);
  assert.match(await root.text(), /<html/i);
  console.log('GET / serves the UI');

  let stored = '';
  const store = { read: async () => stored, write: async (t) => (stored = t) };
  const token = await ensureLocalToken({ daemon, store });
  assert.ok(await tokenWorks(url, token));
  assert.equal((await fetch(`${url}/api/runtime`)).status, 401);
  assert.equal(await ensureLocalToken({ daemon, store }), token);
  console.log('token minted, accepted with bearer, rejected without, reused on second call');

  await daemon.stop();
  await new Promise((r) => setTimeout(r, 200));
  assert.equal(alive(pid), false, 'daemon pid still alive');
  const left = spawnSync('pgrep', ['-f', configPath], { encoding: 'utf8' }).stdout.split('\n').filter(Boolean);
  assert.equal(left.length, 0, `leftover processes: ${left}`);
  console.log('shutdown leaves no process');
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
