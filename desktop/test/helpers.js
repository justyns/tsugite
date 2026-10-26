const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { ensureConfig } = require('../src/config');
const { Daemon } = require('../src/daemon');
const { RotatingLog } = require('../src/log');

const FAKE = path.join(__dirname, 'fake-daemon.js');

function alive(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function tmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'tsugite-desktop-test-'));
}

async function makeDaemon({ mode = '', validTokenFile = '', childPidFile = '', ...opts } = {}) {
  const dir = tmp();
  const { configPath, url } = await ensureConfig({ userData: dir, home: dir, env: {} });
  const daemon = new Daemon({
    runtime: { command: process.execPath, args: [FAKE] },
    configPath,
    url,
    log: new RotatingLog(path.join(dir, 'logs', 'daemon.log')),
    pidFile: path.join(dir, 'daemon.pid'),
    env: { ...process.env, FAKE_MODE: mode, FAKE_VALID_TOKEN_FILE: validTokenFile, FAKE_CHILD_PID_FILE: childPidFile },
    cwd: dir,
    healthTimeoutMs: 5000,
    killTimeoutMs: 500,
    ...opts,
  });
  daemon.dir = dir;
  return daemon;
}

module.exports = { alive, tmp, makeDaemon };
