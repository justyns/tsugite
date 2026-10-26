const { spawn, execFileSync } = require('node:child_process');
const { EventEmitter } = require('node:events');
const fs = require('node:fs');
const path = require('node:path');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitForHealth(url, { timeoutMs = 60000, intervalMs = 250, aborted = () => null } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastError = 'no response';
  for (;;) {
    const reason = aborted();
    if (reason) throw new Error(reason);
    try {
      const res = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) return;
      lastError = `HTTP ${res.status}`;
    } catch (err) {
      lastError = err.cause?.code || err.message;
    }
    if (Date.now() >= deadline) throw new Error(`Daemon did not become healthy within ${timeoutMs / 1000}s (${lastError})`);
    await sleep(intervalMs);
  }
}

function groupAlive(pgid) {
  try {
    process.kill(-pgid, 0);
    return true;
  } catch (err) {
    if (err.code === 'ESRCH') return false;
    throw err;
  }
}

async function groupGone(pgid, timeoutMs = 5000) {
  const deadline = Date.now() + timeoutMs;
  while (groupAlive(pgid)) {
    if (Date.now() >= deadline) return false;
    await sleep(10);
  }
  return true;
}

function within(promise, ms) {
  let timer;
  const expired = new Promise((r) => (timer = setTimeout(r, ms, false)));
  return Promise.race([promise.then(() => true), expired]).finally(() => clearTimeout(timer));
}

function signalGroup(pgid, sig) {
  try {
    process.kill(-pgid, sig);
  } catch (err) {
    if (err.code !== 'ESRCH') throw err;
  }
}

// Kills a daemon left behind by a crashed app, but only when it is a group leader running our daemon invocation.
async function killStale(pidFile, configPath) {
  if (!fs.existsSync(pidFile)) return;
  const pid = Number(fs.readFileSync(pidFile, 'utf8'));
  fs.rmSync(pidFile);
  if (!pid) return;
  let out;
  try {
    out = execFileSync('ps', ['-ww', '-p', String(pid), '-o', 'pgid=,command='], { encoding: 'utf8' });
  } catch (err) {
    if (err.status === 1) return;
    throw err;
  }
  const m = /^\s*(\d+)\s+(.*)$/.exec(out.trim());
  if (!m || Number(m[1]) !== pid || !`${m[2]} `.includes(` daemon --config ${configPath} `)) return;
  signalGroup(pid, 'SIGKILL');
  if (!(await groupGone(pid))) throw new Error(`Stale daemon group ${pid} survived SIGKILL`);
}

class Daemon extends EventEmitter {
  constructor({ runtime, configPath, url, log, pidFile, env = process.env, cwd, healthTimeoutMs = 60000, killTimeoutMs = 30000, runTimeoutMs = 60000 }) {
    super();
    Object.assign(this, { runtime, configPath, url, log, pidFile, env, cwd, healthTimeoutMs, killTimeoutMs, runTimeoutMs });
    this.child = null;
    this.pgid = null;
    this.exit = null;
    this.stopping = false;
  }

  tsuArgs(...args) {
    return [...this.runtime.args, ...args];
  }

  async start() {
    if (this.child) throw new Error('Daemon already running');
    this.exit = null;
    this.stopping = false;
    const child = spawn(this.runtime.command, this.tsuArgs('daemon', '--config', this.configPath), {
      cwd: this.cwd,
      env: this.env,
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    this.child = child;
    const pgid = child.pid ?? null;
    this.pgid = pgid;
    this.finished = new Promise((resolve) => {
      child.once('error', (err) => {
        this.child = null;
        this.exit = `Failed to start ${this.runtime.command}: ${err.message}`;
        resolve();
      });
      child.once('exit', async (code, signal) => {
        this.child = null;
        fs.rmSync(this.pidFile, { force: true });
        this.exit = this.exit || `Daemon exited (${signal || `code ${code}`})`;
        if (!this.stopping) {
          this.emit('exit', this.exit);
          signalGroup(pgid, 'SIGKILL');
          if (await groupGone(pgid)) this.pgid = null;
        }
        resolve();
      });
    });
    const onData = (buf) => this.log.write(buf);
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    if (child.pid) fs.writeFileSync(this.pidFile, String(child.pid));
    try {
      await waitForHealth(this.url, { timeoutMs: this.healthTimeoutMs, aborted: () => this.exit });
    } catch (err) {
      await this.stop();
      throw err;
    }
  }

  async stop() {
    if (this.pgid === null) return;
    const pgid = this.pgid;
    if (this.child) {
      this.stopping = true;
      signalGroup(pgid, 'SIGTERM');
      if (!(await within(this.finished, this.killTimeoutMs))) {
        this.log.write(Buffer.from(`\n--- daemon did not exit within ${this.killTimeoutMs / 1000}s of SIGTERM, sending SIGKILL ---\n`));
      }
    } else {
      await this.finished;
    }
    if (this.pgid === null) return;
    signalGroup(pgid, 'SIGKILL');
    if (!(await groupGone(pgid))) throw new Error(`Process group ${pgid} survived SIGKILL`);
    await this.finished;
    this.pgid = null;
  }

  // Runs a tsu subcommand to completion and returns its stdout.
  run(...args) {
    return new Promise((resolve, reject) => {
      const child = spawn(this.runtime.command, this.tsuArgs(...args), { cwd: this.cwd, env: this.env });
      let out = '';
      let err = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`tsu ${args.join(' ')} timed out after ${this.runTimeoutMs / 1000}s`));
      }, this.runTimeoutMs);
      child.stdout.on('data', (b) => (out += b));
      child.stderr.on('data', (b) => (err += b));
      child.once('error', (e) => {
        clearTimeout(timer);
        reject(e);
      });
      child.once('close', (code) => {
        clearTimeout(timer);
        if (code === 0) resolve(out);
        else reject(new Error(`tsu ${args.join(' ')} failed: ${err.trim()}`));
      });
    });
  }
}

module.exports = { Daemon, waitForHealth, killStale };
