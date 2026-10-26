const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { killStale, waitForHealth } = require('../src/daemon');
const { isPortFree } = require('../src/config');
const { RotatingLog } = require('../src/log');
const { ensureLocalToken, tokenWorks } = require('../src/token');
const { alive, makeDaemon, tmp } = require('./helpers');

const settle = (child) => Promise.race([new Promise((r) => child.once('exit', r)), new Promise((r) => setTimeout(r, 300))]);

test('start waits for health, logs output, and stop ends the process', async () => {
  const d = await makeDaemon();
  await d.start();
  const pid = d.child.pid;
  assert.equal(fs.readFileSync(d.pidFile, 'utf8'), String(pid));
  assert.ok((await fetch(`${d.url}/api/health`)).ok);
  assert.match(d.log.tail(), /fake daemon starting/);
  await d.stop();
  assert.equal(alive(pid), false);
  assert.equal(fs.existsSync(d.pidFile), false);
});

test('stop escalates to SIGKILL when SIGTERM is ignored', async () => {
  const d = await makeDaemon({ mode: 'ignore-term' });
  await d.start();
  const pid = d.child.pid;
  const t0 = Date.now();
  await d.stop();
  assert.ok(Date.now() - t0 >= 400);
  assert.equal(alive(pid), false);
  assert.match(d.log.tail(), /SIGKILL/);
});

test('start rejects with the exit reason when the daemon dies', async () => {
  const d = await makeDaemon({ mode: 'exit-early' });
  await assert.rejects(d.start(), /exited \(code 3\)/);
  assert.match(d.log.tail(), /boom: bad config/);
});

test('start times out and kills a daemon that never turns healthy', async () => {
  const d = await makeDaemon({ mode: 'never-healthy', healthTimeoutMs: 600 });
  const started = d.start();
  await assert.rejects(started, /did not become healthy/);
  assert.equal(d.child, null);
});

test('an unexpected exit after start emits exit', async () => {
  const d = await makeDaemon();
  await d.start();
  const exited = new Promise((r) => d.once('exit', r));
  process.kill(d.child.pid, 'SIGKILL');
  assert.match(await exited, /SIGKILL/);
});

test('killStale kills the stale group and waits until it is gone, so its port is free again', async () => {
  const d = await makeDaemon({ mode: 'orphan', childPidFile: path.join(tmp(), 'child.pid') });
  await d.start();
  const pid = d.child.pid;
  const childPid = Number(fs.readFileSync(d.env.FAKE_CHILD_PID_FILE, 'utf8'));
  const { port } = new URL(d.url);
  await killStale(d.pidFile, d.configPath);
  assert.equal(alive(childPid), false);
  assert.equal(alive(pid), false);
  assert.equal(await isPortFree(Number(port)), true);
});

test('killStale reads the full command line when COLUMNS is narrow', async () => {
  const d = await makeDaemon();
  await d.start();
  const pid = d.child.pid;
  const columns = process.env.COLUMNS;
  process.env.COLUMNS = '20';
  try {
    await killStale(d.pidFile, d.configPath);
  } finally {
    if (columns === undefined) delete process.env.COLUMNS;
    else process.env.COLUMNS = columns;
  }
  assert.equal(alive(pid), false);
});

test('an unexpected exit kills the rest of the group and releases it, so stop does nothing', async () => {
  const d = await makeDaemon({ mode: 'orphan', childPidFile: path.join(tmp(), 'child.pid') });
  await d.start();
  const childPid = Number(fs.readFileSync(d.env.FAKE_CHILD_PID_FILE, 'utf8'));
  process.kill(d.child.pid, 'SIGKILL');
  await d.finished;
  assert.equal(alive(childPid), false);
  assert.equal(d.pgid, null);
  await d.stop();
  assert.equal(d.stopping, false);
});

test('stop gives up on a group that survives SIGKILL instead of hanging', { timeout: 20000 }, async () => {
  // The zombie's parent never reaps it, so its group stays visible after SIGKILL.
  const script = 'my $p = fork; if (!$p) { setpgrp(0, 0); exit 0 } setpgrp($p, $p); $| = 1; print "$p\\n"; sleep 30';
  const holder = spawn('perl', ['-e', script], { stdio: ['ignore', 'pipe', 'ignore'] });
  const zombie = Number(await new Promise((r) => holder.stdout.once('data', r)));
  const d = await makeDaemon({ killTimeoutMs: 100 });
  Object.assign(d, { child: holder, pgid: zombie, finished: new Promise(() => {}) });
  await assert.rejects(d.stop(), new RegExp(`Process group ${zombie} survived SIGKILL`));
  holder.kill('SIGKILL');
});

test('killStale leaves a process whose command does not show our config', async () => {
  const d = await makeDaemon();
  await d.start();
  await killStale(d.pidFile, '/some/other/daemon.yaml');
  assert.ok(alive(d.child.pid));
  await d.stop();
});

test('killStale leaves an unrelated process whose arguments contain the config path', async () => {
  const dir = tmp();
  const configPath = path.join(dir, 'daemon.yaml');
  const editor = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)', configPath], { detached: true, stdio: 'ignore' });
  const pidFile = path.join(dir, 'daemon.pid');
  fs.writeFileSync(pidFile, String(editor.pid));
  await killStale(pidFile, configPath);
  await settle(editor);
  assert.equal(editor.exitCode ?? editor.signalCode, null);
  process.kill(-editor.pid, 'SIGKILL');
});

test('killStale leaves a daemon-looking process that is not a group leader', async () => {
  const dir = tmp();
  const configPath = path.join(dir, 'daemon.yaml');
  const child = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)', 'daemon', '--config', configPath], { stdio: 'ignore' });
  const pidFile = path.join(dir, 'daemon.pid');
  fs.writeFileSync(pidFile, String(child.pid));
  await killStale(pidFile, configPath);
  await settle(child);
  assert.equal(child.exitCode ?? child.signalCode, null);
  child.kill('SIGKILL');
});

test('killStale tolerates a pid that has vanished', async () => {
  const dir = tmp();
  const gone = spawn('true');
  await new Promise((r) => gone.once('exit', r));
  const pidFile = path.join(dir, 'daemon.pid');
  fs.writeFileSync(pidFile, String(gone.pid));
  await killStale(pidFile, path.join(dir, 'daemon.yaml'));
  assert.equal(fs.existsSync(pidFile), false);
});

test('a missing runtime command rejects start and stop resolves', async () => {
  const d = await makeDaemon({ runtime: { command: path.join(tmp(), 'no-such-tsu'), args: [] } });
  await assert.rejects(d.start(), /Failed to start/);
  await d.stop();
  assert.equal(d.child, null);
});

test('stop kills descendants that ignore SIGTERM', async () => {
  const d = await makeDaemon({ mode: 'orphan', childPidFile: path.join(tmp(), 'child.pid') });
  await d.start();
  const pgid = d.child.pid;
  const childPid = Number(fs.readFileSync(d.env.FAKE_CHILD_PID_FILE, 'utf8'));
  assert.ok(alive(childPid));
  await d.stop();
  assert.equal(alive(childPid), false);
  assert.throws(() => process.kill(-pgid, 0), { code: 'ESRCH' });
});

test('waitForHealth stops early when aborted', async () => {
  await assert.rejects(waitForHealth('http://127.0.0.1:1', { aborted: () => 'gone' }), /gone/);
});

const count = (validFile, kind) => Number(fs.readFileSync(`${validFile}.${kind}`, 'utf8'));
const memory = (token = '') => ({ read: async () => token, write: async (t) => (token = t) });

test('ensureLocalToken mints a token once and reuses it while the daemon accepts it', async () => {
  const validFile = path.join(tmp(), 'valid');
  const d = await makeDaemon({ validTokenFile: validFile });
  await d.start();
  const store = memory();
  const first = await ensureLocalToken({ daemon: d, store });
  assert.equal(await ensureLocalToken({ daemon: d, store }), first);
  assert.equal(count(validFile, 'mints'), 1);
  await d.stop();
});

test('ensureLocalToken re-mints a token the daemon rejects', async () => {
  const validFile = path.join(tmp(), 'valid');
  const d = await makeDaemon({ validTokenFile: validFile });
  await d.start();
  const store = memory('stale');
  const minted = await ensureLocalToken({ daemon: d, store });
  assert.equal(minted, fs.readFileSync(validFile, 'utf8'));
  assert.equal(await store.read(), minted);
  assert.equal(count(validFile, 'mints'), 1);
  await d.stop();
});

test('ensureLocalToken revokes every earlier desktop token before minting', async () => {
  const validFile = path.join(tmp(), 'valid');
  fs.writeFileSync(`${validFile}.live`, '3');
  const d = await makeDaemon({ validTokenFile: validFile });
  await d.start();
  await ensureLocalToken({ daemon: d, store: memory() });
  assert.equal(count(validFile, 'live'), 1);
  await ensureLocalToken({ daemon: d, store: memory('stale') });
  assert.equal(count(validFile, 'live'), 1);
  await d.stop();
});

test('RotatingLog rotates by size and keeps the configured number of rotated files', () => {
  const file = path.join(tmp(), 'logs', 'daemon.log');
  const log = new RotatingLog(file, { maxBytes: 10, keep: 3 });
  for (const s of ['aaaaaaaa\n', 'bbbbbbbb\n', 'cccccccc\n', 'dddddddd\n', 'eeeeeeee\n']) log.write(Buffer.from(s));
  assert.equal(fs.readFileSync(file, 'utf8'), 'eeeeeeee\n');
  assert.equal(fs.readFileSync(`${file}.1`, 'utf8'), 'dddddddd\n');
  assert.equal(fs.readFileSync(`${file}.3`, 'utf8'), 'bbbbbbbb\n');
  assert.equal(fs.existsSync(`${file}.4`), false);
});

test('a token check against a daemon that never answers times out', async () => {
  const http = require('node:http');
  const srv = http.createServer(() => {}).listen(0, '127.0.0.1');
  await new Promise((r) => srv.once('listening', r));
  const url = `http://127.0.0.1:${srv.address().port}`;
  const t0 = Date.now();
  await assert.rejects(tokenWorks(url, 'x', { timeoutMs: 300 }));
  assert.ok(Date.now() - t0 < 3000);
  srv.closeAllConnections();
  srv.close();
});