const test = require('node:test');
const assert = require('node:assert/strict');
const { quitHandler } = require('../src/quit');
const { Supervisor } = require('../src/supervisor');
const { alive, makeDaemon } = require('./helpers');

const until = async (cond) => {
  while (!cond()) await new Promise((r) => setTimeout(r, 10));
};

function harness(opts = () => ({})) {
  const h = { daemons: [], overlaps: 0, ready: [], errors: [] };
  h.supervisor = new Supervisor({
    connect: async ({ adopt }) => {
      const d = await makeDaemon(opts(h.daemons.length));
      if (h.daemons.some((o) => o.child && alive(o.child.pid))) h.overlaps++;
      h.daemons.push(d);
      adopt(d);
      await d.start();
      return { origin: d.url, token: 'tok' };
    },
    onReady: (t) => h.ready.push(t),
    onError: (m) => h.errors.push(m),
  });
  return h;
}

test('overlapping restarts never run two daemons and the last request wins', async () => {
  const h = harness((n) => ({ mode: n === 0 ? 'ignore-term' : '' }));
  await h.supervisor.launch();
  assert.equal(h.ready.length, 1);
  const second = h.supervisor.launch();
  const third = h.supervisor.launch();
  await Promise.all([second, third]);
  assert.equal(h.overlaps, 0);
  const last = h.daemons.at(-1);
  assert.deepEqual(h.ready.at(-1), { origin: last.url, token: 'tok' });
  assert.deepEqual(h.supervisor.target, h.ready.at(-1));
  assert.equal(h.daemons.filter((d) => d.child).length, 1);
  assert.equal(h.errors.length, 0);
  await h.supervisor.quit();
});

test('a launch superseded while its daemon is starting cuts that start short and reports nothing', { timeout: 30000 }, async () => {
  const h = harness((n) => (n === 0 ? { mode: 'never-healthy', healthTimeoutMs: 20000 } : {}));
  const first = h.supervisor.launch();
  await until(() => h.daemons[0]?.child);
  const t0 = Date.now();
  await Promise.all([first, h.supervisor.launch()]);
  assert.ok(Date.now() - t0 < 5000);
  assert.equal(h.overlaps, 0);
  assert.equal(h.errors.length, 0);
  assert.deepEqual(h.ready, [{ origin: h.daemons[1].url, token: 'tok' }]);
  await h.supervisor.quit();
});

test('quit during a restart waits for the old daemon and leaves nothing running', async () => {
  const h = harness((n) => ({ mode: n === 0 ? 'ignore-term' : '' }));
  await h.supervisor.launch();
  const old = h.daemons[0].child.pid;
  const restart = h.supervisor.launch();
  await h.supervisor.quit();
  assert.equal(alive(old), false);
  await restart;
  assert.equal(h.daemons.length, 1);
  assert.equal(h.ready.length, 1);
  assert.equal(h.errors.length, 0);
});

test('a stop that fails keeps the daemon owned and blocks the next launch', async () => {
  const errors = [];
  let connects = 0;
  let stopCalls = 0;
  const stuck = { on() {}, stop: () => (++stopCalls === 1 ? Promise.reject(new Error('group survived')) : Promise.resolve()) };
  const s = new Supervisor({
    connect: async ({ adopt }) => {
      connects++;
      adopt(stuck);
      return { origin: 'http://127.0.0.1:1', token: 't' };
    },
    onReady: () => {},
    onError: (m) => errors.push(m),
  });
  await s.launch();
  await s.launch();
  assert.equal(connects, 1);
  assert.match(errors.join('\n'), /group survived/);
  assert.ok(s.owned.has(stuck));
  await s.quit();
  assert.equal(s.owned.size, 0);
});

test('a daemon that dies after it is ready reports one error and the next launch recovers', { timeout: 10000 }, async () => {
  const h = harness();
  await h.supervisor.launch();
  process.kill(h.daemons[0].child.pid, 'SIGKILL');
  await until(() => h.errors.length);
  await new Promise((r) => setTimeout(r, 100));
  assert.equal(h.errors.length, 1);
  assert.match(h.errors[0], /SIGKILL/);
  await h.supervisor.launch();
  assert.equal(h.ready.length, 2);
  assert.ok(alive(h.daemons[1].child.pid));
  assert.equal(h.errors.length, 1);
  await h.supervisor.quit();
});

test('a daemon that dies during quit reports nothing', async () => {
  const h = harness(() => ({ mode: 'ignore-term' }));
  await h.supervisor.launch();
  const quitting = h.supervisor.quit();
  process.kill(h.daemons[0].child.pid, 'SIGKILL');
  await quitting;
  assert.deepEqual(h.errors, []);
});

test('a daemon that exits during start reports its exit once', async () => {
  const h = harness(() => ({ mode: 'exit-early' }));
  await h.supervisor.launch();
  await new Promise((r) => setTimeout(r, 100));
  assert.deepEqual(h.errors, ['Daemon exited (code 3)']);
});

test('a daemon that exits while its token is minted reports the exit, not the failed request', async () => {
  const errors = [];
  const s = new Supervisor({
    connect: async ({ adopt }) => {
      const d = await makeDaemon();
      adopt(d);
      await d.start();
      const exited = new Promise((r) => d.once('exit', r));
      process.kill(d.child.pid, 'SIGKILL');
      await exited;
      throw new Error('fetch failed');
    },
    onReady: () => {},
    onError: (m) => errors.push(m),
  });
  await s.launch();
  await new Promise((r) => setTimeout(r, 100));
  assert.deepEqual(errors, ['Daemon exited (SIGKILL)']);
});

test('a daemon that exits just after the token request fails still reports the exit', async () => {
  const errors = [];
  const s = new Supervisor({
    connect: async ({ adopt }) => {
      const d = await makeDaemon();
      adopt(d);
      await d.start();
      setTimeout(() => process.kill(d.child.pid, 'SIGKILL'), 50);
      throw new Error('fetch failed');
    },
    onReady: () => {},
    onError: (m) => errors.push(m),
  });
  await s.launch();
  await new Promise((r) => setTimeout(r, 100));
  assert.deepEqual(errors, ['Daemon exited (SIGKILL)']);
  await s.quit();
});

test('a connect failure with a daemon that stays up reports the connect error after the grace period', async () => {
  const errors = [];
  const s = new Supervisor({
    connect: async ({ adopt }) => {
      const d = await makeDaemon();
      adopt(d);
      await d.start();
      throw new Error('fetch failed');
    },
    onReady: () => {},
    onError: (m) => errors.push(m),
    exitGraceMs: 100,
  });
  await s.launch();
  assert.deepEqual(errors, ['fetch failed']);
  await s.quit();
});

test('after a failed quit, launch works again', async () => {
  let failing = true;
  let connects = 0;
  const stuck = { on() {}, stop: () => (failing ? Promise.reject(new Error('group survived')) : Promise.resolve()) };
  const s = new Supervisor({
    connect: async ({ adopt }) => {
      connects++;
      adopt(stuck);
      return { origin: 'http://127.0.0.1:1', token: 't' };
    },
    onReady: () => {},
    onError: () => {},
  });
  await s.launch();
  await assert.rejects(s.quit(), /group survived/);
  failing = false;
  await s.launch();
  assert.equal(connects, 2);
  await s.quit();
});

test('quit does not wait for a connect that never settles', { timeout: 5000 }, async () => {
  let connecting = false;
  const connect = () => {
    connecting = true;
    return new Promise(() => {});
  };
  const s = new Supervisor({ connect, onReady: () => {}, onError: () => {} });
  s.launch();
  await until(() => connecting);
  await s.quit();
});

test('every quit request is held until the supervised shutdown finishes, and exit runs once', async () => {
  let release;
  const supervisor = { quitting: false, quit() { this.quitting = true; return new Promise((r) => (release = r)); } };
  let exits = 0;
  const handler = quitHandler({ supervisor, exit: () => exits++ });
  const events = [0, 1, 2].map(() => ({ prevented: false, preventDefault() { this.prevented = true; } }));
  for (const e of events) handler(e);
  assert.deepEqual(events.map((e) => e.prevented), [true, true, true]);
  assert.equal(exits, 0);
  release();
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(exits, 1);
});

test('a failed shutdown keeps the app open and the next quit retries', async () => {
  let calls = 0;
  const supervisor = { quit: () => (++calls === 1 ? Promise.reject(new Error('group survived')) : Promise.resolve()) };
  let exits = 0;
  const failures = [];
  const handler = quitHandler({ supervisor, exit: () => exits++, onFailed: (e) => failures.push(e.message) });
  const ev = () => ({ preventDefault() { this.prevented = true; } });
  const first = ev();
  handler(first);
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(first.prevented);
  assert.equal(exits, 0);
  assert.deepEqual(failures, ['group survived']);
  const second = ev();
  handler(second);
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(second.prevented);
  assert.equal(calls, 2);
  assert.equal(exits, 1);
});