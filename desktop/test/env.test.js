const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { daemonEnv, fallbackPath } = require('../src/env');
const { tmp } = require('./helpers');

function fakeShell(body) {
  const file = path.join(tmp(), 'sh');
  fs.writeFileSync(file, `#!/bin/sh\n${body}\n`, { mode: 0o755 });
  return file;
}

test('uses the PATH the login shell prints, ignoring startup noise', async () => {
  const shell = fakeShell('echo "Welcome PATH=/nope"\necho HOME=/h\necho PATH=/home/u/.local/bin:/usr/bin');
  const env = await daemonEnv({ env: { PATH: '/usr/bin:/bin', KEEP: '1' }, shell });
  assert.equal(env.PATH, '/home/u/.local/bin:/usr/bin');
  assert.equal(env.KEEP, '1');
});

test('falls back to common install directories when the shell fails', async () => {
  const shell = fakeShell('exit 1');
  const logged = [];
  const env = await daemonEnv({
    env: { PATH: '/usr/bin:/bin' },
    shell,
    home: '/h',
    platform: 'darwin',
    log: (m) => logged.push(m),
  });
  assert.equal(env.PATH, '/h/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin');
  assert.equal(logged.length, 1);
});

test('falls back when the shell hangs past the timeout', async () => {
  const shell = fakeShell('exec sleep 5');
  const env = await daemonEnv({ env: { PATH: '/usr/bin' }, shell, home: '/h', timeoutMs: 200 });
  assert.match(env.PATH, /^\/h\/\.local\/bin:/);
});

test('falls back when a shell that ignores SIGTERM hangs past the timeout', async () => {
  const shell = fakeShell('trap "" TERM\nexec sleep 5');
  const started = Date.now();
  const env = await daemonEnv({ env: { PATH: '/usr/bin' }, shell, home: '/h', timeoutMs: 200 });
  assert.match(env.PATH, /^\/h\/\.local\/bin:/);
  assert.ok(Date.now() - started < 2000);
});

test('fallback does not repeat directories already on PATH', () => {
  assert.equal(fallbackPath('/usr/local/bin:/usr/bin', '/h', 'darwin'), '/h/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin');
});

test('fallback adds Homebrew directories on macOS only', () => {
  assert.equal(fallbackPath('/usr/bin', '/h', 'linux'), '/h/.local/bin:/usr/bin');
});
