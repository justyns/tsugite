const { execFile } = require('node:child_process');
const os = require('node:os');
const path = require('node:path');

// Apps launched from Finder, the Dock or a desktop file get the session PATH, which misses
// directories the login shell adds (~/.local/bin, Homebrew, mise).
function loginShellPath({ shell = process.env.SHELL || '/bin/sh', timeoutMs = 5000 } = {}) {
  return new Promise((resolve) => {
    execFile(shell, ['-ilc', '/usr/bin/env'], { timeout: timeoutMs, killSignal: 'SIGKILL', encoding: 'utf8' }, (err, stdout) => {
      const line = (stdout || '').split('\n').reverse().find((l) => l.startsWith('PATH='));
      resolve(line ? { path: line.slice(5) } : { error: err ? err.message : 'no PATH in output' });
    });
  });
}

function fallbackPath(current, home = os.homedir(), platform = process.platform) {
  const extra = [path.join(home, '.local', 'bin'), ...(platform === 'darwin' ? ['/opt/homebrew/bin', '/usr/local/bin'] : [])];
  const dirs = (current || '/usr/bin:/bin').split(path.delimiter);
  return [...extra.filter((d) => !dirs.includes(d)), ...dirs].join(path.delimiter);
}

async function daemonEnv({ env = process.env, log = () => {}, ...opts } = {}) {
  const found = await loginShellPath(opts);
  if (found.path) return { ...env, PATH: found.path };
  log(`Could not read the login shell PATH (${found.error}), using fallback directories`);
  return { ...env, PATH: fallbackPath(env.PATH, opts.home, opts.platform) };
}

module.exports = { daemonEnv, fallbackPath, loginShellPath };
