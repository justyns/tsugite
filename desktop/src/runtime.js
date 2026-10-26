const fs = require('node:fs');
const path = require('node:path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');

function runtimeBin(dir) {
  return path.join(dir, 'bin', 'tsu');
}

function onPath(name, pathVar) {
  return (pathVar || '').split(path.delimiter).some((d) => d && fs.existsSync(path.join(d, name)));
}

// Returns {command, args, source}; `args` is a prefix placed before tsu's own arguments.
function resolveRuntime({ env = process.env, resourcesPath, isPackaged = false, repoRoot = REPO_ROOT } = {}) {
  if (env.TSUGITE_DESKTOP_RUNTIME) {
    const bin = runtimeBin(env.TSUGITE_DESKTOP_RUNTIME);
    if (!fs.existsSync(bin)) throw new Error(`TSUGITE_DESKTOP_RUNTIME has no bin/tsu: ${bin}`);
    return { command: bin, args: [], source: 'env' };
  }
  if (resourcesPath) {
    const bin = runtimeBin(path.join(resourcesPath, 'runtime'));
    if (fs.existsSync(bin)) return { command: bin, args: [], source: 'bundled' };
    if (isPackaged) throw new Error(`Bundled runtime missing: ${bin}`);
  }
  if (!isPackaged && fs.existsSync(path.join(repoRoot, 'pyproject.toml')) && onPath('uv', env.PATH)) {
    return {
      command: 'uv',
      args: ['run', '--project', repoRoot, '--extra', 'daemon', 'tsu'],
      source: 'uv',
    };
  }
  if (onPath('tsu', env.PATH)) return { command: 'tsu', args: [], source: 'path' };
  throw new Error('No tsugite runtime found. Set TSUGITE_DESKTOP_RUNTIME, install uv, or put tsu on PATH.');
}

module.exports = { resolveRuntime };
