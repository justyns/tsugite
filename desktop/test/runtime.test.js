const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { resolveRuntime } = require('../src/runtime');
const { tmp } = require('./helpers');

function runtimeDir(root) {
  fs.mkdirSync(path.join(root, 'bin'), { recursive: true });
  fs.writeFileSync(path.join(root, 'bin', 'tsu'), '');
  return root;
}

function setup() {
  const root = tmp();
  const env = runtimeDir(path.join(root, 'env'));
  const resources = path.join(root, 'res');
  runtimeDir(path.join(resources, 'runtime'));
  const repo = path.join(root, 'repo');
  fs.mkdirSync(repo);
  fs.writeFileSync(path.join(repo, 'pyproject.toml'), '');
  const bin = path.join(root, 'bin');
  fs.mkdirSync(bin);
  for (const n of ['uv', 'tsu']) fs.writeFileSync(path.join(bin, n), '');
  return { env, resources, repo, bin };
}

test('env override is used before the bundled runtime, uv and PATH', () => {
  const s = setup();
  const r = resolveRuntime({ env: { TSUGITE_DESKTOP_RUNTIME: s.env, PATH: s.bin }, resourcesPath: s.resources, repoRoot: s.repo });
  assert.deepEqual([r.source, r.command], ['env', path.join(s.env, 'bin', 'tsu')]);
});

test('bundled runtime comes before uv and PATH', () => {
  const s = setup();
  const r = resolveRuntime({ env: { PATH: s.bin }, resourcesPath: s.resources, repoRoot: s.repo });
  assert.equal(r.source, 'bundled');
});

test('development uses uv run with the daemon extra, then PATH', () => {
  const s = setup();
  const r = resolveRuntime({ env: { PATH: s.bin }, resourcesPath: path.join(s.resources, 'none'), repoRoot: s.repo });
  assert.equal(r.source, 'uv');
  assert.deepEqual(r.args, ['run', '--project', s.repo, '--extra', 'daemon', 'tsu']);
  const noUv = path.join(path.dirname(s.bin), 'bin2');
  fs.mkdirSync(noUv);
  fs.writeFileSync(path.join(noUv, 'tsu'), '');
  assert.equal(resolveRuntime({ env: { PATH: noUv }, repoRoot: s.repo }).source, 'path');
});

test('packaged app without bundled runtime fails instead of falling back', () => {
  const s = setup();
  assert.throws(() => resolveRuntime({ env: { PATH: s.bin }, resourcesPath: path.join(s.resources, 'none'), isPackaged: true, repoRoot: s.repo }), /Bundled runtime missing/);
});

test('bad env override and nothing available are errors', () => {
  const s = setup();
  assert.throws(() => resolveRuntime({ env: { TSUGITE_DESKTOP_RUNTIME: '/nonexistent' } }), /no bin\/tsu/);
  assert.throws(() => resolveRuntime({ env: { PATH: '' }, repoRoot: path.join(s.repo, 'x') }), /No tsugite runtime/);
});
