const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const config = require('../electron-builder');

const root = path.join(__dirname, '..');

function pngSize(file) {
  const buf = fs.readFileSync(file);
  assert.equal(buf.toString('latin1', 1, 4), 'PNG', `${file} is not a PNG`);
  return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
}

test('the macOS icon is a 1024px PNG', () => {
  assert.deepEqual(pngSize(path.join(root, config.mac.icon)), [1024, 1024]);
});

test('every Linux icon matches the size in its file name', () => {
  const dir = path.join(root, config.linux.icon);
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.png'));
  assert.ok(files.length >= 7);
  for (const f of files) {
    const n = Number(f.split('x')[0]);
    assert.deepEqual(pngSize(path.join(dir, f)), [n, n], f);
  }
});

test('the window icon ships inside the app files', () => {
  assert.deepEqual(pngSize(path.join(root, 'src', 'assets', 'icon.png')), [512, 512]);
});