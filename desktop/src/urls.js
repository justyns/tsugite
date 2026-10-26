const path = require('node:path');
const { fileURLToPath } = require('node:url');

const SHELL_FILES = new Set(['error.html', 'remote.html'].map((f) => path.join(__dirname, f)));

function originOf(url) {
  try {
    return new URL(url).origin;
  } catch {
    return null;
  }
}

function isShellFile(url) {
  try {
    const u = new URL(url);
    return u.protocol === 'file:' && SHELL_FILES.has(fileURLToPath(u));
  } catch {
    return false;
  }
}

function tokenFor({ origin, token }, senderUrl) {
  return origin && originOf(senderUrl) === origin ? token : '';
}

module.exports = { originOf, isShellFile, tokenFor };
