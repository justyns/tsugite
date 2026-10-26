const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');

const HOST = '127.0.0.1';
const PORT_RE = /^\s*port:\s*(\d+)\s*$/m;
const PLUGINS_RE = /^plugins:[ \t]*$/m;
const SECRETS_RE = /^[ \t]+secrets:/m;

function isPortFree(port, host = HOST) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once('error', () => resolve(false));
    srv.listen(port, host, () => srv.close(() => resolve(true)));
  });
}

function freePort(host = HOST) {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, host, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

// The desktop daemon gets its own state dir and workspace under XDG data, apart from a CLI daemon's defaults.
function desktopDirs({ env = process.env, home = os.homedir() } = {}) {
  const base = path.join(env.XDG_DATA_HOME || path.join(home, '.local', 'share'), 'tsugite');
  return { stateDir: path.join(base, 'desktop'), workspaceDir: path.join(base, 'workspaces', 'desktop') };
}

function renderConfig({ stateDir, workspaceDir, port }) {
  // JSON strings are valid YAML scalars.
  return [
    `state_dir: ${JSON.stringify(stateDir)}`,
    `default_workspace_dir: ${JSON.stringify(workspaceDir)}`,
    'http:',
    '  enabled: true',
    `  host: ${HOST}`,
    `  port: ${port}`,
    '',
  ].join('\n');
}

// macOS always ships /usr/bin/security.
function keychainAvailable(platform, env) {
  if (platform === 'darwin') return true;
  return (env.PATH || '').split(path.delimiter).some((dir) => dir && fs.existsSync(path.join(dir, 'secret-tool')));
}

function withSecrets(text, stateDir) {
  if (SECRETS_RE.test(text)) return text;
  const entry = [
    '  secrets:',
    '    provider: sqlite',
    `    path: ${JSON.stringify(path.join(stateDir, 'secrets.db'))}`,
    '    keychain: true',
    '    keychain_account: desktop',
  ].join('\n');
  if (PLUGINS_RE.test(text)) return text.replace(PLUGINS_RE, (line) => `${line}\n${entry}`);
  if (/^plugins:/m.test(text)) throw new Error('daemon.yaml has a plugins line with an inline value; add the secrets section by hand');
  return `${text}${text.endsWith('\n') ? '' : '\n'}plugins:\n${entry}\n`;
}

function readPort(configPath) {
  const m = PORT_RE.exec(fs.readFileSync(configPath, 'utf8'));
  if (!m) throw new Error(`No http port in ${configPath}`);
  return Number(m[1]);
}

// Creates daemon.yaml or keeps the user's edited copy, changing only a taken port line
// and adding a keychain secrets section when none exists.
async function ensureConfig({ userData, home = os.homedir(), env = process.env, platform = process.platform }) {
  const configPath = path.join(userData, 'daemon.yaml');
  const { stateDir, workspaceDir } = desktopDirs({ env, home });
  fs.mkdirSync(workspaceDir, { recursive: true });
  let port;
  let text;
  let original = null;
  if (!fs.existsSync(configPath)) {
    port = await freePort();
    fs.mkdirSync(stateDir, { recursive: true });
    text = renderConfig({ stateDir, workspaceDir, port });
  } else {
    port = readPort(configPath);
    text = original = fs.readFileSync(configPath, 'utf8');
    if (!(await isPortFree(port))) {
      port = await freePort();
      text = text.replace(PORT_RE, `  port: ${port}`);
    }
  }
  if (keychainAvailable(platform, env)) text = withSecrets(text, stateDir);
  if (text !== original) fs.writeFileSync(configPath, text);
  return { configPath, port, workspaceDir, url: `http://${HOST}:${port}` };
}

module.exports = { ensureConfig, desktopDirs, readPort, isPortFree };
