const fs = require('node:fs');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');

const HOST = '127.0.0.1';
const PORT_RE = /^\s*port:\s*(\d+)\s*$/m;

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

function readPort(configPath) {
  const m = PORT_RE.exec(fs.readFileSync(configPath, 'utf8'));
  if (!m) throw new Error(`No http port in ${configPath}`);
  return Number(m[1]);
}

// Writes daemon.yaml on first run. On later runs keeps the file (the user may have edited it)
// and rewrites only the port line when that port is taken.
async function ensureConfig({ userData, home = os.homedir(), env = process.env }) {
  const configPath = path.join(userData, 'daemon.yaml');
  const { stateDir, workspaceDir } = desktopDirs({ env, home });
  fs.mkdirSync(workspaceDir, { recursive: true });
  let port;
  if (!fs.existsSync(configPath)) {
    port = await freePort();
    fs.mkdirSync(stateDir, { recursive: true });
    fs.writeFileSync(configPath, renderConfig({ stateDir, workspaceDir, port }));
  } else {
    port = readPort(configPath);
    if (!(await isPortFree(port))) {
      port = await freePort();
      const text = fs.readFileSync(configPath, 'utf8').replace(PORT_RE, `  port: ${port}`);
      fs.writeFileSync(configPath, text);
    }
  }
  return { configPath, port, workspaceDir, url: `http://${HOST}:${port}` };
}

module.exports = { ensureConfig, desktopDirs, readPort, isPortFree };
