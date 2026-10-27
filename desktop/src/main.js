const { app, BrowserWindow, Menu, dialog, ipcMain, safeStorage, screen, shell } = require('electron');
const path = require('node:path');
const { desktopDirs, ensureConfig } = require('./config');
const { Daemon, killStale, waitForHealth } = require('./daemon');
const { daemonEnv } = require('./env');
const { daemonItems } = require('./menu');
const { RotatingLog } = require('./log');
const { resolveRuntime } = require('./runtime');
const { Settings } = require('./settings');
const { quitHandler } = require('./quit');
const { Supervisor } = require('./supervisor');
const { ensureLocalToken } = require('./token');
const { isShellFile, originOf, tokenFor } = require('./urls');

app.setPath('userData', path.join(app.getPath('appData'), 'Tsugite'));

const userData = app.getPath('userData');
const ICON = path.join(__dirname, 'assets', 'icon.png');
const logDir = path.join(userData, 'logs');
const configPath = path.join(userData, 'daemon.yaml');
const pidFile = path.join(userData, 'daemon.pid');

let settings;
let log;
let win = null;
let remoteWin = null;
let supervisor;
let localTokenStore;
let onBeforeQuit = () => {};

function visibleBounds(bounds) {
  const fits = screen.getAllDisplays().some(({ workArea: a }) => bounds.x >= a.x && bounds.y >= a.y && bounds.x < a.x + a.width && bounds.y < a.y + a.height);
  return fits ? bounds : { width: bounds.width, height: bounds.height };
}

function createWindow() {
  const w = new BrowserWindow({
    ...visibleBounds(settings.data.bounds),
    title: 'Tsugite',
    icon: ICON,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true, preload: path.join(__dirname, 'preload.js') },
  });
  win = w;
  const saveBounds = () => !w.isMinimized() && !w.isFullScreen() && settings.update({ bounds: w.getBounds() });
  w.on('resized', saveBounds);
  w.on('moved', saveBounds);
  w.on('closed', () => win === w && (win = null));

  const allowed = (url) => {
    const { origin } = supervisor.target;
    return isShellFile(url) || (origin && originOf(url) === origin);
  };
  const external = (url) => /^(https?|mailto):/.test(url) && shell.openExternal(url);
  const guard = (event, url) => {
    if (allowed(url)) return;
    event.preventDefault();
    external(url);
  };
  w.webContents.on('will-navigate', guard);
  w.webContents.on('will-redirect', guard);
  w.webContents.setWindowOpenHandler(({ url }) => {
    external(url);
    return { action: 'deny' };
  });
  w.webContents.on('did-fail-load', (_event, code, description, url, isMainFrame) => {
    if (isMainFrame && code !== -3 && !isShellFile(url)) showError(`Could not load ${url}: ${description}`);
  });
}

// did-fail-load reports load failures.
const load = (promise) => promise.catch(() => {});

function showTarget({ origin }) {
  if (!win) createWindow();
  load(win.loadURL(origin));
}

let errorInfo = { message: '', logTail: '' };
function showError(message) {
  errorInfo = { message, logTail: log.tail() };
  if (!win) createWindow();
  load(win.loadFile(path.join(__dirname, 'error.html')));
}

function reopen() {
  if (supervisor.target.origin) showTarget(supervisor.target);
  else supervisor.launch();
}

async function connect({ adopt, superseded }) {
  if (settings.data.mode === 'remote') {
    const url = settings.data.remoteUrl;
    await waitForHealth(url, { timeoutMs: 10000, aborted: () => superseded() && 'Launch superseded' });
    return { origin: url, token: await settings.remoteToken() };
  }
  await killStale(pidFile, configPath);
  const { url, workspaceDir } = await ensureConfig({ userData });
  const runtime = resolveRuntime({ resourcesPath: process.resourcesPath, isPackaged: app.isPackaged });
  log.write(Buffer.from(`\n--- starting daemon via ${runtime.source} at ${url} ---\n`));
  const env = await daemonEnv({ log: (msg) => log.write(Buffer.from(`${msg}\n`)) });
  if (superseded()) throw new Error('Launch superseded');
  const daemon = new Daemon({ runtime, configPath, url, log, pidFile, env, cwd: workspaceDir });
  adopt(daemon);
  await daemon.start();
  if (superseded()) throw new Error('Launch superseded');
  return { origin: url, token: await ensureLocalToken({ daemon, store: localTokenStore }) };
}

function openRemoteWindow() {
  if (remoteWin) return remoteWin.focus();
  remoteWin = new BrowserWindow({
    width: 440, height: 340, parent: win || undefined, modal: !!win, title: 'Remote daemon',
    icon: ICON,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true, preload: path.join(__dirname, 'preload.js') },
  });
  remoteWin.setMenuBarVisibility(false);
  remoteWin.on('closed', () => (remoteWin = null));
  remoteWin.loadFile(path.join(__dirname, 'remote.html')).catch((err) => showError(`Could not open the remote daemon form: ${err.message}`));
}

function buildMenu() {
  const open = (p) => () => shell.openPath(p);
  const template = [
    ...(process.platform === 'darwin' ? [{ role: 'appMenu' }] : []),
    {
      label: 'File',
      submenu: [
        { label: 'Restart Daemon', click: () => supervisor.launch() },
        { label: 'Open Logs Folder', click: open(logDir) },
        { label: 'Open Config File', click: open(configPath) },
        { label: 'Open Workspace Folder', click: open(desktopDirs().workspaceDir) },
        { type: 'separator' },
        ...daemonItems({
          mode: settings.data.mode,
          remoteUrl: settings.data.remoteUrl,
          hasRemote: settings.hasRemote(),
          onUse: useMode,
          onConnect: openRemoteWindow,
        }),
        { type: 'separator' },
        process.platform === 'darwin' ? { role: 'close' } : { role: 'quit' },
      ],
    },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function useMode(mode) {
  settings.update({ mode });
  buildMenu();
  return supervisor.launch();
}

ipcMain.on('shell:token', (event) => {
  event.returnValue = tokenFor(supervisor.target, event.senderFrame?.url);
});
ipcMain.handle('shell:info', () => errorInfo);
ipcMain.handle('shell:retry', () => supervisor.launch());
ipcMain.handle('shell:use-local', () => {
  remoteWin?.close();
  return useMode('local');
});
ipcMain.handle('shell:save-remote', async (_event, url, token) => {
  try {
    await settings.setRemote(url, token);
  } catch (err) {
    return err.message;
  }
  remoteWin?.close();
  buildMenu();
  await supervisor.launch();
});

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!win) return supervisor && reopen();
    if (win.isMinimized()) win.restore();
    win.focus();
  });
  app.whenReady().then(() => {
    settings = new Settings(userData, safeStorage);
    log = new RotatingLog(path.join(logDir, 'daemon.log'));
    localTokenStore = settings.localTokenStore((reason) => log.write(Buffer.from(`\n--- keeping the local token in memory only: ${reason} ---\n`)));
    supervisor = new Supervisor({
      connect,
      onReady: showTarget,
      onError: showError,
    });
    onBeforeQuit = quitHandler({
      supervisor,
      exit: () => app.exit(0),
      onFailed: (err) => showError(`The daemon did not stop: ${err.message}. Quit again to retry.`),
    });
    buildMenu();
    createWindow();
    supervisor.launch();
  }).catch((err) => {
    dialog.showErrorBox('Tsugite failed to start', err.stack || String(err));
    app.exit(1);
  });
  app.on('activate', () => !win && reopen());
  app.on('window-all-closed', () => process.platform !== 'darwin' && app.quit());
  app.on('before-quit', (event) => onBeforeQuit(event));
  for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => app.quit());
}
