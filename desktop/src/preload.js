const { contextBridge, ipcRenderer } = require('electron');

if (location.protocol === 'file:') {
  contextBridge.exposeInMainWorld('desktop', {
    info: () => ipcRenderer.invoke('shell:info'),
    retry: () => ipcRenderer.invoke('shell:retry'),
    saveRemote: (url, token) => ipcRenderer.invoke('shell:save-remote', url, token),
    useLocal: () => ipcRenderer.invoke('shell:use-local'),
  });
} else {
  window.localStorage.removeItem('tsugite_token');
  contextBridge.exposeInMainWorld('tsugiteDesktop', { token: ipcRenderer.sendSync('shell:token') });
}
