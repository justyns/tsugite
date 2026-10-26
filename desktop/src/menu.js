function daemonItems({ mode, remoteUrl, hasRemote, onUse, onConnect }) {
  const host = remoteUrl ? new URL(remoteUrl).host : '';
  return [
    { label: 'Local Daemon', type: 'radio', checked: mode !== 'remote', click: () => onUse('local') },
    {
      label: host ? `Remote Daemon (${host})` : 'Remote Daemon',
      type: 'radio',
      checked: mode === 'remote',
      enabled: hasRemote,
      click: () => onUse('remote'),
    },
    { label: hasRemote ? 'Change Remote Daemon...' : 'Connect to Remote Daemon...', click: onConnect },
  ];
}

module.exports = { daemonItems };