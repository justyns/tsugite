async function tokenWorks(url, token, { timeoutMs = 10000 } = {}) {
  const res = await fetch(`${url}/api/runtime`, {
    headers: { Authorization: `Bearer ${token}` },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (res.status === 401) return false;
  if (!res.ok) throw new Error(`Daemon rejected the token check: HTTP ${res.status}`);
  return true;
}

// Returns an admin token the running daemon accepts, revoking earlier `desktop` tokens before minting a new one.
async function ensureLocalToken({ daemon, store }) {
  const stored = await store.read();
  if (stored && (await tokenWorks(daemon.url, stored))) return stored;
  const tsu = (...args) => daemon.run('daemon', 'token', ...args, '--config', daemon.configPath);
  while (await tsu('revoke', 'desktop').then(() => true, () => false));
  const raw = (await tsu('create', '--name', 'desktop')).trim();
  await store.write(raw);
  if (!(await tokenWorks(daemon.url, raw))) throw new Error('Daemon rejects the token it just created');
  return raw;
}

module.exports = { ensureLocalToken, tokenWorks };
