const fs = require('node:fs');
const path = require('node:path');

const DEFAULTS = { mode: 'local', remoteUrl: '', bounds: { width: 1280, height: 860 } };
// v10 marks Chromium's hardcoded key on Linux (no Secret Service running) but is the normal Keychain prefix on macOS.
const LINUX_FALLBACK_PREFIX = 'v10';
class InsecureStorage extends Error {}
const isLoopback = (host) => host === 'localhost' || host === '[::1]' || /^127(\.\d+){3}$/.test(host);

class Settings {
  constructor(dir, safeStorage, platform = process.platform) {
    this.file = path.join(dir, 'settings.json');
    this.remoteTokenFile = path.join(dir, 'remote-token.bin');
    this.localTokenFile = path.join(dir, 'local-token.bin');
    fs.rmSync(path.join(dir, 'local-token'), { force: true });
    this.safeStorage = safeStorage;
    this.platform = platform;
    this.data = { ...DEFAULTS, ...(fs.existsSync(this.file) ? JSON.parse(fs.readFileSync(this.file, 'utf8')) : {}) };
  }

  update(patch) {
    Object.assign(this.data, patch);
    fs.writeFileSync(`${this.file}.tmp`, JSON.stringify(this.data, null, 2));
    fs.renameSync(`${this.file}.tmp`, this.file);
  }

  async encrypt(token) {
    if (!(await this.safeStorage.isAsyncEncryptionAvailable())) throw new InsecureStorage('OS secret storage is unavailable, cannot store the token');
    const encrypted = await this.safeStorage.encryptStringAsync(token);
    if (this.platform === 'linux' && encrypted.subarray(0, 3).toString('latin1') === LINUX_FALLBACK_PREFIX) {
      throw new InsecureStorage('No Secret Service is running, so the token would only be obfuscated. Start gnome-keyring, KeePassXC or KWallet, then try again.');
    }
    return encrypted;
  }

  async writeToken(file, token) {
    fs.writeFileSync(file, await this.encrypt(token), { mode: 0o600 });
    fs.chmodSync(file, 0o600);
  }

  async readToken(file) {
    const { result, shouldReEncrypt } = await this.safeStorage.decryptStringAsync(fs.readFileSync(file));
    if (shouldReEncrypt) await this.writeToken(file, result);
    return result;
  }

  localTokenStore(onMemoryOnly) {
    let memory = '';
    return {
      read: async () => memory || (fs.existsSync(this.localTokenFile) ? this.readToken(this.localTokenFile) : ''),
      write: async (token) => {
        memory = token;
        try {
          await this.writeToken(this.localTokenFile, token);
        } catch (err) {
          if (!(err instanceof InsecureStorage)) throw err;
          onMemoryOnly(err.message);
        }
      },
    };
  }

  hasRemote() {
    return Boolean(this.data.remoteUrl) && fs.existsSync(this.remoteTokenFile);
  }

  async setRemote(url, token) {
    const parsed = new URL(url);
    if (parsed.protocol !== 'https:' && !(parsed.protocol === 'http:' && isLoopback(parsed.hostname))) {
      throw new Error('URL must start with https:// (http:// is allowed only for localhost)');
    }
    await this.writeToken(this.remoteTokenFile, token);
    this.update({ mode: 'remote', remoteUrl: parsed.origin });
  }

  remoteToken() {
    return this.readToken(this.remoteTokenFile);
  }
}

module.exports = { Settings };