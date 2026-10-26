const fs = require('node:fs');
const path = require('node:path');

// TSUGITE_RUNTIME_SRC replaces the runtime directory for either platform (used to test packaging with a stub).
const runtimeFrom = (target) => process.env.TSUGITE_RUNTIME_SRC || `../dist/runtime/${target}`;
const TARGET = { darwin: 'macos-aarch64', linux: 'linux-x86_64' };
const extraResources = (platform) => [{ from: runtimeFrom(TARGET[platform]), to: 'runtime' }];

module.exports = {
  appId: 'io.tsugite.desktop',
  productName: 'Tsugite',
  directories: { output: 'out' },
  files: ['src/**/*', 'package.json'],
  mac: {
    target: [
      { target: 'zip', arch: ['arm64'] },
      { target: 'dir', arch: ['arm64'] },
    ],
    identity: null,
    category: 'public.app-category.developer-tools',
    extraResources: extraResources('darwin'),
  },
  linux: {
    target: [
      { target: 'AppImage', arch: ['x64'] },
      { target: 'dir', arch: ['x64'] },
    ],
    category: 'Development',
    extraResources: extraResources('linux'),
  },
  beforePack: async (context) => {
    const target = TARGET[context.electronPlatformName];
    const dir = path.resolve(__dirname, runtimeFrom(target));
    if (!fs.existsSync(path.join(dir, 'bin', 'tsu'))) {
      throw new Error(`Runtime missing: ${path.join(dir, 'bin', 'tsu')} not found. Build it first (packaging/runtime/build.py --target ${target}) or set TSUGITE_RUNTIME_SRC.`);
    }
  },
};
