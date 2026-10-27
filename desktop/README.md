# Tsugite Desktop

Linux and macOS. Run every command from the repo root.

```sh
# Run in development (needs the built web UI)
mise run web-build
npm --prefix desktop ci
npm --prefix desktop start

# Test
npm --prefix desktop test                # unit tests against a fake daemon
npm --prefix desktop run integration     # real daemon via uv run

# Build the bundled runtime, check it, then package the app
python3 packaging/runtime/build.py --target linux-x86_64     # or macos-aarch64
packaging/runtime/smoke.sh dist/runtime/linux-x86_64
packaging/runtime/check-macos.sh dist/runtime/macos-aarch64
npm --prefix desktop run dist:linux                           # or dist:mac

# Regenerate the app icons from desktop/resources/logo/*.svg (needs rsvg-convert)
desktop/scripts/icons.sh

# Open an unsigned macOS build
xattr -dr com.apple.quarantine /Applications/Tsugite.app
codesign --force --deep --sign - /Applications/Tsugite.app
```
