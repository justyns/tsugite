#!/bin/sh
# Usage: packaging/runtime/check-macos.sh <runtime-dir>
# Static checks for a macos-aarch64 runtime built on another host.
set -eu

root=$(cd "$1" && pwd -P)
fail=0
bad() { echo "FAIL: $*" >&2; fail=1; }

for f in bin/tsu bin/python3 VERSION; do
    [ -e "$root/$f" ] || bad "missing $f"
done
for f in bin/tsu bin/python3; do
    [ -x "$root/$f" ] || bad "$f is not executable"
done

native=$(find "$root" -type f \( -name '*.so' -o -name '*.dylib' \) | wc -l)
# macOS `file` prints one line per slice of a universal binary.
nonarm=$(find "$root" -type f \( -name '*.so' -o -name '*.dylib' \) | while read -r f; do
    file -b "$f" | grep -q 'Mach-O 64-bit.*arm64' || echo "$f"
done)
echo "native libraries: $native"
[ -z "$nonarm" ] || bad "non-arm64 native libraries: $nonarm"

elf=$(find "$root" -type f -exec file {} + | grep -c 'ELF ' || true)
echo "ELF files: $elf"
[ "$elf" -eq 0 ] || { bad "ELF files present:"; find "$root" -type f -exec file {} + | grep 'ELF ' >&2; }

build_root=$(cd "$(dirname "$0")/../.." && pwd -P)
leaks=$(grep -rIl -e "$build_root" -e "$HOME" "$root/bin/tsu" "$root"/lib/python3.12/site-packages/*.pth 2>/dev/null || true)
pth=$(ls "$root"/lib/python3.12/site-packages/*.pth 2>/dev/null | wc -l)
if [ -z "$leaks" ]; then
    echo "launcher and .pth files: $pth .pth, no build host paths"
else
    bad "build host paths in: $leaks"
fi

echo "size: $(du -sh "$root" | cut -f1)"
[ "$fail" -eq 0 ] && echo "check: ok"
exit "$fail"
