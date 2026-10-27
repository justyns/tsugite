#!/bin/sh
# Renders the desktop app icons from the SVGs in resources/logo/. Needs rsvg-convert (librsvg).
set -eu
cd "$(dirname "$0")/.."
logo=resources/logo
mkdir -p resources/icons src/assets
rsvg-convert -w 1024 -h 1024 -o resources/icon.png "$logo/macos.svg"
for size in 16 32; do
  rsvg-convert -w "$size" -h "$size" -o "resources/icons/${size}x${size}.png" "$logo/tile-small.svg"
done
for size in 48 64 128 256 512; do
  rsvg-convert -w "$size" -h "$size" -o "resources/icons/${size}x${size}.png" "$logo/tile.svg"
done
cp resources/icons/512x512.png src/assets/icon.png