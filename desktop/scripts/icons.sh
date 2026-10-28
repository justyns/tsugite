#!/bin/sh
# Renders the desktop app and web UI icons from the SVGs in resources/logo/. Needs rsvg-convert (librsvg).
set -eu
cd "$(dirname "$0")/.."
logo=resources/logo
web=../plugins/tsugite-daemon/frontend/public/icons
mkdir -p resources/icons src/assets "$web"
rsvg-convert -w 1024 -h 1024 -o resources/icon.png "$logo/macos.svg"
for size in 16 32; do
  rsvg-convert -w "$size" -h "$size" -o "resources/icons/${size}x${size}.png" "$logo/tile-small.svg"
done
for size in 48 64 128 256 512; do
  rsvg-convert -w "$size" -h "$size" -o "resources/icons/${size}x${size}.png" "$logo/tile.svg"
done
cp resources/icons/512x512.png src/assets/icon.png
rsvg-convert -w 32 -h 32 -o "$web/favicon-32.png" "$logo/tile-small.svg"
rsvg-convert -w 192 -h 192 -o "$web/icon-192.png" "$logo/tile.svg"
rsvg-convert -w 512 -h 512 -o "$web/icon-512.png" "$logo/tile.svg"
rsvg-convert -w 512 -h 512 -o "$web/icon-512-maskable.png" "$logo/maskable.svg"
rsvg-convert -w 180 -h 180 -o "$web/apple-touch-icon.png" "$logo/square.svg"
