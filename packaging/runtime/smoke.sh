#!/bin/sh
# Usage: packaging/runtime/smoke.sh <runtime-dir>
# Copies the runtime elsewhere with the original moved aside, starts the daemon with temp config and HOME, and checks /api/health and /.
set -eu

src=$(cd "$1" && pwd -P)
away="$src.smoke-away"
[ ! -e "$away" ] || { echo "$away already exists" >&2; exit 1; }
tmp=$(mktemp -d)
pid=
moved=
cleanup() {
    [ -n "$pid" ] && kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
    if [ -n "$moved" ]; then
        if [ -e "$src" ]; then echo "not restoring $away: $src exists" >&2; else mv "$away" "$src"; fi
    fi
    rm -rf "$tmp"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

mv "$src" "$away"
moved=1
cp -R "$away" "$tmp/moved-runtime"
mkdir -p "$tmp/home" "$tmp/state" "$tmp/workspace"
port=$("$tmp/moved-runtime/bin/python3" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
cat > "$tmp/daemon.yaml" <<YAML
state_dir: $tmp/state
default_workspace_dir: $tmp/workspace
http:
  enabled: true
  host: 127.0.0.1
  port: $port
YAML

HOME="$tmp/home" XDG_CONFIG_HOME="$tmp/home/.config" XDG_DATA_HOME="$tmp/home/.local/share" \
    XDG_STATE_HOME="$tmp/home/.local/state" XDG_CACHE_HOME="$tmp/home/.cache" \
    "$tmp/moved-runtime/bin/tsu" daemon --config "$tmp/daemon.yaml" > "$tmp/daemon.log" 2>&1 &
pid=$!

url="http://127.0.0.1:$port"
for _ in $(seq 60); do
    if health=$(curl -sf "$url/api/health"); then break; fi
    kill -0 "$pid" 2>/dev/null || { cat "$tmp/daemon.log"; echo "daemon exited early" >&2; exit 1; }
    sleep 1
done
[ -n "${health:-}" ] || { cat "$tmp/daemon.log"; echo "no /api/health response" >&2; exit 1; }
echo "health: $health"

page=$(curl -sf "$url/")
echo "$page" | grep -qi '<!doctype html' || { echo "/ is not HTML: $page" >&2; exit 1; }
echo "$page" | grep -q '<script' || { echo "/ has no script tag" >&2; exit 1; }
echo "web UI: ok ($(echo "$page" | wc -c) bytes)"

kill -TERM "$pid"
wait "$pid" && status=0 || status=$?
pid=
echo "daemon exit status: $status"
[ "$status" -eq 0 ] || { cat "$tmp/daemon.log"; exit 1; }
echo "smoke: ok"
