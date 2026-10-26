#!/usr/bin/env python3
"""Build a relocatable tsugite runtime: python-build-standalone CPython plus tsugite wheels.

Output: dist/runtime/<target>/ with bin/tsu, bin/python3 and VERSION.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / ".cache"
PBS_RELEASE = "20261021"
PBS_PYTHON = "3.12.15"
PBS_URL = "https://github.com/astral-sh/python-build-standalone/releases/download"

TARGETS = {
    "macos-aarch64": {
        "triple": "aarch64-apple-darwin",
        "uv_platform": "aarch64-apple-darwin",
        "tag_platform": "macosx",
        "tag_arch": ("arm64", "universal2"),
        "sha256": "a18cc33862b53bd54a148c710def5375c1b47470cccedf817a19c2d211764584",
    },
    "linux-x86_64": {
        "triple": "x86_64-unknown-linux-gnu",
        "uv_platform": "x86_64-unknown-linux-gnu",
        "tag_platform": "manylinux",
        "tag_arch": ("x86_64",),
        "sha256": "5ff9f0602ce964a529ed1e8555d3abab6341926d724402ba9a15e9e877191124",
    },
}

PLUGINS = [
    "tsugite-daemon",
    "tsugite-web",
    "tsugite-youtube",
    "tsugite-pty",
    "tsugite-sandbox",
    "tsugite-claude-code",
    "tsugite-codex-cli",
    "tsugite-cc-driver",
]

# Pure-Python packages that PyPI ships only as sdists; check_wheel_tags rejects a compiled result.
SDIST_ONLY = "http-ece"

LAUNCHER = """#!/bin/sh
root=$(cd "$(dirname "$0")/.." && pwd -P)
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec "$root/bin/python3" -c 'from tsugite.tsugite import app; app()' "$@"
"""


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def fetch_interpreter(target):
    spec = TARGETS[target]
    name = f"cpython-{PBS_PYTHON}+{PBS_RELEASE}-{spec['triple']}-install_only.tar.gz"
    path = CACHE / name
    CACHE.mkdir(exist_ok=True)
    if not path.exists():
        url = f"{PBS_URL}/{PBS_RELEASE}/{name.replace('+', '%2B')}"
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        path.unlink()
        sys.exit(f"sha256 mismatch for {name}: got {digest}, pinned {spec['sha256']}")
    return path


def build_frontend():
    frontend = ROOT / "plugins/tsugite-daemon/frontend"
    run(["npm", "--prefix", frontend, "ci"])
    run(["npm", "--prefix", frontend, "run", "build"])


def build_wheels(out):
    projects = [ROOT] + [ROOT / "plugins" / p for p in PLUGINS]
    for project in projects:
        run(["uv", "build", "--wheel", "--out-dir", out, project])
    wheels = sorted(out.glob("*.whl"))
    daemon = next(w for w in wheels if w.name.startswith("tsugite_daemon-"))
    if "tsugite_daemon/web/index.html" not in zipfile.ZipFile(daemon).namelist():
        sys.exit(f"{daemon.name} does not contain tsugite_daemon/web/index.html")
    return wheels


def drop_build_paths(site):
    """Remove site-packages/bin (build-host shebangs) and direct_url.json (local wheel paths) with its RECORD line."""
    shutil.rmtree(site / "bin", ignore_errors=True)
    for direct_url in site.glob("*.dist-info/direct_url.json"):
        direct_url.unlink()
        record = direct_url.with_name("RECORD")
        lines = [ln for ln in record.read_text().splitlines() if "direct_url.json" not in ln]
        record.write_text("\n".join(lines) + "\n")


def check_wheel_tags(site, target):
    """Every installed distribution is pure Python or built for the target platform."""
    spec = TARGETS[target]
    for wheel_file in site.glob("*.dist-info/WHEEL"):
        for line in wheel_file.read_text().splitlines():
            if not line.startswith("Tag: ") or line.endswith("-none-any"):
                continue
            platform = line.rsplit("-", 1)[1]
            if not (
                platform.startswith(spec["tag_platform"])
                and platform.endswith(tuple("_" + a for a in spec["tag_arch"]))
            ):
                sys.exit(f"{wheel_file.parent.name}: {line} does not match {target}")


def precompile(site):
    """Host Python 3.12 writes the same pyc format; unchecked-hash pycs never revalidate against mtimes, and -s keeps the build path out of them."""
    run(
        [
            "uv", "run", "--no-project", "--python", "3.12", "python", "-m", "compileall",
            "-q", "-s", site, "--invalidation-mode", "unchecked-hash", site,
        ]
    )  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    target = parser.parse_args().target
    spec = TARGETS[target]

    archive = fetch_interpreter(target)
    out = ROOT / "dist/runtime" / target
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data", members=[m for m in tar.getmembers() if m.name != "python"])
    for child in (out / "python").iterdir():
        child.rename(out / child.name)
    (out / "python").rmdir()

    site = out / "lib/python3.12/site-packages"
    build_frontend()
    with tempfile.TemporaryDirectory() as tmp:
        wheels = build_wheels(Path(tmp))
        pins = Path(tmp) / "uv-lock-pins.txt"
        with pins.open("w") as f:
            run(
                ["uv", "export", "--locked", "--all-extras", "--no-hashes", "--no-emit-workspace", "--no-dev", "-q"],
                cwd=ROOT,
                stdout=f,
            )
        run(
            [
                "uv", "pip", "install", "--target", site, "--python-platform", spec["uv_platform"],
                "--python-version", "3.12", "--only-binary", ":all:", "--no-binary", SDIST_ONLY, "--no-cache-dir",
                "--constraints", pins, *wheels,
            ]
        )  # fmt: skip
    check_wheel_tags(site, target)
    drop_build_paths(site)
    precompile(site)

    launcher = out / "bin/tsu"
    launcher.write_text(LAUNCHER)
    launcher.chmod(0o755)

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    sha = subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "--short=12", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    (out / "VERSION").write_text(f"{version} {sha}\n")
    meta = {"python": f"{PBS_PYTHON}+{PBS_RELEASE}", "wheels": [w.name for w in wheels]}
    print(json.dumps(meta, indent=2))
    print(f"built {out}")


if __name__ == "__main__":
    main()
