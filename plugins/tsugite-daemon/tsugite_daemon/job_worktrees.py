"""Provision and remove the per-job git worktree.

Shells out to `git worktree`; the rmtree fallback is deliberately confined to our
own subdirectory so a corrupted path cannot remove an arbitrary tree.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_WORKTREE_SUBDIR = ".tsugite-jobs"


def _provision_worktree(repo: str, job_id: str, workspace_root: Optional[Path] = None) -> str:
    """Add a git worktree at `<repo>/.tsugite-jobs/<job_id>` and return its absolute path.

    The worktree starts from the repo's current HEAD on a detached HEAD so the job
    can commit/branch freely without affecting the parent repo's branches.

    A relative `repo` is interpreted against `workspace_root` (the job's workspace),
    not the daemon process CWD. Absolute and `~`-expanded paths are left unchanged.
    """
    repo_path = Path(repo).expanduser()
    if not repo_path.is_absolute() and workspace_root is not None:
        repo_path = Path(workspace_root) / repo_path
    repo_path = repo_path.resolve()
    if not (repo_path / ".git").exists():
        raise ValueError(f"repo path is not a git repository: {repo_path}")
    target = repo_path / _WORKTREE_SUBDIR / job_id
    target.parent.mkdir(parents=True, exist_ok=True)
    # --detach: don't create a branch; the job can branch later if it wants.
    # HEAD: start from the repo's current commit.
    # LC_ALL=C pins git's error messages to English so log scrapes are deterministic.
    # GIT_TERMINAL_PROMPT=0 prevents git from blocking on credential prompts.
    env = {**os.environ, "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"}
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(target), "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Surface git's actual fatal message instead of the bare exit-status string.
        stderr_text = (e.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"git worktree add failed (exit {e.returncode}): {stderr_text or 'no stderr'}") from e
    return str(target)


def _prune_worktree(worktree_path: str) -> bool:
    """Remove a previously-provisioned worktree, returning True when the tree is
    gone. Errors are logged, not raised - cleanup must not fail a Job finalization
    - but the caller needs the outcome to know whether the path is still live.

    Safety: the rmtree fallback REQUIRES the path to live under our own
    `.tsugite-jobs/` subdir, so a corrupted or hand-edited worktree_path
    (e.g. an absolute path pointing at the repo root) cannot rm the wrong tree.

    `git worktree remove` needs to run inside a git repo, otherwise it errors with
    "not a git repository". `<repo>/.tsugite-jobs/<job_id>` is the layout so the
    parent's parent is the repo root - feed that as cwd. If the rmtree fallback
    fires, also run `git worktree prune` to clear the stale metadata so the next
    `worktree add` at the same path doesn't see a "missing but already registered"
    record.
    """
    wt = Path(worktree_path)
    if not wt.exists():
        return True
    # The worktree path is `<repo>/.tsugite-jobs/<job_id>` - walk up two levels for the repo.
    repo_root = wt.parent.parent if wt.parent.name == _WORKTREE_SUBDIR else None
    # `git worktree remove --force` works even if the worktree has uncommitted changes.
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(wt)],
            check=True,
            capture_output=True,
            cwd=str(repo_root) if repo_root else None,
        )
        return True
    except Exception as e:
        stderr = getattr(e, "stderr", b"")
        stderr_text = (
            stderr.decode("utf-8", "replace").strip() if isinstance(stderr, (bytes, bytearray)) else str(stderr)
        )
        logger.warning("git worktree remove failed for %s: %s (stderr: %s); attempting rmtree", wt, e, stderr_text)

    # rmtree fallback - but ONLY if the path is structurally inside .tsugite-jobs/.
    # Any other location indicates corruption or tampering; refuse rather than
    # nuke an arbitrary tree.
    resolved = wt.resolve()
    if _WORKTREE_SUBDIR not in resolved.parts:
        logger.error(
            "Refusing rmtree of %s: path is not inside %s/ - corrupted Job.worktree_path?",
            resolved,
            _WORKTREE_SUBDIR,
        )
        return False
    import shutil

    try:
        shutil.rmtree(wt, ignore_errors=True)
    except Exception:
        logger.exception("Failed to rmtree worktree at %s", wt)

    # Tell git the worktree is gone so a subsequent `worktree add` at the same path
    # doesn't trip on a stale registration.
    if repo_root and repo_root.exists():
        try:
            subprocess.run(
                ["git", "worktree", "prune"],
                check=False,
                capture_output=True,
                cwd=str(repo_root),
            )
        except Exception:
            logger.debug("git worktree prune fallback failed for %s", repo_root)
    return not wt.exists()
