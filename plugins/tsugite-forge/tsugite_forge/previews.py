"""Link previews for pull requests and issues, in the shape `tsugite.link_previews` resolvers return."""

from tsugite_forge.config import Forge
from tsugite_forge.providers import Issue, PullRequest
from tsugite_forge.refs import LINK_KINDS, forge_host

LINK_PATH = rf"/({'|'.join(sorted({segment for segments in LINK_KINDS.values() for segment in segments}))})/\d+/?$"

STATE_TONES = {"open": "ok", "draft": "mute", "merged": "info", "closed": "mute"}
CI_TONES = {"success": "ok", "failure": "err", "pending": "warn"}
REVIEW_TONES = {"approved": "ok", "changes_requested": "warn"}
KIND_NAMES = {"forgejo": "Forgejo", "github": "GitHub", "bitbucket": "Bitbucket"}


def _label(repo: str, number: int) -> str:
    return f"{repo.split('/')[1]}#{number}"


def _state(state: str) -> dict:
    return {"text": state, "tone": STATE_TONES[state]}


def _icon(forge: Forge) -> str:
    return f"/api/plugins/forge/ui/icons/{forge.kind}.svg"


def _forge_field(forge: Forge) -> dict:
    return {"label": "forge", "value": f"{KIND_NAMES[forge.kind]}, {forge_host(forge)}"}


def pr_preview(pr: PullRequest, forge: Forge) -> dict:
    """Shows CI on the chip while the pull request is open, else its state."""
    state = _state("draft" if pr.draft and pr.state == "open" else pr.state)
    ci = {"text": f"CI {pr.ci_state}", "tone": CI_TONES[pr.ci_state]} if pr.ci_state != "none" else None
    badges = [state, *([ci] if ci else [])]
    if pr.unresolved_threads:
        badges.append({"text": f"{pr.unresolved_threads} unresolved", "tone": "warn"})
    if pr.review_state != "none":
        badges.append({"text": pr.review_state.replace("_", " "), "tone": REVIEW_TONES[pr.review_state]})
    badges += [{"text": label, "tone": "mute"} for label in pr.labels]
    return {
        "label": _label(pr.repo, pr.number),
        "title": pr.title,
        "url": pr.url,
        "status": ci if ci and pr.state == "open" else state,
        "badges": badges,
        "fields": [
            _forge_field(forge),
            {"label": "repo", "value": pr.repo},
            {"label": "author", "value": pr.author},
            {"label": "branch", "value": f"{pr.base_ref} <- {pr.head_ref}"},
            {"label": "updated", "value": pr.updated_at},
        ],
        "icon": _icon(forge),
    }


def issue_preview(issue: Issue, forge: Forge) -> dict:
    state = _state(issue.state)
    return {
        "label": _label(issue.repo, issue.number),
        "title": issue.title,
        "url": issue.url,
        "status": state,
        "badges": [state, *({"text": label, "tone": "mute"} for label in issue.labels)],
        "fields": [
            _forge_field(forge),
            {"label": "repo", "value": issue.repo},
            {"label": "author", "value": issue.author},
        ],
        "icon": _icon(forge),
    }
