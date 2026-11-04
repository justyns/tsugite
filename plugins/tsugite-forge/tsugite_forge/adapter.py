"""The forge plugin's daemon adapter, which polls each forge and caches its open pull requests."""

import asyncio
import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from tsugite_daemon.adapters.base import BaseAdapter

from tsugite.link_previews import register_link_resolver
from tsugite.secrets.registry import get_registry
from tsugite_forge import previews, refs
from tsugite_forge.cache import LruCache
from tsugite_forge.config import REPO_PATTERN, Forge, ForgePluginConfig
from tsugite_forge.providers import ForgeError, Provider, PullRequest, kinds
from tsugite_forge.tools import INCLUDE_PARTS, set_forge_adapter

MAX_BACKOFF_SECONDS = 3600
# Seconds a checkout's origin URL is reused.
ORIGIN_TTL = 300
# Seconds live checks and review threads are reused for one head commit.
DETAIL_TTL = 60
CACHE_ENTRIES = 256

logger = logging.getLogger(__name__)


@dataclass
class ForgeCache:
    config: Forge
    provider: Provider
    repos: list[str] = field(default_factory=list)
    prs: list[PullRequest] = field(default_factory=list)
    last_refresh: float | None = None
    error: str | None = None
    failures: int = 0
    next_attempt: float = 0.0
    viewer: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _describe(exc: Exception) -> str:
    return get_registry().mask(str(exc) if isinstance(exc, ForgeError) else f"{type(exc).__name__}: {exc}")


def _iso(timestamp: float | None) -> str | None:
    return None if timestamp is None else datetime.fromtimestamp(timestamp, UTC).isoformat()


class ForgeAdapter(BaseAdapter):
    def __init__(self, config: ForgePluginConfig, clock=time.time, session_store=None):
        # Not agent-scoped, so BaseAdapter.__init__ is skipped. The gateway
        # overwrites event_bus with the SSE broadcaster.
        self.config = config
        self.event_bus = None
        self.session_store = session_store
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clock = clock
        self.interval = config.poll_minutes * 60
        self.forges = {forge.name: ForgeCache(forge, kinds[forge.kind](forge)) for forge in config.forges}
        self._pollers: list[asyncio.Task] = []
        self._derived = LruCache(CACHE_ENTRIES)
        self._origins = LruCache(CACHE_ENTRIES)
        self._details = LruCache(CACHE_ENTRIES)

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._pollers = [asyncio.create_task(self._poll(cache)) for cache in self.forges.values()]
        set_forge_adapter(self)
        logger.info("forge adapter started (%s)", ", ".join(self.forges))

    async def stop(self) -> None:
        set_forge_adapter(None)
        for task in self._pollers:
            task.cancel()
        for task in self._pollers:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pollers = []
        logger.info("forge adapter stopped")

    async def _poll(self, cache: ForgeCache) -> None:
        while True:
            if self.clock() >= cache.next_attempt:
                await self._refresh(cache)
            await asyncio.sleep(max(cache.next_attempt - self.clock(), 1))

    async def refresh(self) -> None:
        await asyncio.gather(*(self._refresh(cache) for cache in self.forges.values()))

    async def _refresh(self, cache: ForgeCache) -> None:
        async with cache.lock:
            try:
                repos = await asyncio.to_thread(cache.provider.list_repos)
                prs = await asyncio.to_thread(cache.provider.list_prs, repos)
            # Catch everything. An error stays on its forge and must not end the poller.
            except Exception as exc:
                cache.error = _describe(exc)
                cache.failures += 1
                backoff = self.interval * 2**cache.failures
                cache.next_attempt = self.clock() + min(backoff, max(MAX_BACKOFF_SECONDS, self.interval))
                logger.warning(
                    "forge %s refresh failed (%d in a row): %s", cache.config.name, cache.failures, cache.error
                )
                return
            cache.repos, cache.prs = repos, prs
            cache.last_refresh = self.clock()
            cache.error = None
            cache.failures = 0
            cache.next_attempt = cache.last_refresh + self.interval
            if cache.viewer is None:
                try:
                    cache.viewer = await asyncio.to_thread(cache.provider.viewer)
                except Exception as exc:
                    cache.error = f"viewer lookup failed: {_describe(exc)}"
                    logger.warning("forge %s %s", cache.config.name, cache.error)

    def prs(
        self,
        state: str = "open",
        repo: str | None = None,
        author: str | None = None,
        label: str | None = None,
        needs_review: bool | None = None,
        forge: str | None = None,
    ) -> list[PullRequest]:
        """Cached pull requests matching every given filter, most recently updated first."""
        if state != "open":
            raise ValueError(f"state must be 'open', got {state!r}. Only open pull requests are cached.")
        if repo and ":" in repo:
            forge, repo = repo.split(":", 1)
        if forge is not None:
            self.forge(forge)
        matches = [
            pr
            for cache in self.forges.values()
            if forge in (None, cache.config.name)
            for pr in cache.prs
            if repo in (None, pr.repo)
            and author in (None, pr.author)
            and (label is None or label in pr.labels)
            and (needs_review is None or (cache.viewer in pr.requested_reviewers) == needs_review)
        ]
        return sorted(matches, key=lambda pr: pr.updated_at, reverse=True)

    def resolve(self, spec: str) -> tuple[ForgeCache, str]:
        """The one forge whose cached repos include `owner/repo`, or `forge:owner/repo`."""
        forge, repo = spec.split(":", 1) if ":" in spec else (None, spec)
        matches = [
            cache for cache in self.forges.values() if forge in (None, cache.config.name) and repo in cache.repos
        ]
        if forge is not None:
            self.forge(forge)
        if not matches:
            raise ValueError(f"{repo} is not in any configured forge's repos (forge_repos lists them)")
        if len(matches) > 1:
            options = ", ".join(f"{cache.config.name}:{repo}" for cache in matches)
            raise ValueError(f"{repo} is on more than one forge, pass one of {options}")
        return matches[0], repo

    def pinned(self, session_id: str, spec: str, pr: int | None) -> tuple[str, str]:
        """The forge and repo of the session's pin on `owner/repo` or `forge:owner/repo` and `pr`."""
        forge, repo = spec.split(":", 1) if ":" in spec else (None, spec)
        stored = refs.stored_refs(self._session(session_id).metadata.get("refs"))
        forges = sorted(
            {
                e["forge"]
                for e in stored
                if e.get("source") == "pinned" and (e["repo"], e["pr"]) == (repo, pr) and forge in (None, e["forge"])
            }
        )
        if not forges:
            raise ValueError(f"{spec}{'' if pr is None else f'#{pr}'} is not pinned to this session")
        if len(forges) > 1:
            options = ", ".join(f"{name}:{repo}" for name in forges)
            raise ValueError(f"{repo} is pinned on more than one forge, pass one of {options}")
        return forges[0], repo

    def forge(self, name: str) -> ForgeCache:
        if name not in self.forges:
            raise ValueError(f"no forge named {name!r}, configured: {', '.join(self.forges)}")
        return self.forges[name]

    def pr_detail(self, forge: str, repo: str, number: int, include=INCLUDE_PARTS) -> dict:
        """One pull request, plus the `include` parts read live from the forge. Blocks on forge calls."""
        unknown = set(include) - set(INCLUDE_PARTS)
        if unknown:
            raise ValueError(f"unknown include {', '.join(sorted(unknown))}, choose from {', '.join(INCLUDE_PARTS)}")
        cache = self.forge(forge)
        # Only a cached repo name may reach the provider's URLs.
        if repo not in cache.repos:
            raise LookupError(f"{repo} is not in forge {forge!r}'s repos")
        provider = cache.provider
        pr = next((pr for pr in cache.prs if pr.repo == repo and pr.number == number), None)
        pr = pr or provider.get_pr(repo, number)
        live = {
            "checks": lambda: provider.checks(repo, number),
            "threads": lambda: asdict(provider.review_threads(repo, number)),
        }
        detail = {"pr": asdict(pr)}
        for part in include:
            if part == "diff":
                detail[part] = provider.diff(repo, number)
                continue
            key = (forge, repo, number, pr.head_sha, part)
            hit = self._details.get(key)
            if hit is None or self.clock() - hit[1] > DETAIL_TTL:
                hit = (live[part](), self.clock())
                self._details.put(key, hit)
            detail[part] = hit[0]
        return get_registry().mask_obj(detail)

    def link_preview(self, url: str) -> dict | None:
        """Preview a pull request or issue link into a configured repo. Blocks on a forge call on a cache miss."""
        ref = refs.link_ref(url, self.config.forges)
        if ref is None:
            return None
        forge, repo, kind, number = ref
        cache = self.forges[forge]
        if repo not in cache.repos:
            return None
        is_pr = kind != "issues"
        if is_pr and (pr := next((pr for pr in cache.prs if pr.repo == repo and pr.number == number), None)):
            return previews.pr_preview(pr, cache.config)
        key = (forge, repo, kind, number)
        hit = self._details.get(key)
        if hit and self.clock() - hit[1] <= DETAIL_TTL:
            return hit[0]
        try:
            if is_pr:
                preview = previews.pr_preview(cache.provider.get_pr(repo, number), cache.config)
            else:
                preview = previews.issue_preview(cache.provider.get_issue(repo, number), cache.config)
        except ForgeError as exc:
            logger.info("forge %s has no preview for %s: %s", forge, url, exc)
            return None
        # Kept for the next request when this read outlives the daemon's resolve timeout.
        self._details.put(key, (preview, self.clock()))
        return preview

    def get_http_routes(self) -> list:
        return [
            Route("/prs", self._list_prs, methods=["GET"]),
            Route("/prs/{forge}/{owner}/{repo}/{number:int}", self._get_pr, methods=["GET"]),
            Route("/status", self._status, methods=["GET"]),
            Route("/refresh", self._refresh_now, methods=["POST"]),
            Route("/sessions/{session_id}/refs", self._get_refs, methods=["GET"]),
            Route("/sessions/{session_id}/refs", self._post_refs, methods=["POST"]),
        ]

    async def _list_prs(self, request: Request) -> JSONResponse:
        query = request.query_params
        needs_review = query.get("needs_review")
        if needs_review not in (None, "true", "false"):
            return JSONResponse({"error": "needs_review must be true or false"}, status_code=400)
        try:
            prs = self.prs(
                state=query.get("state", "open"),
                repo=query.get("repo"),
                author=query.get("author"),
                label=query.get("label"),
                needs_review=None if needs_review is None else needs_review == "true",
                forge=query.get("forge"),
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(get_registry().mask_obj({"prs": [asdict(pr) for pr in prs]}))

    async def _get_pr(self, request: Request) -> JSONResponse:
        params = request.path_params
        if params["forge"] not in self.forges:
            return JSONResponse({"error": f"no forge named {params['forge']!r}"}, status_code=404)
        include = request.query_params.get("include")
        parts = INCLUDE_PARTS if include is None else [part for part in include.split(",") if part]
        repo = f"{params['owner']}/{params['repo']}"
        try:
            detail = await asyncio.to_thread(self.pr_detail, params["forge"], repo, params["number"], parts)
        except LookupError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except ForgeError as exc:
            return JSONResponse({"error": get_registry().mask(str(exc))}, status_code=502)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(detail)

    async def _get_refs(self, request: Request) -> JSONResponse:
        try:
            rows = await asyncio.to_thread(self.session_refs, request.path_params["session_id"])
        except LookupError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse({"refs": rows})

    async def _post_refs(self, request: Request) -> JSONResponse:
        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(body, dict) or "pr" not in body:
            return JSONResponse({"error": "body needs action, forge, repo and pr"}, status_code=400)
        try:
            self.update_refs(session_id, body.get("action"), body.get("forge"), body.get("repo"), body["pr"])
        except LookupError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return await self._get_refs(request)

    async def _status(self, request: Request) -> JSONResponse:
        return JSONResponse({"forges": self.status()})

    async def _refresh_now(self, request: Request) -> JSONResponse:
        await self.refresh()
        return JSONResponse({"forges": self.status()})

    def _session(self, session_id: str):
        session = self.session_store.resolve_live(session_id)
        if session is None:
            raise LookupError(f"no session {session_id!r}")
        return session

    def _resolve_quietly(self, spec: str) -> tuple[str, str] | None:
        try:
            cache, repo = self.resolve(spec)
        except ValueError:
            return None
        return cache.config.name, repo

    def session_refs(self, session_id: str) -> list[dict]:
        """The session's pinned, touched and mentioned refs. Blocks on history reads and git."""
        session = self._session(session_id)
        stored = refs.stored_refs(session.metadata.get("refs"))
        pinned_repos = {(e["forge"], e["repo"]) for e in stored if not e.get("hidden")}
        bare_repo = next(iter(pinned_repos)) if len(pinned_repos) == 1 else None
        chain = refs.session_chain(session.id)
        # A refresh can change which repos a forge tool call in history resolves to.
        refreshed = tuple(cache.last_refresh for cache in self.forges.values())
        key = (refs.history_key(chain), bare_repo, refreshed)
        hit = self._derived.get(session.id)
        if hit is None or hit[0] != key:
            events = refs.load_histories(chain)
            hit = (key, refs.derive(events, self.config.forges, self._resolve_quietly, bare_repo, self._origin))
            self._derived.put(session.id, hit)
        derived = hit[1]
        pulls = {(pr.forge, pr.repo, pr.number): asdict(pr) for cache in self.forges.values() for pr in cache.prs}
        return get_registry().mask_obj(refs.merge(stored, derived, pulls))

    def _origin(self, root) -> str | None:
        hit = self._origins.get(root)
        if hit is None or self.clock() - hit[1] > ORIGIN_TTL:
            hit = (refs.git_origin(root), self.clock())
            self._origins.put(root, hit)
        return hit[0]

    def update_refs(self, session_id: str, action: str, forge: str, repo: str, pr: int | None):
        """Apply a pin, unpin, hide or unhide to the session's stored refs. Runs on the daemon loop."""
        if not isinstance(forge, str) or forge not in self.forges:
            raise ValueError(f"no forge named {forge!r}")
        if not isinstance(repo, str) or not REPO_PATTERN.fullmatch(repo) or repo.endswith("/*"):
            raise ValueError(f"repo {repo!r} must be owner/repo")
        if pr is not None and (not isinstance(pr, int) or isinstance(pr, bool)):
            raise ValueError("pr must be a number or null")
        session = self._session(session_id)
        stored = refs.apply_action(refs.stored_refs(session.metadata.get("refs")), action, forge, repo, pr)
        session = self.session_store.set_metadata_bulk(session.id, {"refs": stored})
        if self.event_bus is not None:
            self.event_bus.emit(
                "session_update", {"action": "metadata_updated", "id": session.id, "metadata": session.metadata}
            )

    def status(self) -> list[dict]:
        now = self.clock()
        return get_registry().mask_obj(
            [
                {
                    "name": cache.config.name,
                    "kind": cache.config.kind,
                    "ok": cache.last_refresh is not None and cache.error is None,
                    "last_refresh": _iso(cache.last_refresh),
                    "cache_age_seconds": None if cache.last_refresh is None else int(now - cache.last_refresh),
                    "error": cache.error,
                    "next_attempt": _iso(cache.next_attempt),
                    "viewer": cache.viewer,
                }
                for cache in self.forges.values()
            ]
        )


def create_adapter(*, config, runtime, session_store, identity_map):
    """Adapter-plugin factory (the tsugite.adapters entry point)."""
    if not config.get("forges"):
        logger.info("forge plugin installed but no forges configured (add plugins.forge.forges to activate)")
        return None
    adapter = ForgeAdapter(ForgePluginConfig(**config), session_store=session_store)
    register_link_resolver(
        name="links",
        hosts=sorted({refs.forge_host(forge) for forge in adapter.config.forges}),
        path=previews.LINK_PATH,
        resolve=adapter.link_preview,
    )
    return adapter
