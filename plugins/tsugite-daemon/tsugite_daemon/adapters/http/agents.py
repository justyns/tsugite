"""AgentsMixin: agents HTTP handlers for HTTPServer."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tsugite.attachments.delegation import can_inline_file
from tsugite_daemon.adapters.base import ChannelContext
from tsugite_daemon.adapters.http.helpers import (
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_TOTAL,
    ActiveChat,
    HTTPAgentAdapter,
    _deduplicate_dest,
    _file_handler,
    _format_upload_message_suffix,
    _resolve_full_model_id,
    _sanitize_filename,
    build_session_event_persister,
    logger,
)
from tsugite_daemon.adapters.http.sse import (
    HTTPInteractionBackend,
    SSEProgressHandler,
    resolve_pending_ask,
)


def _session_or_default(adapter: HTTPAgentAdapter, session_id: Optional[str], user_id: str):
    """Resolve an explicit session id, falling back to the user's default session."""
    session = None
    if session_id:
        try:
            session = adapter.session_store.get_session(session_id)
        except ValueError:
            session = None
    if session is None:
        session = adapter.session_store.find_default_session(user_id, adapter.agent_name)
    return session


def _load_session_events(session_id: str) -> list:
    """Load a session's events as Event objects, the same source resume reads.

    Reconstruction needs Event objects (`.type`/`.data`/`.ts`), not the dicts
    session_store.read_events yields. The history backend's iter_events is what
    load_conversation_messages (the resume path) consumes, so rebuilding from it
    reproduces exactly what the model saw.
    """
    from tsugite.history import get_history_backend

    backend = get_history_backend()
    if not backend.exists(session_id):
        return []
    return list(backend.load(session_id).iter_events())


class AgentsMixin:
    def _agent_routes(self) -> list:
        return [
            Route("/api/agents/{agent}/sessions", self._list_sessions, methods=["GET"]),
            Route("/api/agents/{agent}/sessions/new", self._new_interactive_session, methods=["POST"]),
            Route("/api/agents/{agent}/sessions/{session_id}/branch", self._branch, methods=["POST"]),
            Route("/api/agents/{agent}/chat", self._chat, methods=["POST"]),
            Route("/api/agents/{agent}/chat/cancel", self._cancel_chat, methods=["POST"]),
            Route("/api/agents/{agent}/upload", self._upload, methods=["POST"]),
            Route("/api/agents/{agent}/status", self._status, methods=["GET"]),
            Route("/api/agents/{agent}/attachments", self._attachments, methods=["GET"]),
            Route("/api/agents/{agent}/history", self._history, methods=["GET"]),
            Route("/api/agents/{agent}/prompt-snapshot", self._prompt_snapshot, methods=["GET"]),
            Route("/api/agents/{agent}/raw-messages", self._raw_messages, methods=["GET"]),
            Route("/api/agents/{agent}/config", self._update_agent_config, methods=["PATCH"]),
            Route("/api/agents/{agent}/compact", self._compact, methods=["POST"]),
            Route("/api/agents/{agent}/respond", self._respond, methods=["POST"]),
            Route("/api/agents/{agent}/unload-skill", self._unload_skill, methods=["POST"]),
            Route("/api/agents/{agent}/effort-levels", self._effort_levels, methods=["GET"]),
            Route("/api/agents/{agent}/commands/{command_name}", self._run_command, methods=["POST"]),
        ]

    def _permissions_runtime_path(self) -> Path:
        """The mutable permissions.yaml that "Always allow" writes to.

        Kept beside daemon.yaml so it survives restarts; falls back to the XDG
        write location when the daemon was started without an explicit config
        path (the same place a default daemon.yaml would be written)."""
        if self.gateway and self.gateway.config_path:
            return Path(self.gateway.config_path).parent / "permissions.yaml"
        from tsugite.config import get_xdg_write_path

        return get_xdg_write_path("permissions.yaml")

    async def _run_command(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        from tsugite_daemon.commands import get_commands

        command_name = request.path_params["command_name"]
        commands = get_commands()
        if command_name not in commands:
            return JSONResponse({"error": f"Unknown command: {command_name}"}, status_code=404)

        cmd = commands[command_name]
        try:
            body = await request.json()
        except Exception:
            body = {}

        allowed_keys = {p.name for p in cmd.params}
        filtered = {k: v for k, v in body.items() if k in allowed_keys}

        missing = [p.name for p in cmd.params if p.required and p.name not in filtered]
        if missing:
            return JSONResponse({"error": f"Missing required params: {', '.join(missing)}"}, status_code=400)

        from tsugite_daemon.commands import CommandError

        try:
            result = await cmd.handler(adapter, **filtered)
        except CommandError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse({"result": result})

    def _session_busy(self, agent_name: str, session) -> bool:
        """The one definition of busy, shared by the sessions payload, /status,
        and the /chat 409 guard - the server must never 409 a send while
        reporting the session idle. True when the store reports durable live
        work (an in-flight turn, or a background/scheduled run) OR a live HTTP
        chat task exists (covers the brief window between task creation and
        begin_turn, which only this layer can see)."""
        if session.has_live_work:
            return True
        return any(
            a == agent_name and sid == session.id and chat.task is not None and not chat.task.done()
            for (a, _user, sid), chat in self._active_chats.items()
        )

    async def _list_sessions(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        source = request.query_params.get("source")
        status = request.query_params.get("status")
        parent_id = request.query_params.get("parent_id")
        include_superseded = request.query_params.get("include_superseded", "").lower() in ("1", "true", "yes")
        limit = self._parse_limit(request, default=100, cap=1000)

        q = request.query_params.get("q")
        if q:
            # Search scans the FULL session set (the recency limit exists to
            # bound the sidebar payload, not to hide sessions from search).
            all_sessions = adapter.session_store.search_sessions(adapter.agent_name, q, limit=limit)
            if source:
                all_sessions = [s for s in all_sessions if s.source == source]
            if status:
                all_sessions = [s for s in all_sessions if s.status == status]
        else:
            all_sessions = adapter.session_store.list_sessions(
                agent=adapter.agent_name,
                source=source,
                status=status,
                parent_id=parent_id,
                limit=limit,
                include_superseded=include_superseded,
            )

        from tsugite_daemon.session_store import SessionStatus

        default_ids = adapter.session_store.default_primary_ids(adapter.agent_name)
        live_statuses = {SessionStatus.RUNNING.value, SessionStatus.ACTIVE.value}

        def _user_label(user_id: str, source: str) -> str:
            if user_id.isdigit():
                return f"Discord: {user_id}"
            if user_id.startswith("web-"):
                return f"Web: {user_id}"
            return user_id or source

        sessions = []
        for s in all_sessions:
            user_id = s.user_id or ""
            label = _user_label(user_id, s.source)
            unread = bool(s.last_active and (not s.last_viewed_at or s.last_active > s.last_viewed_at))
            row = {
                "id": s.id,
                "user_id": user_id,
                "label": label,
                "conversation_id": s.id,
                "source": s.source,
                "status": s.status,
                "state": s.status,
                "created_at": s.created_at,
                "last_active": s.last_active,
                "parent_id": s.parent_id,
                "prompt": s.prompt or "",
                "model": s.model,
                "error": s.error,
                "result": s.result,
                "title": s.title,
                "is_default": default_ids.get(user_id) == s.id,
                "metadata": s.metadata or {},
                "pinned": s.pinned,
                "pin_position": s.pin_position,
                "last_viewed_at": s.last_viewed_at,
                "superseded_by": s.superseded_by,
                "unread": unread,
                "is_primary": s.is_primary,
                # Authoritative busy flag. The UI must render busy state from
                # this, never infer it from cached progress labels.
                "busy": self._session_busy(adapter.agent_name, s),
            }
            if s.status in live_statuses:
                progress = adapter.session_store.session_progress_summary(s.id)
                if not row["busy"]:
                    # The fold reports the last status-bearing event, which is the
                    # current status only while the log ends on a terminator. An
                    # idle session whose log ends mid-turn (compaction drops the
                    # retained turns' session_end markers; a crash truncates the
                    # same way) would otherwise show "Waiting on LLM..." forever.
                    progress = {**progress, "status_text": ""}
                row["progress"] = progress
            sessions.append(row)

        return JSONResponse({"sessions": sessions})

    async def _new_interactive_session(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))
        title = body.get("title")
        if title is not None and not isinstance(title, str):
            return JSONResponse({"error": "title must be a string"}, status_code=400)

        from tsugite_daemon.session_store import SessionSource, create_interactive_session

        session_id = create_interactive_session(
            adapter.session_store,
            adapter.agent_name,
            user_id,
            title=title,
            event_bus=self.event_bus,
            source=SessionSource.WEB.value,
        )
        return JSONResponse({"id": session_id}, status_code=201)

    async def _status(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        session_id = request.query_params.get("session_id")
        session = _session_or_default(adapter, session_id, user_id)

        model = adapter.resolve_model()
        resolved_model = _resolve_full_model_id(model)
        attachments = [
            {"name": a.name, "content_type": a.content_type.value, "mime_type": a.mime_type}
            for a in adapter._get_all_attachments()
        ]
        if session is not None:
            tokens = session.cumulative_tokens
            message_count = session.message_count
            session_metadata = session.metadata or {}
            chat = self._active_chats.get((adapter.agent_name, user_id, session.id))
            backend = chat.backend if chat else None
        else:
            tokens, message_count, session_metadata, backend = 0, 0, {}, None

        return JSONResponse(
            {
                "model": model,
                "resolved_model": resolved_model if resolved_model != model else None,
                "tokens": tokens,
                "context_limit": (
                    adapter.session_store.get_session_context_limit(session.id)
                    if session
                    else adapter.session_store.get_context_limit(adapter.agent_name)
                ),
                "threshold": (
                    adapter.session_store.get_session_compaction_threshold(session.id)
                    if session
                    else adapter.session_store.get_compaction_threshold(adapter.agent_name)
                ),
                "message_count": message_count,
                "compacting": adapter.session_store.is_compacting(
                    user_id, adapter.agent_name, session_id=session.id if session else None
                ),
                "metadata": session_metadata,
                "busy": bool(session and self._session_busy(adapter.agent_name, session)),
                "pending_message": backend.pending_message if backend else None,
                "attachments": attachments,
            }
        )

    async def _update_agent_config(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        body = await request.json()
        agent_name = request.path_params["agent"]
        agent_config = adapter.agent_config

        if "model" in body:
            new_model = body["model"].strip() if body["model"] else None
            # Pin existing sessions to the current model before mutating the
            # agent default so they don't silently switch on their next turn.
            # The default change should only affect sessions created after it.
            adapter.session_store.freeze_session_models_to_current(adapter.agent_name, agent_config.model)
            agent_config.model = new_model

            from tsugite_daemon.memory import DEFAULT_CONTEXT_LIMIT, get_context_limit

            if new_model:
                context_limit = get_context_limit(new_model, fallback=DEFAULT_CONTEXT_LIMIT)
                agent_config.context_limit = context_limit
            else:
                context_limit = DEFAULT_CONTEXT_LIMIT
                agent_config.context_limit = None
            adapter.session_store.update_context_limit(adapter.agent_name, context_limit)

            if self.gateway:
                from tsugite_daemon.config import save_daemon_config

                save_daemon_config(self.gateway.config, self.gateway.config_path)

        self.event_bus.emit("agent_status", {"agent": agent_name})
        return JSONResponse(
            {"status": "ok", "model": adapter.resolve_model(), "context_limit": agent_config.context_limit}
        )

    async def _attachments(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err
        attachments = []
        for a in adapter._get_all_attachments():
            entry = {"name": a.name, "content_type": a.content_type.value, "mime_type": a.mime_type}
            if a.content_type.value == "text":
                entry["content"] = a.content
            else:
                entry["size_bytes"] = len(a.content) if a.content else 0
            attachments.append(entry)
        return JSONResponse({"attachments": attachments})

    @staticmethod
    def _collect_events(session_id: str, limit: int = 0) -> list[dict]:
        """Read one session's JSONL and return its events as raw dicts in file
        order. ``limit`` trims to the last N ``user_input`` bubbles plus
        whatever follows them.

        Predecessor files are not walked: the new file's leading ``compaction``
        event already carries the canonical pre-compaction summary, so reading
        ancestor files would duplicate context the agent has already received.
        Offline chain traversal is supported via the ``compacted_into`` /
        ``source_session_id`` pointers written into each file at compaction
        time.
        """
        from tsugite.history import get_history_backend

        backend = get_history_backend()
        if not backend.exists(session_id):
            return []
        events: list[dict] = []
        user_input_offsets: list[int] = []
        for event in backend.load(session_id).iter_events():
            if event.type == "user_input":
                user_input_offsets.append(len(events))
            # Same {type, ts, data} shape the raw JSONL lines had (id never exposed).
            events.append(event.model_dump(mode="json", exclude={"id"}, exclude_none=True))

        if limit > 0 and len(user_input_offsets) > limit:
            events = events[user_input_offsets[-limit] :]

        return events

    @staticmethod
    def _resolve_session_id(adapter, user_id: str, request: Request) -> Optional[str]:
        """Use ?session_id= when given, otherwise the user's primary session, otherwise None."""
        session_id = request.query_params.get("session_id")
        if session_id:
            return session_id
        primary = adapter.session_store.find_default_session(user_id, adapter.agent_name)
        return primary.id if primary else None

    async def _history(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        limit = self._parse_limit(request, default=100, cap=1000)
        conversation_id = self._resolve_session_id(adapter, user_id, request)
        if conversation_id is None:
            return JSONResponse({"conversation_id": None, "events": []})

        events = self._collect_events(conversation_id, limit=limit)

        # UI events (prompt_snapshots) are now part of the same
        # session JSONL as conversation events, so they're already included.

        return JSONResponse(
            {
                "conversation_id": conversation_id,
                "events": events,
            }
        )

    async def _prompt_snapshot(self, request: Request) -> JSONResponse:
        """Return the latest prompt snapshot for the current session.

        Includes full messages from the live progress handler if available,
        otherwise just the token breakdown from the persisted event log.
        """
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        agent_name = request.path_params["agent"]

        session_id = self._resolve_session_id(adapter, user_id, request)
        if session_id is None:
            return JSONResponse({"prompt_snapshot": None})
        events = adapter.session_store.read_events(session_id)
        snapshots = [e for e in events if e.get("type") == "prompt_snapshot"]
        breakdown = snapshots[-1].get("token_breakdown", {}) if snapshots else {}

        backend_key = (agent_name, user_id, session_id)
        chat = self._active_chats.get(backend_key)
        live_progress = chat.progress if chat else None
        if live_progress and live_progress.latest_prompt_messages:
            return JSONResponse(
                {
                    "prompt_snapshot": {
                        "messages": live_progress.latest_prompt_messages,
                        "token_breakdown": breakdown,
                    }
                }
            )

        if not snapshots:
            return JSONResponse({"prompt_snapshot": None})
        return JSONResponse({"prompt_snapshot": {"token_breakdown": breakdown}})

    async def _raw_messages(self, request: Request) -> JSONResponse:
        """Reconstruct, per turn, the raw request messages the model saw and its
        raw response, rebuilt on demand from the persisted event log.

        Nothing is stored: the reconstruction runs off the same event source
        resume reads, so it is replay-safe and always matches the live prompt.
        `system_prompt` is best-effort and null for now (the stable prompt isn't
        in the log; the UI notes it isn't shown).
        """
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        session_id = self._resolve_session_id(adapter, user_id, request)
        if session_id is None:
            return JSONResponse({"raw_messages": None})

        from tsugite.history import reconstruct_raw_turns

        events = _load_session_events(session_id)
        return JSONResponse(
            {
                "raw_messages": {
                    "system_prompt": None,
                    "turns": reconstruct_raw_turns(events),
                }
            }
        )

    async def _branch(self, request: Request) -> JSONResponse:
        """Fork a session at an event into an independent branch (#400)."""
        adapter, err = self._get_adapter(request)
        if err:
            return err

        session_id = request.path_params["session_id"]
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        at_event_id = body.get("at_event_id")
        if at_event_id is None:
            return JSONResponse({"error": "at_event_id is required"}, status_code=400)

        try:
            branch = adapter.session_store.branch_session(session_id, int(at_event_id), label=body.get("label"))
        except (ValueError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        self.event_bus.emit("agent_status", {"agent": request.path_params["agent"]})
        return JSONResponse({"session_id": branch.id})

    async def _compact(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))

        session_id = body.get("session_id")
        session = _session_or_default(adapter, session_id, user_id)

        if session is None or session.message_count == 0:
            return JSONResponse({"error": "no session to compact"}, status_code=404)

        agent_name = request.path_params["agent"]
        old_conv_id = session.id

        if not adapter.session_store.begin_compaction(user_id, adapter.agent_name, session_id=old_conv_id):
            return JSONResponse({"error": "compaction already in progress"}, status_code=409)

        adapter._broadcast_compaction("compaction_started", agent_name, old_conv_id)
        new_session = None
        try:
            instructions = body.get("instructions")
            new_session = await adapter._compact_session(
                session.id,
                instructions=instructions,
                reason="manual",
                progress_callback=adapter._compaction_progress_cb(old_conv_id),
            )
        except Exception as e:
            msg = str(e) or repr(e)
            logger.exception("Compaction failed for agent %s", adapter.agent_name)
            return JSONResponse({"error": f"compaction failed: {msg}"}, status_code=500)
        finally:
            adapter.session_store.end_compaction(user_id, adapter.agent_name, session_id=old_conv_id)
            adapter._broadcast_compaction("compaction_finished", agent_name, old_conv_id)

        self.event_bus.emit("agent_status", {"agent": agent_name})
        if new_session:
            self.event_bus.emit(
                "session_update",
                {"action": "compacted", "id": old_conv_id, "successor_id": new_session.id},
            )
        return JSONResponse(
            {
                "status": "compacted",
                "old_conversation_id": old_conv_id,
                "new_conversation_id": new_session.id if new_session else old_conv_id,
            }
        )

    async def _unload_skill(self, request: Request) -> JSONResponse:
        """Suppress a skill for the rest of this session's lifetime.

        AgentPreparer will skip the skill on subsequent turns so it does not
        reload from auto_load_skills or a trigger match. In-memory only; a
        daemon restart resets suppression by design.
        """
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        skill_name = body.get("name")
        if not isinstance(skill_name, str) or not skill_name:
            return JSONResponse({"error": "name is required"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))
        session = adapter.session_store.find_default_session(user_id, adapter.agent_name)
        if session is None:
            return JSONResponse({"error": "no default session"}, status_code=404)
        adapter.session_store.suppress_skill(session.id, skill_name)

        # Drop it from the currently-loaded manager too so any in-flight code path
        # that still reads the global manager's state sees the removal. Best-effort:
        # if the manager hasn't been initialised yet (no prior `prepare()` call)
        # there's nothing to clear, and the suppression above still takes effect
        # on the next turn.
        from tsugite.tools.skills import get_skill_manager

        try:
            get_skill_manager().unload_skill(skill_name)
        except (AttributeError, RuntimeError):
            logger.debug("unload_skill on global manager skipped for %s", skill_name, exc_info=True)

        return JSONResponse({"status": "ok", "session_id": session.id, "name": skill_name})

    async def _effort_levels(self, request: Request) -> JSONResponse:
        """Return the effort levels supported by the session's resolved model."""
        adapter, err = self._get_adapter(request)
        if err:
            return err

        session_id = request.query_params.get("session_id")
        return JSONResponse(
            {
                "model": adapter.resolve_session_model(session_id),
                "supported_effort_levels": adapter.session_effort_levels(session_id),
            }
        )

    async def _session_scoped_request(
        self, request: Request
    ) -> tuple[Optional["HTTPAgentAdapter"], Optional[dict], Optional[JSONResponse]]:
        """Parse a JSON body carrying an optional ``session_id`` and resolve the
        adapter that owns that session. Returns ``(adapter, body, err)``; on failure
        ``err`` is a JSONResponse and adapter/body are None.

        Routing by the body's session_id (not just the URL agent) lets the web UI
        open a session whose owning agent differs from the URL - e.g. a job's
        session carrying the worker agent-file. Auth is still enforced in
        ``_get_adapter``.
        """
        try:
            body = await request.json()
        except Exception:
            body = None
        session_id = body.get("session_id") if isinstance(body, dict) else None
        adapter, err = self._get_adapter(request, fallback_session_id=session_id)
        if err:
            return None, None, err
        if not isinstance(body, dict):
            return None, None, JSONResponse({"error": "invalid JSON body"}, status_code=400)
        return adapter, body, None

    async def _respond(self, request: Request) -> JSONResponse:
        """Submit a response to an active ask_user prompt.

        Resolution is by durable ``ask_id`` (required): it survives a reload and a
        rotated session triple, so a reloaded prompt can still be answered. An ask
        whose in-memory backend is gone (timeout or daemon restart) is settled
        durably instead of silently lost.
        """
        adapter, body, err = await self._session_scoped_request(request)
        if err:
            return err
        session_id = body.get("session_id")

        response = body.get("response", "")
        if not isinstance(response, str):
            return JSONResponse({"error": "response must be a string"}, status_code=400)
        if len(response) > 10_000:
            return JSONResponse({"error": "response too long (max 10000 chars)"}, status_code=400)

        ask_id = body.get("ask_id")
        if not isinstance(ask_id, str) or not ask_id:
            return JSONResponse({"error": "ask_id is required"}, status_code=400)
        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))
        logger.info("[%s] respond user_id=%s session_id=%s ask_id=%s", adapter.agent_name, user_id, session_id, ask_id)

        # The still-blocking ask, resolved by its durable id.
        backend = resolve_pending_ask(ask_id)
        if backend is not None:
            backend.submit_response(response)
            return JSONResponse({"status": "ok"})

        # Durably pending but its in-memory backend is gone (timed out, or the
        # daemon restarted): record ask_answered so a reload stops re-prompting,
        # and tell the client the ask is no longer live instead of a bare 404.
        self._settle_stale_ask(adapter, session_id, ask_id, response)
        return JSONResponse({"status": "expired", "detail": "This prompt is no longer active."})

    def _settle_stale_ask(self, adapter: HTTPAgentAdapter, session_id: str, ask_id: str, response: str) -> None:
        """Durably answer an ask whose backend is no longer in memory.

        Writes only when the ask is durably pending (an ``ask_user`` for this id
        with no later ``ask_answered``), so a stale or replayed request can't spam
        the log. Persists ``ask_answered`` and broadcasts it so an open tab clears
        the prompt.
        """
        try:
            events = adapter.session_store.read_events(session_id)
        except Exception:
            return
        pending = False
        for e in events:
            if e.get("ask_id") != ask_id:
                continue
            if e.get("type") == "ask_user":
                pending = True
            elif e.get("type") == "ask_answered":
                pending = False
        if not pending:
            return
        payload = {"type": "ask_answered", "ask_id": ask_id, "answer": response}
        try:
            build_session_event_persister(adapter.session_store, session_id)(payload)
        except Exception:
            logger.debug("Failed to persist stale ask_answered for %s", ask_id, exc_info=True)
        if self.event_bus:
            try:
                self.event_bus.emit(
                    "session_event",
                    {"session_id": session_id, "event_type": "ask_answered", "ask_id": ask_id, "answer": response},
                )
            except Exception:
                logger.debug("Failed to broadcast stale ask_answered for %s", ask_id, exc_info=True)

    async def _upload(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        uploads_dir = adapter.agent_config.workspace_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Reject from the header BEFORE request.form() spools the whole
        # multipart body to disk/memory - the per-file check below only fires
        # after the bytes are already materialized.
        try:
            declared = int(request.headers.get("content-length", "0"))
        except ValueError:
            declared = 0
        if declared > MAX_UPLOAD_TOTAL:
            return JSONResponse({"error": "total upload size exceeds 100MB"}, status_code=413)

        form = await request.form()
        files = form.getlist("files")
        if not files:
            return JSONResponse({"error": "no files provided"}, status_code=400)
        if len(files) > MAX_UPLOAD_FILES:
            return JSONResponse({"error": f"too many files (max {MAX_UPLOAD_FILES})"}, status_code=400)

        total_size = 0
        results = []
        written_paths = []
        for upload in files:
            content = await upload.read()
            total_size += len(content)
            if total_size > MAX_UPLOAD_TOTAL:
                for p in written_paths:
                    p.unlink(missing_ok=True)
                return JSONResponse({"error": "total upload size exceeds 100MB"}, status_code=413)

            name = _sanitize_filename(upload.filename or "upload")
            dest = uploads_dir / name
            if not dest.resolve().is_relative_to(uploads_dir.resolve()):
                continue
            dest, dedup_err = _deduplicate_dest(uploads_dir, name)
            if dedup_err:
                continue

            dest.write_bytes(content)
            written_paths.append(dest)
            file_size = len(content)
            mime_type, content_type = _file_handler.detect_content_type(dest)
            context_attach = can_inline_file(dest, file_size)

            results.append(
                {
                    "name": dest.name,
                    "content_type": content_type.value,
                    "mime_type": mime_type,
                    "size": file_size,
                    "context_attach": context_attach,
                }
            )

        await form.close()
        return JSONResponse({"files": results})

    async def _chat(self, request: Request) -> Response:
        adapter, body, err = await self._session_scoped_request(request)
        if err:
            return err
        session_id = body.get("session_id")

        message = body.get("message", "").strip()
        uploaded_files = body.get("uploaded_files", [])
        if not isinstance(uploaded_files, list):
            uploaded_files = []

        if not message and not uploaded_files:
            return JSONResponse({"error": "message or uploaded_files is required"}, status_code=400)

        raw_user_id = body.get("user_id", "web-anonymous")
        agent_name = adapter.agent_name
        user_id = adapter.resolve_http_user(raw_user_id)
        logger.info("[%s] <- %s (http): %s", agent_name, user_id, message[:100])

        reasoning_effort, err_resp = self._resolve_effort_or_400(adapter, body.get("reasoning_effort"))
        if err_resp:
            return err_resp

        # Process uploaded files -- only accept filenames, resolve against uploads dir
        uploaded_attachments = []
        workspace_only_files = []
        uploads_dir = adapter.agent_config.workspace_dir / "uploads"
        # A non-vision model can't read an inlined image; route its images to the
        # workspace-only path (saved + path hint) instead of dropping them.
        from tsugite.models import model_supports_vision

        supports_vision = model_supports_vision(adapter.resolve_session_model(session_id))

        for file_info in uploaded_files:
            if not isinstance(file_info, dict):
                continue
            filename = _sanitize_filename(file_info.get("name", ""))
            file_path = (uploads_dir / filename).resolve()
            if not file_path.is_relative_to(uploads_dir.resolve()) or not file_path.exists():
                continue

            if can_inline_file(file_path, file_path.stat().st_size, supports_vision=supports_vision):
                try:
                    attachment = _file_handler.fetch(str(file_path))
                    attachment.user_upload = True
                    uploaded_attachments.append(attachment)
                except Exception as e:
                    logger.warning("Failed to create attachment for %s: %s", file_path, e)
                    workspace_only_files.append(filename)
            else:
                workspace_only_files.append(filename)

        message += _format_upload_message_suffix(workspace_only_files, [a.name for a in uploaded_attachments])

        metadata = {"client_ip": request.client.host if request.client else "unknown"}
        if uploaded_attachments:
            metadata["uploaded_attachments"] = uploaded_attachments
        if reasoning_effort:
            metadata["reasoning_effort_override"] = reasoning_effort
        context_metadata = body.get("context_metadata")
        if isinstance(context_metadata, list) and context_metadata:
            metadata["context_metadata"] = context_metadata

        from tsugite_daemon.session_store import FINISHED_STATUSES, SessionSource

        target_session = None
        if session_id:
            try:
                target_session = adapter.session_store.get_session(session_id)
            except ValueError:
                target_session = None
            if target_session is not None and target_session.status in FINISHED_STATUSES:
                successor = adapter.session_store.resolve_compacted_successor(session_id)
                if successor is not None and successor.status not in FINISHED_STATUSES:
                    target_session = successor
                else:
                    return JSONResponse(
                        {
                            "error": f"Session is {target_session.status}. Start a new session to continue.",
                            "code": "session_finished",
                        },
                        status_code=409,
                    )
            if target_session is not None:
                metadata["conv_id_override"] = target_session.id

        if target_session is None:
            target_session = adapter.session_store.get_or_create_interactive(
                user_id, adapter.agent_name, source=SessionSource.WEB.value
            )
        target_session_id = target_session.id

        backend_key = (agent_name, user_id, target_session_id)
        # Same predicate the sessions payload and /status report - the server
        # must not 409 a send while telling the sidebar the session is idle.
        if self._session_busy(agent_name, target_session):
            return JSONResponse(
                {"error": "a turn is already running for this session", "code": "turn_in_flight"},
                status_code=409,
            )

        channel_context = ChannelContext(
            source="http",
            channel_id=None,
            user_id=raw_user_id,
            reply_to=f"http:{raw_user_id}",
            metadata=metadata,
        )

        progress = SSEProgressHandler()
        progress.set_loop(asyncio.get_running_loop())
        progress.set_session_id(target_session_id)
        progress.set_broadcaster(self.event_bus)
        progress.set_event_persister(build_session_event_persister(adapter.session_store, target_session_id))
        custom_logger = SimpleNamespace(ui_handler=progress)

        interaction_backend = HTTPInteractionBackend(progress)
        interaction_backend.pending_message = message
        chat_state = ActiveChat(backend=interaction_backend, progress=progress)
        self._active_chats[backend_key] = chat_state

        async def run_agent():
            from tsugite.agent_runner.helpers import set_current_daemon_agent
            from tsugite.cancellation import set_cancel_event
            from tsugite.interaction import set_interaction_backend
            from tsugite.permissions import Permissions, set_permissions

            set_interaction_backend(interaction_backend)
            # Expose the adapter's REGISTERED name (its key in the daemon adapter
            # registry) so spawn/start-session tools resolve to an agent that has
            # a live adapter, not the agent-file config name. Rides the same
            # context copy asyncio.to_thread makes for the executor worker.
            set_current_daemon_agent(adapter.agent_name)
            # Bind the approval permissions store into the run context alongside the
            # interaction backend, so the context detector (which runs via
            # asyncio.to_thread and inherits this context) can gate a web fetch on
            # the allowlist and prompt through the same cross-surface machinery.
            set_permissions(
                Permissions(
                    runtime_path=self._permissions_runtime_path(),
                    workspace_dir=adapter.agent_config.workspace_dir,
                )
            )
            # Bind the cooperative cancel Event into the run context so the agent
            # loop (copy_context + to_thread) observes a user Stop and exits cleanly.
            set_cancel_event(chat_state.cancel_event)
            try:
                response = await adapter.handle_message(
                    user_id=user_id,
                    message=message,
                    channel_context=channel_context,
                    custom_logger=custom_logger,
                )
                # Only emit final_result if the EventBus didn't already
                # (FinalAnswerEvent fires during handle_message for normal completions,
                # but not for max_turns/error cases)
                logger.info("[%s] -> %s (http): %s", adapter.agent_name, user_id, (response or "")[:100])
                if not progress.has_final:
                    progress._emit("final_result", {"result": response})

                self.event_bus.emit("agent_status", {"agent": agent_name})
                self.event_bus.emit("history_update", {"agent": agent_name, "session_id": target_session_id})

                try:
                    refreshed = adapter.session_store.get_session(target_session_id)
                except ValueError:
                    refreshed = None
                if refreshed is not None:
                    progress._emit(
                        "session_info",
                        {
                            "session_id": target_session_id,
                            "tokens": refreshed.cumulative_tokens,
                            "context_limit": adapter.session_store.get_session_context_limit(target_session_id),
                            "threshold": adapter.session_store.get_session_compaction_threshold(target_session_id),
                            "message_count": refreshed.message_count,
                            "model": adapter.resolve_model(),
                            "attachments": [a.name for a in adapter._get_all_attachments()],
                        },
                    )
            except asyncio.CancelledError:
                logger.info("[%s] Chat cancelled by user for %s", adapter.agent_name, user_id)
                progress._emit("cancelled", {"reason": "User cancelled"})
            except Exception as e:
                logger.exception("[%s] Chat error", adapter.agent_name)
                progress._emit("error", {"error": str(e)})
            finally:
                self._active_chats.pop(backend_key, None)
                progress.signal_done()

        task = asyncio.create_task(run_agent())
        chat_state.task = task

        return StreamingResponse(
            progress.event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Session-Id": target_session_id,
            },
        )

    async def _cancel_chat(self, request: Request) -> JSONResponse:
        adapter, body, err = await self._session_scoped_request(request)
        if err:
            return err
        session_id = body.get("session_id")
        raw_user_id = body.get("user_id", "web-anonymous")
        user_id = adapter.resolve_http_user(raw_user_id)
        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)
        backend_key = (adapter.agent_name, user_id, session_id)
        chat = self._active_chats.get(backend_key)
        if chat and chat.task and not chat.task.done():
            # Signal the worker thread to stop at its next safe checkpoint (the real
            # stop - task.cancel alone only tears down the awaiting coroutine/SSE
            # stream, leaving the to_thread worker running to completion).
            chat.cancel_event.set()
            chat.task.cancel()
            return JSONResponse({"status": "cancelled"})
        return JSONResponse({"error": "no active chat"}, status_code=404)
