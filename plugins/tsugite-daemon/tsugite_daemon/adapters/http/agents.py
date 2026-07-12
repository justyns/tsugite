"""AgentsMixin: agents HTTP handlers for HTTPServer (split from adapters/http.py)."""

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tsugite_daemon.adapters.base import ChannelContext
from tsugite_daemon.adapters.http.helpers import (
    MAX_UPLOAD_FILES,
    MAX_UPLOAD_TOTAL,
    ActiveChat,
    HTTPAgentAdapter,
    _deduplicate_dest,
    _file_handler,
    _format_upload_message_suffix,
    _is_text_mime,
    _resolve_full_model_id,
    _sanitize_filename,
    _should_context_attach,
    build_session_event_persister,
    logger,
)
from tsugite_daemon.adapters.http.sse import (
    HTTPInteractionBackend,
    SSEProgressHandler,
)

if TYPE_CHECKING:
    pass


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
            Route("/api/agents/{agent}/config", self._update_agent_config, methods=["PATCH"]),
            Route("/api/agents/{agent}/compact", self._compact, methods=["POST"]),
            Route("/api/agents/{agent}/respond", self._respond, methods=["POST"]),
            Route("/api/agents/{agent}/unload-skill", self._unload_skill, methods=["POST"]),
            Route("/api/agents/{agent}/effort-levels", self._effort_levels, methods=["GET"]),
            Route("/api/agents/{agent}/workspace", self._list_workspace_files, methods=["GET"]),
            Route("/api/agents/{agent}/workspace/content", self._read_workspace_file, methods=["GET"]),
            Route("/api/agents/{agent}/workspace/content", self._save_workspace_file, methods=["PUT"]),
            Route("/api/agents/{agent}/workspace/attach", self._attach_workspace_file, methods=["POST"]),
            Route("/api/agents/{agent}/commands/{command_name}", self._run_command, methods=["POST"]),
        ]

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
        reporting the session idle. True when the durable turn_in_flight marker
        is set OR a live HTTP chat task exists (covers the brief window between
        task creation and begin_turn)."""
        if session.turn_in_flight:
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
        try:
            limit = max(1, min(int(request.query_params.get("limit", "100")), 1000))
        except (ValueError, TypeError):
            limit = 100

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
                row["progress"] = adapter.session_store.session_progress_summary(s.id)
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

        from tsugite_daemon.session_store import create_interactive_session

        session_id = create_interactive_session(
            adapter.session_store, adapter.agent_name, user_id, title=title, event_bus=self.event_bus
        )
        return JSONResponse({"id": session_id}, status_code=201)

    async def _status(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        user_id = adapter.resolve_http_user(request.query_params.get("user_id", "web-anonymous"))
        session_id = request.query_params.get("session_id")
        session = None
        if session_id:
            try:
                session = adapter.session_store.get_session(session_id)
            except ValueError:
                session = None
        if session is None:
            session = adapter.session_store.find_default_session(user_id, adapter.agent_name)

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
        try:
            limit = max(1, min(int(request.query_params.get("limit", "100")), 1000))
        except (ValueError, TypeError):
            limit = 100
        conversation_id = self._resolve_session_id(adapter, user_id, request)
        if conversation_id is None:
            return JSONResponse({"conversation_id": None, "events": []})

        events = self._collect_events(conversation_id, limit=limit)

        # UI events (reactions, prompt_snapshots) are now part of the same
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
        session = None
        if session_id:
            try:
                session = adapter.session_store.get_session(session_id)
            except ValueError:
                session = None
        if session is None:
            session = adapter.session_store.find_default_session(user_id, adapter.agent_name)

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
            new_session = await adapter._compact_session(session.id, instructions=instructions, reason="manual")
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

    @staticmethod
    def _resolve_session_model(adapter: "HTTPAgentAdapter", session_id: Optional[str]) -> str:
        """Resolve the effective model for a session, honoring a per-session override.

        Falls back to the agent/daemon default (``adapter.resolve_model()``) when
        no session is given or the session has no model override.
        """
        if session_id:
            override = adapter.session_store.get_model_override(session_id)
            if override:
                return override
        return adapter.resolve_model()

    async def _effort_levels(self, request: Request) -> JSONResponse:
        """Return the effort levels supported by the session's resolved model."""
        adapter, err = self._get_adapter(request)
        if err:
            return err

        from tsugite.models import get_model_id, parse_model_string, resolve_model_alias
        from tsugite.providers import get_provider

        model_string = self._resolve_session_model(adapter, request.query_params.get("session_id"))
        levels: list[str] | None = None
        try:
            resolved = resolve_model_alias(model_string)
            provider_name, _, _ = parse_model_string(resolved)
            provider = get_provider(provider_name)
            info = provider.get_model_info(get_model_id(resolved))
            if info and info.supported_effort_levels:
                levels = list(info.supported_effort_levels)
        except (ValueError, Exception):  # noqa: BLE001 — treat any resolution failure as "unknown"
            pass

        return JSONResponse({"model": model_string, "supported_effort_levels": levels})

    async def _respond(self, request: Request) -> JSONResponse:
        """Submit a response to an active ask_user prompt."""
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        response = body.get("response", "")
        if not isinstance(response, str):
            return JSONResponse({"error": "response must be a string"}, status_code=400)
        if len(response) > 10_000:
            return JSONResponse({"error": "response too long (max 10000 chars)"}, status_code=400)

        user_id = adapter.resolve_http_user(body.get("user_id", "web-anonymous"))
        agent_name = request.path_params["agent"]
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)
        logger.info("[%s] respond from user_id=%s session_id=%s", agent_name, user_id, session_id)

        key = (agent_name, user_id, session_id)
        chat = self._active_chats.get(key)
        if not chat:
            return JSONResponse({"error": "no pending question for this session"}, status_code=404)

        chat.backend.submit_response(response)
        return JSONResponse({"status": "ok"})

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
            mime_type, content_type = _file_handler._detect_content_type(dest)
            context_attach = _should_context_attach(dest, file_size)

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
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        message = body.get("message", "").strip()
        uploaded_files = body.get("uploaded_files", [])
        if not isinstance(uploaded_files, list):
            uploaded_files = []

        if not message and not uploaded_files:
            return JSONResponse({"error": "message or uploaded_files is required"}, status_code=400)

        raw_user_id = body.get("user_id", "web-anonymous")
        agent_name = request.path_params["agent"]
        user_id = adapter.resolve_http_user(raw_user_id)
        logger.info("[%s] <- %s (http): %s", agent_name, user_id, message[:100])

        reasoning_effort, err_resp = self._resolve_effort_or_400(adapter, body.get("reasoning_effort"))
        if err_resp:
            return err_resp

        # Process uploaded files — only accept filenames, resolve against uploads dir
        uploaded_attachments = []
        workspace_only_files = []
        uploads_dir = adapter.agent_config.workspace_dir / "uploads"

        for file_info in uploaded_files:
            if not isinstance(file_info, dict):
                continue
            filename = _sanitize_filename(file_info.get("name", ""))
            file_path = (uploads_dir / filename).resolve()
            if not file_path.is_relative_to(uploads_dir.resolve()) or not file_path.exists():
                continue

            if _should_context_attach(file_path, file_path.stat().st_size):
                try:
                    attachment = _file_handler.fetch(str(file_path))
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

        from tsugite_daemon.session_store import FINISHED_STATUSES

        session_id = body.get("session_id")
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
            target_session = adapter.session_store.get_or_create_interactive(user_id, adapter.agent_name)
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
            from tsugite.cancellation import set_cancel_event
            from tsugite.interaction import set_interaction_backend

            set_interaction_backend(interaction_backend)
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
        adapter, err = self._get_adapter(request)
        if err:
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        raw_user_id = body.get("user_id", "web-anonymous")
        user_id = adapter.resolve_http_user(raw_user_id)
        session_id = body.get("session_id")
        if not session_id:
            return JSONResponse({"error": "session_id is required"}, status_code=400)
        backend_key = (request.path_params["agent"], user_id, session_id)
        chat = self._active_chats.get(backend_key)
        if chat and chat.task and not chat.task.done():
            # Signal the worker thread to stop at its next safe checkpoint (the real
            # stop - task.cancel alone only tears down the awaiting coroutine/SSE
            # stream, leaving the to_thread worker running to completion).
            chat.cancel_event.set()
            chat.task.cancel()
            return JSONResponse({"status": "cancelled"})
        return JSONResponse({"error": "no active chat"}, status_code=404)

    def _validate_workspace_path(
        self, adapter: "HTTPAgentAdapter", path_str: str
    ) -> tuple[Path, Optional[JSONResponse]]:
        """Validate a workspace file path stays within the workspace directory."""
        workspace_dir = adapter.agent_config.workspace_dir
        try:
            resolved = (workspace_dir / path_str).resolve()
        except (ValueError, OSError):
            return Path(), JSONResponse({"error": "invalid path"}, status_code=400)
        if not resolved.is_relative_to(workspace_dir.resolve()):
            return Path(), JSONResponse({"error": "path outside workspace"}, status_code=403)
        return resolved, None

    async def _list_workspace_files(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        workspace_dir = adapter.agent_config.workspace_dir
        if not workspace_dir.is_dir():
            return JSONResponse({"entries": [], "subdir": "", "workspace_dir": str(workspace_dir)})

        subdir = request.query_params.get("subdir", "")
        if subdir:
            target, path_err = self._validate_workspace_path(adapter, subdir)
            if path_err:
                return path_err
            if not target.is_dir():
                return JSONResponse({"error": "not a directory"}, status_code=400)
        else:
            target = workspace_dir

        from tsugite.tools.fs import _build_gitignore_matcher

        gitignore_spec = _build_gitignore_matcher(workspace_dir)
        entries = []
        try:
            import stat as stat_mod

            for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                try:
                    st = item.lstat()
                except OSError:
                    continue
                if stat_mod.S_ISLNK(st.st_mode):
                    continue
                is_dir = stat_mod.S_ISDIR(st.st_mode)
                rel = str(item.relative_to(workspace_dir))
                if gitignore_spec and gitignore_spec.match_file(rel + ("/" if is_dir else "")):
                    continue
                if is_dir:
                    entries.append({"path": rel, "name": item.name, "is_dir": True})
                elif stat_mod.S_ISREG(st.st_mode) and _is_text_mime(item):
                    entries.append(
                        {
                            "path": rel,
                            "name": item.name,
                            "is_dir": False,
                            "size": st.st_size,
                            "modified": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                        }
                    )
        except OSError as e:
            return JSONResponse({"error": f"listing failed: {e}"}, status_code=500)

        return JSONResponse({"entries": entries, "subdir": subdir, "workspace_dir": str(workspace_dir)})

    async def _read_workspace_file(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        path_str = request.query_params.get("path", "")
        if not path_str:
            return JSONResponse({"error": "path parameter required"}, status_code=400)

        resolved, path_err = self._validate_workspace_path(adapter, path_str)
        if path_err:
            return path_err
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        if resolved.is_dir():
            return JSONResponse({"error": "path is a directory"}, status_code=400)

        st = resolved.stat()

        if not _is_text_mime(resolved):
            return JSONResponse({"path": path_str, "content": None, "is_text": False, "size": st.st_size})

        max_size = self.config.max_workspace_file_size
        if st.st_size > max_size:
            return JSONResponse(
                {"error": f"file too large (max {max_size // 1024}KB for text viewing)"}, status_code=413
            )

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return JSONResponse({"path": path_str, "content": None, "is_text": False, "size": st.st_size})
        except OSError as e:
            return JSONResponse({"error": f"read failed: {e}"}, status_code=500)

        return JSONResponse({"path": path_str, "content": content, "is_text": True})

    async def _save_workspace_file(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)

        path_str = body.get("path", "")
        content = body.get("content")
        if not path_str or content is None:
            return JSONResponse({"error": "path and content required"}, status_code=400)

        max_size = self.config.max_workspace_file_size
        if len(content) > max_size:
            return JSONResponse({"error": f"content too large (max {max_size // 1024}KB)"}, status_code=413)

        resolved, path_err = self._validate_workspace_path(adapter, path_str)
        if path_err:
            return path_err
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        if not _is_text_mime(resolved):
            return JSONResponse({"error": "file type not editable"}, status_code=400)

        try:
            resolved.write_text(content, encoding="utf-8")
        except OSError as e:
            return JSONResponse({"error": f"write failed: {e}"}, status_code=500)

        return JSONResponse({"status": "saved"})

    async def _attach_workspace_file(self, request: Request) -> JSONResponse:
        adapter, err = self._get_adapter(request)
        if err:
            return err

        path_str = request.query_params.get("path", "")
        if not path_str:
            return JSONResponse({"error": "path parameter required"}, status_code=400)

        resolved, path_err = self._validate_workspace_path(adapter, path_str)
        if path_err:
            return path_err
        if not resolved.exists():
            return JSONResponse({"error": "file not found"}, status_code=404)
        if not resolved.is_file():
            return JSONResponse({"error": "not a file"}, status_code=400)

        uploads_dir = adapter.agent_config.workspace_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        dest, dedup_err = _deduplicate_dest(uploads_dir, resolved.name)
        if dedup_err:
            return JSONResponse({"error": dedup_err}, status_code=409)
        if not dest.resolve().is_relative_to(uploads_dir.resolve()):
            return JSONResponse({"error": "invalid filename"}, status_code=400)

        try:
            shutil.copy2(resolved, dest)
        except OSError as e:
            return JSONResponse({"error": f"copy failed: {e}"}, status_code=500)

        file_size = dest.stat().st_size
        mime_type, content_type = _file_handler._detect_content_type(dest)
        context_attach = _should_context_attach(dest, file_size)

        return JSONResponse(
            {
                "files": [
                    {
                        "name": dest.name,
                        "content_type": content_type.value,
                        "mime_type": mime_type,
                        "size": file_size,
                        "context_attach": context_attach,
                    }
                ]
            }
        )
