"""TerminalsMixin: terminals HTTP handlers for HTTPServer."""

import asyncio
import json
import threading

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from tsugite_daemon.adapters.http.helpers import (
    logger,
    mounted_api_routes,
)


class TerminalsMixin:
    def _terminal_routes(self) -> list:
        return [
            *mounted_api_routes(
                "/api/terminals",
                "terminals",
                [
                    Route("/", self._api_list_terminals, methods=["GET"]),
                    Route("/", self._api_create_terminal, methods=["POST"]),
                    Route("/{terminal_id}", self._api_get_terminal, methods=["GET"]),
                    Route("/{terminal_id}/kill", self._api_kill_terminal, methods=["POST"]),
                    Route("/{terminal_id}/stdin", self._api_terminal_stdin, methods=["POST"]),
                    Route("/{terminal_id}/restart", self._api_restart_terminal, methods=["POST"]),
                    Route("/{terminal_id}/stream", self._api_terminal_stream, methods=["GET"]),
                ],
            ),
        ]

    def _terminal_to_dict(self, terminal) -> dict:
        from dataclasses import asdict

        proc = self.pty_manager.get(terminal.id) if self.pty_manager else None
        data = asdict(terminal)
        # Surface live runtime info even before the on_exit hook persists the
        # final counts. Stale-on-disk wins for terminated terminals (proc dropped).
        if proc is not None:
            data["bytes_out"] = max(data.get("bytes_out", 0), proc.bytes_out)
            data["lines_out"] = max(data.get("lines_out", 0), proc.lines_out)
            data["truncated"] = proc.truncated
            if proc.last_line:
                data["last_line"] = proc.last_line
        else:
            data.setdefault("truncated", False)
        return data

    def _emit_terminal_state(self, terminal_id: str, new_state: str) -> None:
        """on_state_change callback for spawn_terminal: broadcast a state change."""
        self.event_bus.emit("terminal_state", {"terminal_id": terminal_id, "state": new_state})

    async def _api_list_terminals(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        parent = request.query_params.get("parent_session_id")
        if parent:
            terminals = self.terminal_store.list_for_parent(parent)
        else:
            terminals = self.terminal_store.list_all()
        return JSONResponse({"terminals": [self._terminal_to_dict(t) for t in terminals]})

    async def _api_get_terminal(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        terminal = self.terminal_store.get(terminal_id)
        if terminal is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)
        return JSONResponse(self._terminal_to_dict(terminal))

    async def _api_create_terminal(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        cmd = (body.get("cmd") or "").strip()
        if not cmd:
            return JSONResponse({"error": "cmd is required"}, status_code=400)
        cwd = body.get("cwd")
        parent_session_id = body.get("parent_session_id")
        env = body.get("env") if isinstance(body.get("env"), dict) else None

        from tsugite_pty.terminal_runtime import spawn_terminal

        try:
            terminal = spawn_terminal(
                store=self.terminal_store,
                manager=self.pty_manager,
                cmd=cmd,
                cwd=cwd,
                env=env,
                parent_session_id=parent_session_id,
                on_state_change=self._emit_terminal_state,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.exception("Failed to spawn terminal")
            return JSONResponse({"error": str(e)}, status_code=500)
        return JSONResponse(self._terminal_to_dict(terminal), status_code=201)

    async def _api_kill_terminal(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        terminal = self.terminal_store.get(terminal_id)
        if terminal is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)
        self.pty_manager.kill(terminal_id)
        return JSONResponse({"status": "killed", "terminal_id": terminal_id})

    async def _api_terminal_stdin(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        terminal = self.terminal_store.get(terminal_id)
        if terminal is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        data = body.get("data", "")
        if not isinstance(data, str):
            return JSONResponse({"error": "data must be a string"}, status_code=400)
        written = self.pty_manager.write_stdin(terminal_id, data.encode("utf-8", errors="replace"))
        return JSONResponse({"status": "ok", "bytes_written": written})

    async def _api_restart_terminal(self, request: Request) -> JSONResponse:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        old = self.terminal_store.get(terminal_id)
        if old is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)

        from tsugite_pty.terminal_runtime import spawn_terminal
        from tsugite_pty.terminal_store import TerminalState

        # Refuse to restart a still-live PTY - caller should kill first to avoid
        # leaking the original process.
        if old.state not in (
            TerminalState.SUCCEEDED.value,
            TerminalState.FAILED.value,
            TerminalState.CANCELLED.value,
            TerminalState.STREAM_LOST.value,
        ):
            return JSONResponse(
                {"error": f"cannot restart terminal in '{old.state}' state; kill it first"},
                status_code=409,
            )

        try:
            new_terminal = spawn_terminal(
                store=self.terminal_store,
                manager=self.pty_manager,
                cmd=old.cmd,
                cwd=old.cwd,
                parent_session_id=old.parent_session_id,
                on_state_change=self._emit_terminal_state,
            )
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:
            logger.exception("Failed to restart terminal")
            return JSONResponse({"error": str(e)}, status_code=500)

        return JSONResponse(
            {**self._terminal_to_dict(new_terminal), "restarted_from": terminal_id},
            status_code=201,
        )

    async def _api_terminal_stream(self, request: Request) -> Response:
        if err := self._require_auth_and_terminals(request):
            return err
        terminal_id = request.path_params["terminal_id"]
        terminal = self.terminal_store.get(terminal_id)
        if terminal is None:
            return JSONResponse({"error": f"unknown terminal: {terminal_id}"}, status_code=404)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        closing = asyncio.Event()

        def _safe_put(payload) -> None:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Slow client; drop the chunk rather than backing up the daemon.
                pass

        def _push(payload: dict) -> None:
            if threading.current_thread() is threading.main_thread() and loop.is_running():
                try:
                    running = asyncio.get_running_loop()
                except RuntimeError:
                    running = None
                if running is loop:
                    _safe_put(payload)
                    return
            try:
                loop.call_soon_threadsafe(_safe_put, payload)
            except RuntimeError:
                pass

        def _close() -> None:
            # Guarantee stream teardown: enqueue the sentinel (best-effort under
            # backpressure) AND set `closing` so the generator breaks even if the
            # sentinel was dropped from a full queue.
            try:
                loop.call_soon_threadsafe(_safe_put, None)
                loop.call_soon_threadsafe(closing.set)
            except RuntimeError:
                pass

        # Emit the current state up front so a late-connecting client doesn't
        # need a separate fetch to know whether the terminal is still running.
        # Pushed before any output/exit events so consumers can size their
        # rendering up front.
        _push({"type": "state", "state": terminal.state})

        proc = self.pty_manager.get(terminal_id)
        unsub = None
        exit_unsub = None
        if proc is not None:

            def _on_chunk(chunk: bytes) -> None:
                _push({"type": "output", "chunk": chunk.decode("utf-8", errors="replace")})

            def _on_exit(p) -> None:
                _push({"type": "exit", "exit_code": p.exit_code})
                _close()

            # Snapshot + subscribe atomically so a chunk produced between the two
            # is never lost. The buffer is the ring-capped window (1 MB default);
            # anything older is gone, hence the `truncated` flag the frontend uses.
            existing, unsub = proc.snapshot_and_subscribe(_on_chunk)
            if existing:
                _push({"type": "output", "chunk": existing.decode("utf-8", errors="replace"), "replay": True})
            # on_exit fires synchronously if the process has already exited, which
            # is fine - it just queues the exit event after what we already pushed.
            exit_unsub = proc.on_exit(_on_exit)
        else:
            # No live PTY (already evicted post-exit, or pre-spawn failure).
            # Try the on-disk log first so re-opening an old terminal still
            # shows what it printed. terminal_runtime writes this when the
            # process exits with a non-empty buffer.
            current = self.terminal_store.get(terminal_id)
            log_path = self.terminal_store.log_path(terminal_id)
            try:
                if log_path.is_file():
                    contents = log_path.read_bytes()
                    if contents:
                        _push(
                            {
                                "type": "output",
                                "chunk": contents.decode("utf-8", errors="replace"),
                                "replay": True,
                            }
                        )
            except OSError:
                logger.exception("Failed to read terminal log for '%s'", terminal_id)
            _push({"type": "exit", "exit_code": current.exit_code})
            _close()

        async def generator():
            try:
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        # If close fired but its sentinel was dropped under
                        # backpressure, `closing` still tears the stream down.
                        if closing.is_set() and queue.empty():
                            break
                        yield ": keepalive\n\n"
                        continue
                    if payload is None:
                        break
                    event_type = payload.pop("type", "message")
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(payload)}\n\n"
            finally:
                for fn in (unsub, exit_unsub):
                    if fn is not None:
                        try:
                            fn()
                        except Exception:
                            pass

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
