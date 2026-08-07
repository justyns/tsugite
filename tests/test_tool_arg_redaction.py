"""Sensitive tool arguments are redacted by key/path, not only by registered value.

`get_secret()` values are masked because the registry knows them. A bearer token
minted during a login flow is never registered, so before this the rendered
`http_request` call showed `{"Authorization": "Bearer <raw token>"}` in the live
event, in the persisted `code_execution.tool_calls`, and on replay.
"""

import pytest

from tsugite.secrets.redaction import redact_sensitive_obj


class TestBuiltinSensitiveKeys:
    def test_redacts_authorization_preserving_the_scheme(self):
        out = redact_sensitive_obj({"Authorization": "Bearer abc123", "Content-Type": "application/json"})
        assert out == {"Authorization": "Bearer ***", "Content-Type": "application/json"}

    def test_preserves_basic_and_other_schemes(self):
        out = redact_sensitive_obj({"Proxy-Authorization": "Basic Zm9vOmJhcg=="})
        assert out == {"Proxy-Authorization": "Basic ***"}

    def test_schemeless_values_redact_whole(self):
        assert redact_sensitive_obj({"Cookie": "session=abc; theme=dark"}) == {"Cookie": "***"}
        assert redact_sensitive_obj({"X-API-Key": "sk-live-1234"}) == {"X-API-Key": "***"}

    def test_key_matching_is_case_insensitive(self):
        out = redact_sensitive_obj({"authorization": "Bearer t", "SET-COOKIE": "sid=1"})
        assert out == {"authorization": "Bearer ***", "SET-COOKIE": "***"}

    def test_walks_nested_dicts_and_lists(self):
        out = redact_sensitive_obj(
            {
                "url": "https://api.example.com",
                "headers": {"Authorization": "Bearer abc", "Accept": "*/*"},
                "retries": [{"headers": {"X-Auth-Token": "tok"}}],
            }
        )
        assert out["headers"] == {"Authorization": "Bearer ***", "Accept": "*/*"}
        assert out["retries"][0]["headers"]["X-Auth-Token"] == "***"
        assert out["url"] == "https://api.example.com"

    def test_leaves_ordinary_values_alone(self):
        payload = {"path": "/tmp/x", "count": 3, "flag": True, "nothing": None}
        assert redact_sensitive_obj(payload) == payload

    def test_does_not_mutate_the_caller_object(self):
        original = {"headers": {"Authorization": "Bearer abc"}}
        redact_sensitive_obj(original)
        assert original["headers"]["Authorization"] == "Bearer abc"


class TestDeclaredSensitivePaths:
    def test_redacts_a_declared_nested_path(self):
        out = redact_sensitive_obj(
            {"body": {"password": "hunter2", "username": "jo"}},
            sensitive_paths=["body.password"],
        )
        assert out == {"body": {"password": "***", "username": "jo"}}

    def test_redacts_a_declared_top_level_arg(self):
        out = redact_sensitive_obj({"token": "abc", "name": "x"}, sensitive_paths=["token"])
        assert out == {"token": "***", "name": "x"}

    def test_declared_path_reaches_through_lists(self):
        out = redact_sensitive_obj(
            {"items": [{"token": "a"}, {"token": "b"}]},
            sensitive_paths=["items.token"],
        )
        assert [i["token"] for i in out["items"]] == ["***", "***"]

    def test_a_declared_path_does_not_redact_a_same_named_key_elsewhere(self):
        out = redact_sensitive_obj(
            {"body": {"token": "secret"}, "meta": {"token": "public-id"}},
            sensitive_paths=["body.token"],
        )
        assert out["body"]["token"] == "***"
        assert out["meta"]["token"] == "public-id"


class TestRedactionAtTheCallBoundary:
    """The audit surfaces: the live event, the per-call record history persists,
    and what replay therefore serves."""

    async def _run(self, code: str, tools: list):
        from tsugite.core.executor import LocalExecutor

        executor = LocalExecutor()
        executor.set_tools(tools)
        return await executor.execute(code)

    def test_tool_call_event_redacts_a_derived_bearer_token(self):
        from tsugite.events import ToolCallEvent

        event = ToolCallEvent(
            tool_name="http_request",
            arguments={
                "url": "https://api.example.com/v1/me",
                "headers": {"Authorization": "Bearer eyJraw.token", "Content-Type": "application/json"},
            },
        )
        assert event.arguments["headers"]["Authorization"] == "Bearer ***"
        assert event.arguments["headers"]["Content-Type"] == "application/json"
        assert "eyJraw.token" not in str(event.arguments)

    @staticmethod
    def _fake_http_tool():
        from tsugite.core.tools import create_tool_from_function

        def fake_http(url: str, headers: dict = None) -> str:
            """stub tool"""
            return "ok"

        return create_tool_from_function(fake_http)

    @pytest.mark.asyncio
    async def test_recorded_tool_call_arguments_are_redacted_before_repr(self):
        """`_jsonable_call_args` turns a nested dict into a capped repr string.
        Redaction must happen first, or the raw token is baked into that string
        and every later consumer (history, SSE, replay) shows it."""
        result = await self._run(
            "fake_http(url='https://api.example.com', headers={'Authorization': 'Bearer eyJraw.token',"
            " 'Content-Type': 'application/json'})",
            [self._fake_http_tool()],
        )
        recorded = result.tool_calls[0]["arguments"]
        assert "eyJraw.token" not in repr(recorded)
        assert "Bearer ***" in repr(recorded)
        assert "application/json" in repr(recorded)

    @pytest.mark.asyncio
    async def test_a_tool_can_declare_its_own_sensitive_argument(self):
        from tsugite.core.tools import create_tool_from_function
        from tsugite.tools import tool

        @tool(sensitive_args=["credentials.password", "api_key"])
        def _redaction_probe(api_key: str, credentials: dict) -> str:
            """stub tool"""
            return "ok"

        result = await self._run(
            "_redaction_probe(api_key='sk-live-1', credentials={'user': 'jo', 'password': 'hunter2'})",
            [create_tool_from_function(_redaction_probe)],
        )
        recorded = result.tool_calls[0]["arguments"]
        assert recorded["api_key"] == "***"
        assert "hunter2" not in repr(recorded)
        assert "jo" in repr(recorded)


class TestHttpResponseHeaders:
    def test_http_request_redacts_set_cookie_in_the_returned_headers(self, monkeypatch):
        """The response object is what the model sees and what str()s into the
        result summary, so a Set-Cookie must not survive it."""
        import tsugite.tools.http as http_mod

        class _Resp:
            status_code = 200
            headers = {"Set-Cookie": "session=abc123; HttpOnly", "Content-Type": "text/plain"}
            text = "ok"
            url = "https://example.com"

        monkeypatch.setattr(http_mod, "_simple_request", lambda *a, **k: _Resp())
        response = http_mod.http_request(url="https://example.com")
        assert response.headers["Set-Cookie"] == "***"
        assert response.headers["Content-Type"] == "text/plain"


class TestSubprocessExecutorPath:
    """The production default executor runs tools in a child process and builds
    its own per-call records there, so redaction has to reach that copy too."""

    @pytest.mark.asyncio
    async def test_child_process_records_are_redacted(self):
        from tsugite.core.subprocess_executor import SubprocessExecutor
        from tsugite.core.tools import Tool

        async def fake_http(url: str = "", headers: dict = None) -> str:
            return "ok"

        tool = Tool(
            name="fake_http",
            description="stub tool",
            parameters={"type": "object", "properties": {}, "required": []},
            function=fake_http,
        )
        tool._parent_only = True

        executor = SubprocessExecutor()
        executor.set_tools([tool])
        try:
            result = await executor.execute(
                "fake_http(url='https://api.example.com', headers={'Authorization': 'Bearer eyJraw.token',"
                " 'Content-Type': 'application/json'})"
            )
            assert result.error is None
            recorded = result.tool_calls[0]["arguments"]
            assert "eyJraw.token" not in repr(recorded)
            assert "Bearer ***" in repr(recorded)
            assert "application/json" in repr(recorded)
        finally:
            executor.cleanup()

    def test_check_url_redacts_set_cookie_too(self, monkeypatch):
        """The HEAD probe returns the same header dict shape and leaks the same way."""
        import tsugite.tools.http as http_mod

        class _Resp:
            status_code = 200
            headers = {"Set-Cookie": "session=abc123", "Content-Type": "text/html", "Content-Length": "9"}
            url = "https://example.com"

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def head(self, *a, **k):
                return _Resp()

        monkeypatch.setattr(http_mod.httpx, "Client", lambda *a, **k: _Client())
        out = http_mod.check_url(url="https://example.com")
        assert out["headers"]["Set-Cookie"] == "***"
        assert out["headers"]["Content-Type"] == "text/html"


class TestRealRegisteredTools:
    """The mechanism tests above all use synthetic tools. These pin what the
    tools we actually ship redact - the gap that let `sensitive_args=["headers"]`
    (a path naming a whole dict) blank out every header, including the
    non-sensitive ones the audit trail exists to show.

    Declarations are read off the module functions, not the registry: an autouse
    conftest fixture empties the registry per test, and re-registering with a
    bare `tool(fn)` would reset the very attribute under test.
    """

    def test_http_request_keeps_non_sensitive_headers_readable(self):
        from tsugite.tools.http import http_request

        redacted = redact_sensitive_obj(
            {
                "url": "https://api.example.com/v1/me",
                "headers": {
                    "Authorization": "Bearer eyJraw.token",
                    "Cookie": "sid=1",
                    "X-API-Key": "sk-live",
                    "Content-Type": "application/json",
                },
            },
            getattr(http_request, "_sensitive_args", ()),
        )
        assert redacted["headers"]["Authorization"] == "Bearer ***"
        assert redacted["headers"]["Cookie"] == "***"
        assert redacted["headers"]["X-API-Key"] == "***"
        assert redacted["headers"]["Content-Type"] == "application/json"
        assert redacted["url"] == "https://api.example.com/v1/me"

    def test_no_shipped_tool_declares_a_path_naming_a_whole_container(self):
        """A declared path that names a dict/list argument redacts it wholesale.
        Declare the leaf inside it, or rely on the built-in keys, which already
        reach any depth."""
        import inspect
        import typing

        from tsugite.tools import _ensure_tools_loaded

        _ensure_tools_loaded()

        def is_container(annotation) -> bool:
            if annotation in (dict, list):
                return True
            origin = typing.get_origin(annotation)
            if origin in (dict, list):
                return True
            if origin is typing.Union:
                return any(is_container(a) for a in typing.get_args(annotation))
            return False

        for module in _tool_modules():
            for name, fn in vars(module).items():
                declared = getattr(fn, "_sensitive_args", ()) if callable(fn) else ()
                if not declared:
                    continue
                params = inspect.signature(fn).parameters
                for path in declared:
                    if "." in path:
                        continue
                    param = params.get(path)
                    assert param is None or not is_container(param.annotation), (
                        f"{name} declares '{path}', which names a whole container argument; "
                        "declare the leaf paths inside it instead"
                    )


def _tool_modules():
    """The built-in tool modules, for auditing shipped @tool declarations."""
    import importlib
    import pkgutil

    import tsugite.tools as tools_pkg

    for info in pkgutil.iter_modules(tools_pkg.__path__):
        yield importlib.import_module(f"tsugite.tools.{info.name}")
