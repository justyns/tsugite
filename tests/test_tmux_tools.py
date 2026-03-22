"""Tests for tmux session management tools."""

import json
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from tsugite.tools.tmux import (
    _get_session_status,
    _list_managed_sessions,
    _strip_ansi,
    _validate_name,
    get_tmux_sessions,
    tmux_create,
    tmux_kill,
    tmux_list,
    tmux_read,
    tmux_send,
)


@pytest.fixture
def mock_metadata(tmp_path, monkeypatch):
    """Redirect metadata and log paths to tmp_path."""
    meta_dir = tmp_path / "tmux"
    log_dir = tmp_path / "tmux-logs"
    meta_dir.mkdir()
    log_dir.mkdir()

    monkeypatch.setattr("tsugite.tools.tmux._get_metadata_path", lambda: meta_dir / "sessions.json")
    monkeypatch.setattr("tsugite.tools.tmux._get_log_dir", lambda: log_dir)
    return tmp_path


def _make_run_result(returncode=0, stdout="", stderr=""):
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class TestStripAnsi:
    def test_strips_sgr_sequences(self):
        assert _strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strips_bold_and_color(self):
        assert _strip_ansi("\x1b[1;32mgreen bold\x1b[0m") == "green bold"

    def test_strips_osc_sequences(self):
        assert _strip_ansi("\x1b]0;title\x07text") == "text"

    def test_strips_charset_designator(self):
        assert _strip_ansi("\x1b(Btext") == "text"

    def test_passthrough_clean_text(self):
        assert _strip_ansi("hello world") == "hello world"

    def test_mixed_ansi(self):
        text = "\x1b[1m\x1b[32mOK\x1b[0m: \x1b[34mtest\x1b[0m passed"
        assert _strip_ansi(text) == "OK: test passed"


class TestValidateName:
    def test_valid_names(self):
        for name in ["test", "my-session", "project_1", "A-b_C-3"]:
            _validate_name(name)

    def test_invalid_names(self):
        for name in ["has space", "semi;colon", "pipe|char", "slash/path", ""]:
            with pytest.raises(ValueError, match="Invalid session name"):
                _validate_name(name)


class TestGetSessionStatus:
    @patch("tsugite.tools.tmux._get_pane_command")
    def test_idle_when_shell(self, mock_cmd):
        for shell in ["bash", "zsh", "sh", "fish"]:
            mock_cmd.return_value = shell
            assert _get_session_status("tsu-test") == "idle"

    @patch("tsugite.tools.tmux._get_pane_command")
    def test_active_when_process(self, mock_cmd):
        mock_cmd.return_value = "python3"
        assert _get_session_status("tsu-test") == "active: python3"

    @patch("tsugite.tools.tmux._get_pane_command")
    def test_idle_when_empty(self, mock_cmd):
        mock_cmd.return_value = ""
        assert _get_session_status("tsu-test") == "idle"


class TestTmuxCreate:
    @patch("tsugite.tools.tmux._session_exists", return_value=False)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_create_session(self, mock_run, mock_exists, mock_metadata):
        mock_run.return_value = _make_run_result()

        result = tmux_create("test")

        assert result["name"] == "test"
        assert result["tmux_session"] == "tsu-test"
        assert result["status"] == "created"
        assert "log_file" in result

        calls = mock_run.call_args_list
        assert calls[0] == call(
            ["tmux", "new-session", "-d", "-s", "tsu-test", "-x", "200", "-y", "50"],
            capture_output=True,
            text=True,
        )
        assert calls[1].args[0][:4] == ["tmux", "pipe-pane", "-t", "tsu-test"]

    @patch("tsugite.tools.tmux._session_exists", return_value=False)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_create_with_command(self, mock_run, mock_exists, mock_metadata):
        mock_run.return_value = _make_run_result()

        tmux_create("test", command="htop")

        new_session_call = mock_run.call_args_list[0]
        assert "htop" in new_session_call.args[0]

    @patch("tsugite.tools.tmux._session_exists", return_value=True)
    def test_create_already_exists(self, mock_exists):
        with pytest.raises(RuntimeError, match="already exists"):
            tmux_create("test")

    def test_create_invalid_name(self):
        with pytest.raises(ValueError, match="Invalid session name"):
            tmux_create("bad name")

    @patch("tsugite.tools.tmux._session_exists", return_value=False)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_create_saves_metadata(self, mock_run, mock_exists, mock_metadata):
        mock_run.return_value = _make_run_result()

        tmux_create("myproject", command="python3")

        meta_path = mock_metadata / "tmux" / "sessions.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert "myproject" in data
        assert data["myproject"]["command"] == "python3"
        assert data["myproject"]["prefixed_name"] == "tsu-myproject"

    @patch("tsugite.tools.tmux._session_exists", return_value=False)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_create_cleans_up_on_pipe_failure(self, mock_run, mock_exists, mock_metadata):
        mock_run.side_effect = [
            _make_run_result(),  # new-session succeeds
            _make_run_result(returncode=1, stderr="pipe error"),  # pipe-pane fails
            _make_run_result(),  # kill-session cleanup
        ]

        with pytest.raises(RuntimeError, match="Failed to set up logging"):
            tmux_create("test")

    @patch("tsugite.tools.tmux._session_exists", return_value=False)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_create_pipe_pane_uses_shlex_quote(self, mock_run, mock_exists, mock_metadata):
        mock_run.return_value = _make_run_result()

        tmux_create("test")

        pipe_call = mock_run.call_args_list[1]
        pipe_arg = pipe_call.args[0][5]  # The -o argument value
        assert pipe_arg.startswith("cat >> ")


class TestTmuxRead:
    @patch("tsugite.tools.tmux._session_exists", return_value=True)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_read_pane(self, mock_run, mock_exists):
        mock_run.return_value = _make_run_result(stdout="\x1b[32mhello\x1b[0m world\n")

        result = tmux_read("test", lines=10)

        assert result == "hello world\n"
        mock_run.assert_called_once_with(
            ["tmux", "capture-pane", "-t", "tsu-test", "-p", "-S", "-10"],
            capture_output=True,
            text=True,
        )

    def test_read_log(self, mock_metadata):
        log_dir = mock_metadata / "tmux-logs"
        log_file = log_dir / "test.log"
        log_file.write_text("line1\nline2\nline3\n\x1b[31mline4\x1b[0m\n")

        result = tmux_read("test", lines=2, source="log")

        assert result == "line3\nline4\n"

    @patch("tsugite.tools.tmux._session_exists", return_value=False)
    def test_read_nonexistent_pane(self, mock_exists):
        with pytest.raises(RuntimeError, match="not found"):
            tmux_read("nonexistent")

    def test_read_nonexistent_log(self, mock_metadata):
        with pytest.raises(RuntimeError, match="No log file"):
            tmux_read("nonexistent", source="log")

    def test_read_invalid_source(self):
        with pytest.raises(ValueError, match="Invalid source"):
            tmux_read("test", source="invalid")

    @patch("tsugite.tools.tmux._session_exists", return_value=True)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_read_clamps_lines(self, mock_run, mock_exists):
        mock_run.return_value = _make_run_result(stdout="text\n")

        tmux_read("test", lines=99999)

        args = mock_run.call_args.args[0]
        assert "-5000" in args


class TestTmuxSend:
    @patch("tsugite.tools.tmux._session_exists", return_value=True)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_send_with_enter(self, mock_run, mock_exists):
        mock_run.return_value = _make_run_result()

        result = tmux_send("test", "ls -la")

        mock_run.assert_called_once_with(
            ["tmux", "send-keys", "-t", "tsu-test", "ls -la", "Enter"],
            capture_output=True,
            text=True,
        )
        assert "command" in result

    @patch("tsugite.tools.tmux._session_exists", return_value=True)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_send_without_enter(self, mock_run, mock_exists):
        mock_run.return_value = _make_run_result()

        result = tmux_send("test", "q", enter=False)

        mock_run.assert_called_once_with(
            ["tmux", "send-keys", "-t", "tsu-test", "q"],
            capture_output=True,
            text=True,
        )
        assert "keys" in result

    @patch("tsugite.tools.tmux._session_exists", return_value=False)
    def test_send_nonexistent(self, mock_exists):
        with pytest.raises(RuntimeError, match="not found"):
            tmux_send("nonexistent", "hello")


class TestListManagedSessions:
    """Tests for the shared _list_managed_sessions helper used by tmux_list and get_tmux_sessions."""

    @patch("tsugite.tools.tmux.subprocess.run")
    def test_filters_by_prefix(self, mock_run, mock_metadata):
        mock_run.return_value = _make_run_result(
            stdout="tsu-project1\tbash\nuser-session\tvim\ntsu-project2\tpython3\n"
        )

        meta_path = mock_metadata / "tmux" / "sessions.json"
        meta_path.write_text(
            json.dumps(
                {
                    "project1": {"command": "htop", "created_at": "2026-01-01T00:00:00", "log_file": "/tmp/p1.log"},
                    "project2": {"command": None, "created_at": "2026-01-02T00:00:00", "log_file": "/tmp/p2.log"},
                }
            )
        )

        result = _list_managed_sessions()

        assert len(result) == 2
        names = [s["name"] for s in result]
        assert "project1" in names
        assert "project2" in names

    @patch("tsugite.tools.tmux.subprocess.run")
    def test_idle_status_for_shell(self, mock_run, mock_metadata):
        mock_run.return_value = _make_run_result(stdout="tsu-test\tbash\n")

        result = _list_managed_sessions()

        assert result[0]["status"] == "idle"

    @patch("tsugite.tools.tmux.subprocess.run")
    def test_active_status_for_process(self, mock_run, mock_metadata):
        mock_run.return_value = _make_run_result(stdout="tsu-test\tpython3\n")

        result = _list_managed_sessions()

        assert result[0]["status"] == "active: python3"

    @patch("tsugite.tools.tmux.subprocess.run")
    def test_no_server(self, mock_run):
        mock_run.return_value = _make_run_result(returncode=1, stderr="no server running")

        assert _list_managed_sessions() == []

    @patch("tsugite.tools.tmux.subprocess.run")
    def test_no_managed_sessions(self, mock_run, mock_metadata):
        mock_run.return_value = _make_run_result(stdout="user-session\tbash\n")

        assert _list_managed_sessions() == []


class TestTmuxList:
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_list_delegates_to_shared_helper(self, mock_run, mock_metadata):
        mock_run.return_value = _make_run_result(
            stdout="tsu-project1\tbash\ntsu-project2\thtop\n"
        )

        result = tmux_list()

        assert len(result) == 2
        assert result[0]["status"] == "idle"
        assert result[1]["status"] == "active: htop"

    @patch("tsugite.tools.tmux.subprocess.run")
    def test_list_no_server(self, mock_run):
        mock_run.return_value = _make_run_result(returncode=1)

        assert tmux_list() == []


class TestTmuxKill:
    @patch("tsugite.tools.tmux._session_exists", return_value=True)
    @patch("tsugite.tools.tmux.subprocess.run")
    def test_kill_session(self, mock_run, mock_exists, mock_metadata):
        mock_run.return_value = _make_run_result()

        meta_path = mock_metadata / "tmux" / "sessions.json"
        meta_path.write_text(json.dumps({"test": {"command": "htop"}}))

        result = tmux_kill("test")

        mock_run.assert_called_once_with(
            ["tmux", "kill-session", "-t", "tsu-test"],
            capture_output=True,
            text=True,
        )
        assert "terminated" in result

        data = json.loads(meta_path.read_text())
        assert "test" not in data

    @patch("tsugite.tools.tmux._session_exists", return_value=False)
    def test_kill_nonexistent(self, mock_exists):
        with pytest.raises(RuntimeError, match="not found"):
            tmux_kill("nonexistent")


class TestGetTmuxSessions:
    @patch("tsugite.tools.tmux.shutil.which", return_value=None)
    def test_no_tmux_installed(self, mock_which):
        assert get_tmux_sessions() == []

    @patch("tsugite.tools.tmux.subprocess.run")
    @patch("tsugite.tools.tmux.shutil.which", return_value="/usr/bin/tmux")
    def test_filters_prefix(self, mock_which, mock_run, mock_metadata):
        mock_run.return_value = _make_run_result(stdout="tsu-myproject\tbash\nother-session\tvim\n")

        result = get_tmux_sessions()

        assert len(result) == 1
        assert result[0]["name"] == "myproject"
        assert "created_at" not in result[0]
        assert "log_file" not in result[0]

    @patch("tsugite.tools.tmux.subprocess.run")
    @patch("tsugite.tools.tmux.shutil.which", return_value="/usr/bin/tmux")
    def test_idle_status(self, mock_which, mock_run, mock_metadata):
        mock_run.return_value = _make_run_result(stdout="tsu-test\tzsh\n")

        result = get_tmux_sessions()

        assert result[0]["status"] == "idle"

    @patch("tsugite.tools.tmux.subprocess.run")
    @patch("tsugite.tools.tmux.shutil.which", return_value="/usr/bin/tmux")
    def test_active_status(self, mock_which, mock_run, mock_metadata):
        mock_run.return_value = _make_run_result(stdout="tsu-test\thtop\n")

        result = get_tmux_sessions()

        assert result[0]["status"] == "active: htop"

    @patch("tsugite.tools.tmux.subprocess.run")
    @patch("tsugite.tools.tmux.shutil.which", return_value="/usr/bin/tmux")
    def test_no_server_running(self, mock_which, mock_run):
        mock_run.return_value = _make_run_result(returncode=1)

        assert get_tmux_sessions() == []
