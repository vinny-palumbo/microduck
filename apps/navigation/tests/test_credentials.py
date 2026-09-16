"""Credential regressions never use the developer's vault or a real key."""

import base64
import subprocess
from types import SimpleNamespace

import pytest

from duck_nav import credentials


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_windows_host", lambda: True)
    monkeypatch.setattr(credentials.shutil, "which", lambda name: "powershell.exe")


def test_environment_precedence_never_reads_vault(monkeypatch):
    def forbidden(*args):
        pytest.fail("Environment override must not touch the vault")

    monkeypatch.setattr(credentials, "_credential_command", forbidden)
    monkeypatch.setenv("GEMINI_API_KEY", " first-key ")
    monkeypatch.setenv("GOOGLE_API_KEY", "second-key")
    assert credentials.load_gemini_key() == "first-key"
    monkeypatch.setenv("GEMINI_API_KEY", " ")
    assert credentials.load_gemini_key() == "second-key"


def test_save_uses_pipe_and_verifies_vault_despite_environment(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="private-fixture" if len(calls) == 2 else "")

    monkeypatch.setattr(credentials.subprocess, "run", run)
    monkeypatch.setenv("GEMINI_API_KEY", "wrong-environment-key")
    credentials.save_gemini_key("private-fixture")
    assert len(calls) == 2
    assert calls[0][1]["input"] == "private-fixture"
    assert calls[1][1]["input"] == ""
    for command, kwargs in calls:
        assert "private-fixture" not in " ".join(command)
        assert "private-fixture" not in base64.b64decode(command[-1]).decode("utf-16-le")
        assert kwargs["capture_output"]
        assert kwargs["timeout"] == 20
        assert not kwargs.get("shell")


@pytest.mark.parametrize("code", [1, 4])
def test_vault_errors_do_not_disclose_output(monkeypatch, code):
    monkeypatch.setattr(
        credentials.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=code, stdout="private-fixture", stderr="private-fixture"
        ),
    )
    with pytest.raises(RuntimeError) as caught:
        credentials.load_gemini_key()
    assert "private-fixture" not in str(caught.value)


def test_timeout_does_not_disclose_captured_key(monkeypatch):
    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("command", 20, output="private-fixture")

    monkeypatch.setattr(credentials.subprocess, "run", run)
    with pytest.raises(RuntimeError) as caught:
        credentials.load_gemini_key()
    assert "private-fixture" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_missing_credential_is_distinct_from_failure(monkeypatch):
    monkeypatch.setattr(
        credentials.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=3, stdout="", stderr=""),
    )
    assert credentials.load_gemini_key() is None
    with pytest.raises(RuntimeError):
        credentials.save_gemini_key("fixture")


def test_save_detects_readback_mismatch(monkeypatch):
    monkeypatch.setattr(credentials, "_credential_command", lambda *args: "other-key")
    with pytest.raises(RuntimeError, match="read-back"):
        credentials.save_gemini_key("fixture")


def test_non_windows_keeps_environment_only_workflow(monkeypatch):
    monkeypatch.setattr(credentials, "_windows_host", lambda: False)
    assert credentials.load_gemini_key() is None
    with pytest.raises(RuntimeError, match="requires Windows"):
        credentials.save_gemini_key("fixture")


def test_missing_interop_has_actionable_error(monkeypatch):
    monkeypatch.setattr(credentials.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="interop"):
        credentials.load_gemini_key()


@pytest.mark.parametrize("key", ["", " ", "GEMINI_API_KEY = fixture", "a" * 1025])
def test_bad_key_never_reaches_vault(monkeypatch, key):
    def forbidden(*args):
        pytest.fail("Invalid key must not reach the vault")

    monkeypatch.setattr(credentials, "_credential_command", forbidden)
    with pytest.raises(ValueError):
        credentials.save_gemini_key(key)


def test_status_never_prints_credential(monkeypatch, capsys):
    monkeypatch.setattr(credentials, "_credential_command", lambda *args: "private-fixture")
    monkeypatch.setattr(credentials.sys, "argv", ["duck-agent-key", "status"])
    assert credentials.main() == 0
    output = capsys.readouterr()
    assert "saved" in output.out
    assert "private-fixture" not in output.out + output.err


def test_setup_refuses_noninteractive_echo(monkeypatch, capsys):
    monkeypatch.setattr(credentials.sys, "argv", ["duck-agent-key", "save"])
    monkeypatch.setattr(credentials.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    assert credentials.main() == 1
    assert "interactive terminal" in capsys.readouterr().err
