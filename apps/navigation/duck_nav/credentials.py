"""Load Gemini credentials without storing plaintext in the checkout."""

from __future__ import annotations

import argparse
import base64
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path

TARGET = "Pollen/Microduck/Gemini"


def _windows_host() -> bool:
    if sys.platform == "win32" or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _credential_command(action: str, secret: str | None = None) -> str | None:
    if action not in {"read", "save"}:
        raise ValueError("unsupported credential operation")
    if not _windows_host():
        if action == "read":
            return None
        raise RuntimeError("Windows Credential Manager requires Windows or WSL")
    executable = shutil.which("powershell.exe")
    if not executable:
        raise RuntimeError("Windows PowerShell is unavailable; enable WSL Windows interop")
    # Only public source and a fixed operation go in the command line. The key is
    # carried on stdin for saving and a captured stdout pipe for loading.
    source = Path(__file__).with_name("windows_credential.ps1").read_text()
    source = f"$operation = '{action}'\n" + source
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    try:
        completed = subprocess.run(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            input=secret or "",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # Subprocess exceptions may retain stdin/stdout. Never include them in
        # the public error, log or exception chain.
        raise RuntimeError("Windows credential operation failed or timed out") from None
    if completed.returncode == 3 and action == "read":
        return None
    if completed.returncode:
        raise RuntimeError("Windows Credential Manager could not complete the operation")
    return completed.stdout.strip() or None


def load_gemini_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if value := os.environ.get(name, "").strip():
            return value
    return _credential_command("read")


def save_gemini_key(secret: str) -> None:
    secret = secret.strip()
    if not secret or len(secret) > 1024 or any(c.isspace() for c in secret):
        raise ValueError("Paste only the API key, without its variable name or whitespace")
    _credential_command("save", secret)
    # Read the vault directly: an environment override must not mask a failed save.
    if _credential_command("read") != secret:
        raise RuntimeError("Credential read-back verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["save", "status"])
    args = parser.parse_args()
    try:
        if args.action == "save":
            # Never fall back to echoed stdin for interactive setup.
            if not sys.stdin.isatty():
                raise RuntimeError("Run save in an interactive terminal for hidden key entry")
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                save_gemini_key(getpass.getpass("Gemini API key (hidden): "))
            print(f"Saved and verified Windows credential: {TARGET}")
        else:
            stored = _credential_command("read") is not None
            print(f"Windows credential {TARGET}: {'saved' if stored else 'not found'}")
            override = any(
                os.environ.get(n, "").strip() for n in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
            )
            if override:
                print("An environment key currently overrides the saved credential.")
            return 0 if stored else 1
    except (RuntimeError, ValueError, getpass.GetPassWarning) as error:
        print(str(error), file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("Credential setup cancelled", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
