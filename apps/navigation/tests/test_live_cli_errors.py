"""Setup errors expose bounded protocol codes, never provider response text."""

from enum import IntEnum

import pytest

from duck_nav import live

PRIVATE = "https://provider.invalid/?key=PRIVATE_KEY provider response body"


class Status(IntEnum):
    RATE_LIMITED = 429
    INTERNAL_ERROR = 1011


class UnsafeInteger(int):
    def __int__(self):
        raise AssertionError("Do not invoke custom conversion")

    def __str__(self):
        raise AssertionError("Do not format an untrusted value")


class UnsafeObject:
    def __int__(self):
        raise AssertionError("Do not coerce arbitrary provider values")

    def __str__(self):
        raise AssertionError("Do not format arbitrary provider values")


class ProviderError(Exception):
    def __str__(self):
        raise AssertionError("Do not format provider error text")


def invoke_main(monkeypatch, capsys, error):
    async def failed_run(_args):
        raise error

    monkeypatch.setattr(live, "run", failed_run)
    monkeypatch.setattr("sys.argv", ["duck-voice"])
    with pytest.raises(SystemExit) as ended:
        live.main()
    assert ended.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert PRIVATE not in output.err
    assert "PRIVATE_KEY" not in output.err
    assert "provider.invalid" not in output.err
    return output.err


@pytest.mark.parametrize(
    "code", [100, 200, 599, 1000, 1011, 4999, Status.RATE_LIMITED, Status.INTERNAL_ERROR]
)
def test_setup_failure_reports_only_numeric_http_or_websocket_status(monkeypatch, capsys, code):
    error = ProviderError(PRIVATE)
    error.code = code
    output = invoke_main(monkeypatch, capsys, error)
    assert output == (
        f"ProviderError (protocol_code={int.__int__(code)}): voice session could not run; "
        "check connection/configuration\n"
    )


@pytest.mark.parametrize(
    "code",
    [
        None,
        True,
        False,
        -1,
        0,
        99,
        600,
        999,
        5000,
        10**100,
        1011.0,
        "1011",
        PRIVATE,
        UnsafeInteger(1011),
        UnsafeObject(),
        {"code": 1011, "body": PRIVATE},
    ],
    ids=[
        "none",
        "true",
        "false",
        "negative",
        "zero",
        "below-http",
        "above-http",
        "below-ws",
        "above-ws",
        "huge",
        "float",
        "numeric-string",
        "private-string",
        "int-subclass",
        "int-like-object",
        "dict",
    ],
)
def test_unsafe_or_out_of_range_codes_do_not_change_redacted_error(monkeypatch, capsys, code):
    error = ProviderError(PRIVATE)
    error.code = code
    assert invoke_main(monkeypatch, capsys, error) == (
        "ProviderError: voice session could not run; check connection/configuration\n"
    )


@pytest.mark.parametrize("property_raises", [False, True])
def test_absent_or_failing_code_attribute_remains_redacted(monkeypatch, capsys, property_raises):
    class FailingPropertyError(ProviderError):
        @property
        def code(self):
            raise RuntimeError(PRIVATE)

    error = FailingPropertyError(PRIVATE) if property_raises else ProviderError(PRIVATE)
    output = invoke_main(monkeypatch, capsys, error)
    assert "protocol_code" not in output
    assert "check connection/configuration" in output
