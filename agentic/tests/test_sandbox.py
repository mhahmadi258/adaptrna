"""The sandbox bounds accidents: hangs, memory runaways, crashes — all reported as data."""

from pathlib import Path

import pytest

from adaptrna_agentic.codegen.sandbox import (
    RESULT_MARKER,
    extract_payload,
    run_python,
)


def script(tmp_path, body: str) -> str:
    path = tmp_path / "script.py"
    path.write_text(body)
    return str(path)


def test_payload_round_trip(tmp_path):
    body = (
        "from adaptrna_agentic.codegen.sandbox import emit_payload\n"
        "emit_payload({'checks': [{'name': 'x', 'status': 'pass'}]})\n"
    )

    result = run_python([script(tmp_path, body)], timeout=60)

    assert result.ok
    assert result.returncode == 0
    assert result.payload["checks"][0]["status"] == "pass"


def test_noisy_stdout_does_not_corrupt_the_payload(tmp_path):
    body = (
        "from adaptrna_agentic.codegen.sandbox import emit_payload\n"
        "print('loading model...'); print('100%|#####| 10/10')\n"
        "emit_payload({'ok': True})\n"
        "print('trailing chatter')\n"
    )

    result = run_python([script(tmp_path, body)], timeout=60)

    assert result.payload == {"ok": True}


def test_timeout_kills_a_hanging_script(tmp_path):
    result = run_python([script(tmp_path, "while True:\n    pass\n")], timeout=3)

    assert result.timed_out is True
    assert result.ok is False
    assert "time limit" in result.failure_summary


def test_memory_limit_stops_a_runaway(tmp_path):
    body = "x = bytearray(2 * 1024**3)\nprint(len(x))\n"

    result = run_python([script(tmp_path, body)], timeout=60, memory_mb=256)

    assert result.ok is False
    assert result.returncode != 0
    assert "MemoryError" in result.stderr or result.returncode != 0


def test_crash_is_reported_not_raised(tmp_path):
    body = "raise RuntimeError('generated code blew up')\n"

    result = run_python([script(tmp_path, body)], timeout=60)

    assert result.ok is False
    assert result.payload is None
    assert "generated code blew up" in result.stderr
    assert "no result payload" in result.failure_summary


def test_working_directory_is_temporary_by_default(tmp_path):
    body = (
        "import os, pathlib\n"
        "from adaptrna_agentic.codegen.sandbox import emit_payload\n"
        "pathlib.Path('side_effect.txt').write_text('x')\n"
        "emit_payload({'cwd': os.getcwd()})\n"
    )

    result = run_python([script(tmp_path, body)], timeout=60)

    assert "adaptrna-sandbox-" in result.payload["cwd"]
    # The scratch dir is gone, so the stray write went nowhere permanent.
    assert not Path(result.payload["cwd"]).exists()
    assert not (tmp_path / "side_effect.txt").exists()


def test_explicit_cwd_is_honored(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    body = (
        "import os\n"
        "from adaptrna_agentic.codegen.sandbox import emit_payload\n"
        "emit_payload({'cwd': os.getcwd()})\n"
    )

    result = run_python([script(tmp_path, body)], timeout=60, cwd=workdir)

    assert Path(result.payload["cwd"]).resolve() == workdir.resolve()


def test_cuda_is_hidden_from_the_child(tmp_path):
    body = (
        "import os\n"
        "from adaptrna_agentic.codegen.sandbox import emit_payload\n"
        "emit_payload({'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')})\n"
    )

    assert run_python([script(tmp_path, body)], timeout=60).payload["cuda"] == ""


def test_limits_are_reported(tmp_path):
    body = ("from adaptrna_agentic.codegen.sandbox import emit_payload\n"
            "emit_payload({})\n")

    result = run_python([script(tmp_path, body)], timeout=42, memory_mb=1234)

    assert result.limits["timeout_s"] == 42
    assert result.limits["memory_mb"] == 1234


def test_extract_payload_ignores_unmarked_json():
    assert extract_payload('{"not": "marked"}\n') is None
    assert extract_payload(f'{RESULT_MARKER}{{"a": 1}}\n') == {"a": 1}
