"""Run generated code in a bounded subprocess.

**This is accident-isolation, not adversarial sandboxing** — and the distinction is
deliberate (plans/PHASE_6_TOOLSMITH_VERIFIER.md §2). The code executed here was written by
this project's own model, from the user's description, on the user's machine, and a human
reviews the diff before it lands anywhere permanent. The realistic risks are therefore
accidents: an infinite loop, runaway memory, a stray write. Those are what this module
stops. It does **not** stop deliberately hostile code; if generated code ever starts
arriving from an untrusted source, the upgrade path is a container, not a bigger `rlimit`.

The human diff gate is the real security boundary.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import resource
import subprocess
import sys
import tempfile

#: The runner prints its report on one line behind this marker, so ordinary stdout
#: (progress bars, warnings, torch chatter) cannot corrupt the result.
RESULT_MARKER = "__ADAPTRNA_RESULT__"

DEFAULT_TIMEOUT = 600

#: `RLIMIT_AS` caps *virtual* address space, and PyTorch reserves far more of it than it
#: uses — measured here (2026-08-12): 10.4 GiB just to import torch and build a nano
#: model with 128 OpenMP threads, 5.5 GiB with the thread caps below. The limit is
#: therefore a bound on pathological runaway (a datamodule that reads a 400 GB array),
#: not a tight cap; CPU time is the better guard against infinite loops.
DEFAULT_MEMORY_MB = 32768
DEFAULT_CPU_SECONDS = 900
DEFAULT_FILE_SIZE_MB = 512


@dataclass
class SandboxResult:
    ok: bool
    returncode: Optional[int]
    timed_out: bool
    stdout: str
    stderr: str
    payload: Optional[Dict[str, Any]] = None
    limits: Dict[str, Any] = field(default_factory=dict)

    @property
    def failure_summary(self) -> str:
        if self.timed_out:
            return "the script exceeded its time limit (possible infinite loop)"
        if self.payload is None:
            tail = (self.stderr or self.stdout).strip().splitlines()[-8:]
            return "the script produced no result payload: " + " / ".join(tail)
        return ""


def _limiter(memory_mb: int, cpu_seconds: int, file_size_mb: int):
    """Applied in the child between fork and exec."""

    def apply() -> None:
        os.setsid()  # own process group, so a timeout can kill the whole tree
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024**2,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_mb * 1024**2,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def run_python(
    args: List[str],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    file_size_mb: int = DEFAULT_FILE_SIZE_MB,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> SandboxResult:
    """
    Run `python <args>` under limits, in a temporary working directory by default.

    Returns a `SandboxResult`; nothing raises for the child's misbehaviour — a crash, a
    hang and a failed check are all *data*, because they are exactly what the verification
    report needs to describe.
    """
    child_env = {
        **os.environ,
        # The harness runs on CPU: no CUDA context means the address-space limit is a
        # meaningful bound rather than a fight with the driver's virtual reservations.
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # Each OpenMP thread reserves stack and arena: 128 threads cost ~5 GiB of virtual
        # address space for no benefit on a nano model with a two-sequence batch.
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        **(env or {}),
    }

    limits = {"timeout_s": timeout, "memory_mb": memory_mb,
              "cpu_seconds": cpu_seconds, "file_size_mb": file_size_mb}

    with tempfile.TemporaryDirectory(prefix="adaptrna-sandbox-") as scratch:
        try:
            completed = subprocess.run(
                [sys.executable, *args],
                cwd=str(cwd or scratch),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limiter(memory_mb, cpu_seconds, file_size_mb),
            )
        except subprocess.TimeoutExpired as expired:
            return SandboxResult(
                ok=False, returncode=None, timed_out=True,
                stdout=_text(expired.stdout), stderr=_text(expired.stderr),
                limits=limits,
            )

    payload = extract_payload(completed.stdout)

    return SandboxResult(
        ok=completed.returncode == 0 and payload is not None,
        returncode=completed.returncode,
        timed_out=False,
        stdout=completed.stdout,
        stderr=completed.stderr,
        payload=payload,
        limits=limits,
    )


def extract_payload(stdout: str) -> Optional[Dict[str, Any]]:
    """The last marked result line in stdout, parsed."""
    payload = None
    for line in (stdout or "").splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                payload = json.loads(line[len(RESULT_MARKER):])
            except json.JSONDecodeError:
                payload = None

    return payload


def emit_payload(payload: Dict[str, Any]) -> None:
    """Called by the child to report its result."""
    print(RESULT_MARKER + json.dumps(payload, default=str), flush=True)


def _text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode(errors="replace")
