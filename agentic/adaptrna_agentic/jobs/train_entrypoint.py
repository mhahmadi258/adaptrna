"""Thin training entrypoint — the MASTER_PLAN §3.4 seam.

Imports every registered custom-task package so `@register_task` has fired, then delegates
to the engine's own training CLI unchanged. The JobRunner always launches through here, so
Phase 6's generated task modules become runnable by adding them to `CUSTOM_TASK_MODULES`
— with no change to the runner, the plan format, or the engine.

    python -m adaptrna_agentic.jobs.train_entrypoint --task splice_site --use_lora ...

Also writes `<output_dir>/exit_code` on completion, so job state survives a lost PID.
"""

from pathlib import Path
from typing import List, Optional
import sys

#: Task modules generated/added outside the engine. Empty until Phase 6.
CUSTOM_TASK_MODULES: List[str] = []


def _import_custom_tasks() -> None:
    import importlib

    for module_path in CUSTOM_TASK_MODULES:
        importlib.import_module(module_path)


def _output_dir(argv: List[str]) -> Optional[Path]:
    if "--output_dir" in argv:
        index = argv.index("--output_dir")
        if index + 1 < len(argv):
            return Path(argv[index + 1])
    return None


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    _import_custom_tasks()

    from rinalmo_hub.cli.train import main as train_main

    code = 1
    try:
        code = train_main(argv) or 0
    finally:
        directory = _output_dir(argv)
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "exit_code").write_text(f"{code}\n")

    return code


if __name__ == "__main__":
    raise SystemExit(main())
