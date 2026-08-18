"""Thin training entrypoint — the MASTER_PLAN §3.4 seam.

Imports every registered custom-task package so `@register_task` has fired, then delegates
to the engine's own training CLI unchanged. The JobRunner always launches through here, so
Phase 6's generated task modules become runnable by adding them to `CUSTOM_TASK_MODULES`
— with no change to the runner, the plan format, or the engine.

    python -m adaptrna_agentic.jobs.train_entrypoint --task <your_task> --use_lora ...

Also writes `<output_dir>/exit_code` on completion, so job state survives a lost PID.
"""

from pathlib import Path
from typing import List, Optional
import sys


def _import_custom_tasks() -> None:
    """Import generated tasks from `<repo>/adaptrna_custom/tasks/` (Phase 6)."""
    from adaptrna_agentic.codegen.discovery import describe_failures, load_all

    failures = load_all()
    if failures:
        # Reported, not fatal: the engine will raise a clear "unknown task" error if the
        # run actually needed one of these.
        print(f"warning: custom task(s) failed to import — {describe_failures(failures)}",
              file=sys.stderr)


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
