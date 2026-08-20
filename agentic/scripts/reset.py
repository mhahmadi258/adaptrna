#!/usr/bin/env python
"""Reset the install to fresh — everything a `git clone` + `pip install -e` would leave,
except `outputs/`, which holds training results and is never touched by this script.

Dry run by default; nothing is deleted without `--yes`, the same convention as
`toolhub prune` (`agentic/adaptrna_agentic/toolhub/prune.py`). Kept as a standalone script
rather than a `toolhub` subcommand on purpose: `prune` is documented as "the one
destructive command, and the only one" in the shipped CLI, and a whole-install wipe is a
bigger hammer than that surface should advertise.

    python agentic/scripts/reset.py                 # prints the plan, deletes nothing
    python agentic/scripts/reset.py --yes           # applies it

What it clears: every registered tool (manifest entries + registry-owned adapter copies),
staged-but-never-landed generated code, the job store, the chat/session database, and
generated task/tool code under `adaptrna_custom/` (keeping the four git-tracked skeleton
files — `.gitignore` excludes the rest of that directory, so deleting them would not be
recoverable with `git checkout`).

What it preserves: `outputs/` (training results), `dataset/`, `weights/`, `.venv/`, `.env`
(holds the live API key — never deleted, never printed), the cached backbone checkpoint
under `~/.cache/rinalmo_pretrained/`, and the manifest's `backbone` block (pass
`--forget-backbone` to drop that too).

Clearing `jobs_data/jobs.json` while keeping `outputs/` orphans the run directories: their
metrics files are untouched, but `analyze_run` on an old job id will no longer resolve
since there is no longer a job record to look it up by.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import shutil
import sys
import tempfile

from adaptrna_agentic.settings import REPO_ROOT
from adaptrna_agentic.toolhub import manifest as manifest_module
from adaptrna_agentic.toolhub import prune as prune_module
from adaptrna_agentic.toolhub.prune import Candidate

# The only adaptrna_custom/ files that are actually git-tracked (.gitignore:16 ignores the
# rest of the directory) -- everything else here is generated and safe to delete.
CUSTOM_KEEP = {
    Path("README.md"),
    Path("__init__.py"),
    Path("tasks/__init__.py"),
    Path("tools/__init__.py"),
}


def _dir_size(path: Path) -> int:
    return prune_module._dir_size(path)


def _remove(candidate: Candidate) -> None:
    prune_module._remove(candidate)


# --------------------------------------------------------------------------- candidates

def _manifest_candidates(data_dir: Path, forget_backbone: bool) -> List[Candidate]:
    manifest_path = data_dir / "tools.json"
    if not manifest_path.exists():
        return []

    man = manifest_module.Manifest.load(data_dir)
    if not man.tools and not forget_backbone:
        return []

    label = "tools.json (backbone kept)" if not forget_backbone else "tools.json (deleted)"
    return [Candidate(label=label, path=manifest_path, bytes=manifest_path.stat().st_size,
                       data={"kind": "manifest", "forget_backbone": forget_backbone})]


def _adapter_candidates(data_dir: Path) -> List[Candidate]:
    adapters_dir = data_dir / "adapters"
    out = []
    if adapters_dir.is_dir():
        for path in sorted(adapters_dir.glob("*.pt")) + sorted(adapters_dir.glob("*.pt.incoming")):
            out.append(Candidate(label=path.name, path=path, bytes=path.stat().st_size))
    return out


def _staging_candidates(data_dir: Path) -> List[Candidate]:
    staging_dir = data_dir / "staging"
    out = []
    if staging_dir.is_dir():
        for path in sorted(staging_dir.iterdir()):
            out.append(Candidate(label=path.name, path=path, bytes=_dir_size(path)))
    return out


def _jobs_candidates(jobs_dir: Path) -> List[Candidate]:
    jobs_path = jobs_dir / "jobs.json"
    out = []
    if jobs_path.exists():
        out.append(Candidate(label="jobs.json", path=jobs_path, bytes=jobs_path.stat().st_size))
    return out


def _chat_candidates(chat_dir: Path) -> List[Candidate]:
    out = []
    for suffix in ("", "-wal", "-shm"):
        path = chat_dir / f"sessions.sqlite{suffix}"
        if path.exists():
            out.append(Candidate(label=path.name, path=path, bytes=path.stat().st_size))
    return out


def _custom_candidates() -> List[Candidate]:
    custom_root = REPO_ROOT / "adaptrna_custom"
    if not custom_root.is_dir():
        return []

    out = []
    for path in sorted(custom_root.rglob("*")):
        if path.name == "__pycache__":
            continue
        if "__pycache__" in path.relative_to(custom_root).parts:
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(custom_root)
        if rel in CUSTOM_KEEP:
            continue
        # Only files directly under tasks/<name>/ or tools/<name>.py are ours to remove;
        # everything else in the tree is either a keep-file or a stray we shouldn't touch.
        parts = rel.parts
        if parts[0] not in ("tasks", "tools"):
            continue
        out.append(Candidate(label=str(rel), path=path, bytes=path.stat().st_size))

    # Directories become empty once their files are gone; sweep those separately so
    # `_remove` (file-or-dir aware) can drop each `tasks/<name>/` in one shot instead.
    task_dirs = set()
    for path in (custom_root / "tasks").glob("*") if (custom_root / "tasks").is_dir() else []:
        if path.is_dir() and path.name != "__pycache__":
            task_dirs.add(path)

    file_candidates = [c for c in out if Path(c.label).parts[0] == "tools"]
    dir_candidates = [
        Candidate(label=f"tasks/{d.name}/", path=d, bytes=_dir_size(d))
        for d in sorted(task_dirs)
    ]
    return file_candidates + dir_candidates


def _pycache_candidates() -> List[Candidate]:
    custom_root = REPO_ROOT / "adaptrna_custom"
    out = []
    for path in sorted(custom_root.rglob("__pycache__")) if custom_root.is_dir() else []:
        out.append(Candidate(label=str(path.relative_to(REPO_ROOT)), path=path, bytes=_dir_size(path)))
    return out


def _temp_candidates(data_dir: Path, jobs_dir: Path) -> List[Candidate]:
    out = []
    for pattern in (".tools.*.tmp",):
        for path in sorted(data_dir.glob(pattern)):
            out.append(Candidate(label=path.name, path=path, bytes=path.stat().st_size))
    for pattern in (".jobs.*.tmp",):
        for path in sorted(jobs_dir.glob(pattern)):
            out.append(Candidate(label=path.name, path=path, bytes=path.stat().st_size))

    tmp_root = Path(tempfile.gettempdir())
    for pattern in ("adaptrna-sandbox-*", "adaptrna-harness-*"):
        for path in sorted(tmp_root.glob(pattern)):
            out.append(Candidate(label=str(path), path=path, bytes=_dir_size(path)))
    return out


# --------------------------------------------------------------------------- plan / apply

def plan_reset(
    *,
    data_dir: Optional[Path] = None,
    jobs_dir: Optional[Path] = None,
    chat_dir: Optional[Path] = None,
    forget_backbone: bool = False,
    apply: bool = False,
) -> Dict[str, Any]:
    data_dir = manifest_module.resolve_data_dir(data_dir)
    from adaptrna_agentic.jobs.store import resolve_jobs_dir
    jobs_dir = resolve_jobs_dir(jobs_dir)
    if chat_dir is None:
        from adaptrna_agentic.cli.chat import CHAT_DIR_VAR
        import os
        chat_dir = Path(os.environ.get(CHAT_DIR_VAR) or REPO_ROOT / "chat_data").expanduser()
    else:
        chat_dir = Path(chat_dir).expanduser()

    groups = {
        "manifest": _manifest_candidates(data_dir, forget_backbone),
        "adapters": _adapter_candidates(data_dir),
        "staging": _staging_candidates(data_dir),
        "jobs": _jobs_candidates(jobs_dir),
        "chat": _chat_candidates(chat_dir),
        "adaptrna_custom": _custom_candidates(),
        "pycache": _pycache_candidates(),
        "temp": _temp_candidates(data_dir, jobs_dir),
    }

    removed: Dict[str, List[str]] = {}
    if apply:
        # Manifest first: clearing tool entries before deleting their artifact copies
        # means nothing is ever left pointing at a missing file mid-operation.
        for candidate in groups["manifest"]:
            if candidate.data.get("forget_backbone"):
                candidate.path.unlink()
            else:
                man = manifest_module.Manifest.load(data_dir)
                man.tools.clear()
                man.save()
            removed.setdefault("manifest", []).append(candidate.label)

        for kind, candidates in groups.items():
            if kind == "manifest":
                continue
            for candidate in candidates:
                _remove(candidate)
                removed.setdefault(kind, []).append(candidate.label)

    reclaimed = sum(c.bytes for candidates in groups.values() for c in candidates)

    return {
        "applied": apply,
        "data_dir": str(data_dir),
        "jobs_dir": str(jobs_dir),
        "chat_dir": str(chat_dir),
        "groups": {
            kind: [{"label": c.label, "bytes": c.bytes} for c in candidates]
            for kind, candidates in groups.items()
        },
        "removed": removed,
        "reclaimed_bytes": reclaimed,
    }


PRESERVED = [
    "outputs/            -- training results; this is not a git-clean, that data is yours",
    "dataset/            -- downloaded/prepared datasets",
    "weights/            -- local backbone checkpoint copy, if any",
    ".venv/              -- virtualenv",
    ".env                -- holds ANTHROPIC_API_KEY; never touched",
    "~/.cache/rinalmo_pretrained/  -- the downloaded backbone",
    "toolhub_data/tools.json 'backbone' block  -- lm_config/weights/device/dtype "
    "(unless --forget-backbone)",
]


def format_report(report: Dict[str, Any]) -> str:
    verb = "removed" if report["applied"] else "would remove"
    lines = [f"reset: {verb} state under:",
             f"  toolhub  {report['data_dir']}",
             f"  jobs     {report['jobs_dir']}",
             f"  chat     {report['chat_dir']}",
             ""]

    total_items = 0
    for kind, items in report["groups"].items():
        if not items:
            continue
        total_items += len(items)
        lines.append(f"[{kind}]")
        for item in items:
            lines.append(f"  - {item['label']} ({item['bytes'] / 1e6:.2f} MB)")

    lines.append("")
    lines.append(f"{verb} {total_items} item(s), {report['reclaimed_bytes'] / 1e6:.1f} MB total")

    if not report["applied"]:
        if total_items:
            lines.append("\nDry run -- nothing was deleted. Re-run with --yes to apply.")
        else:
            lines.append("\nAlready fresh -- nothing to do.")

    lines.append("\nPreserved (never touched by this script):")
    for item in PRESERVED:
        lines.append(f"  * {item}")

    if report["applied"]:
        lines.append(
            "\nNote: job records are gone but outputs/ was left in place, so old run "
            "directories now have no job pointing at them. Their metrics files are "
            "untouched; `analyze_run <old_job_id>` will simply no longer find a record."
        )
        lines.append(
            "\nVerify with:\n"
            "  python -m adaptrna_agentic.cli.toolhub list     # expect: no tools\n"
            "  python -m adaptrna_agentic.cli.toolhub config   # backbone unchanged\n"
            "  python -m adaptrna_agentic.cli.toolhub doctor"
        )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python agentic/scripts/reset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yes", action="store_true", help="Actually delete (default: dry run)")
    parser.add_argument("--cancel-running", action="store_true",
                        help="Cancel any running training job first, instead of refusing")
    parser.add_argument("--forget-backbone", action="store_true",
                        help="Also delete tools.json's 'backbone' block (lm_config/weights)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="ToolHub state dir (default: $ADAPTRNA_TOOLHUB_DIR or <repo>/toolhub_data)")
    parser.add_argument("--jobs-dir", type=str, default=None,
                        help="Job store dir (default: $ADAPTRNA_JOBS_DIR or <repo>/jobs_data)")
    parser.add_argument("--chat-dir", type=str, default=None,
                        help="Chat/session dir (default: $ADAPTRNA_CHAT_DIR or <repo>/chat_data)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from adaptrna_agentic.jobs.store import JobStore, resolve_jobs_dir

    jobs_dir = resolve_jobs_dir(args.jobs_dir)
    store = JobStore(jobs_dir)
    running = store.running()
    if running:
        if not args.cancel_running:
            ids = ", ".join(r.id for r in running)
            print(f"error: {len(running)} job(s) still running: {ids}", file=sys.stderr)
            print("Pass --cancel-running to stop them first, or wait for them to finish.",
                  file=sys.stderr)
            return 1

        from adaptrna_agentic.jobs.runner import JobRunner
        runner = JobRunner(store)
        for record in running:
            print(f"cancelling {record.id} ...")
            runner.cancel(record.id)

    report = plan_reset(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        jobs_dir=Path(args.jobs_dir) if args.jobs_dir else None,
        chat_dir=Path(args.chat_dir) if args.chat_dir else None,
        forget_backbone=args.forget_backbone,
        apply=args.yes,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
