"""Staging: where generated code lives while it is being verified and reviewed.

The staging tree **mirrors the final layout exactly**

    <staging_root>/adaptrna_custom/tasks/<name>/{__init__,task,datamodule}.py + config.yaml

so the module path that gets verified (`adaptrna_custom.tasks.<name>.task`) is the module
path that will be imported after landing. Verifying code at one import path and shipping
it at another is how you get a green report for something that then fails on the first
real use.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import uuid

from adaptrna_agentic.codegen.discovery import (
    CUSTOM_PACKAGE,
    CUSTOM_ROOT,
    TASKS_DIRNAME,
    TOOLS_DIRNAME,
    task_module_path,
)
from adaptrna_agentic.settings import REPO_ROOT
from adaptrna_agentic.toolhub.errors import ToolHubError
from adaptrna_agentic.toolhub.manifest import resolve_data_dir

TASK_FILES = ("task.py", "datamodule.py", "config.yaml")


def staging_root(data_dir=None) -> Path:
    return resolve_data_dir(data_dir) / "staging"


@dataclass
class Stage:
    """One staged artifact: a task package or an external wrapper, awaiting approval."""

    id: str
    kind: str                    # "task" | "tool"
    name: str
    root: Path                   # <staging_root>/<id>  — goes on sys.path for verification
    files: Dict[str, str]        # repo-relative destination -> content

    @property
    def package_dir(self) -> Path:
        if self.kind == "task":
            return self.root / CUSTOM_PACKAGE / TASKS_DIRNAME / self.name
        return self.root / CUSTOM_PACKAGE / TOOLS_DIRNAME

    @property
    def module_path(self) -> str:
        if self.kind == "task":
            return task_module_path(self.name)
        return f"{CUSTOM_PACKAGE}.{TOOLS_DIRNAME}.{self.name}"

    @property
    def config_path(self) -> Optional[Path]:
        candidate = self.package_dir / "config.yaml"
        return candidate if candidate.exists() else None

    def summary(self) -> List[Dict[str, object]]:
        return [
            {"path": destination, "lines": len(content.splitlines())}
            for destination, content in sorted(self.files.items())
        ]

    def diff(self) -> str:
        """What landing would write, as reviewable text."""
        blocks = []
        for destination, content in sorted(self.files.items()):
            existing = REPO_ROOT / destination
            marker = "modified" if existing.exists() else "new file"
            blocks.append(f"--- {destination}  ({marker}, {len(content.splitlines())} lines)\n{content}")

        return "\n\n".join(blocks)


def stage_task(name: str, files: Dict[str, str], data_dir=None) -> Stage:
    """Write generated task files into a fresh staging tree mirroring the final layout.

    `files` maps bare file names ("task.py", "config.yaml", …) to their content.
    """
    missing = [f for f in TASK_FILES if f not in files]
    if missing:
        raise ToolHubError(
            f"Generated task '{name}' is missing {missing}. A task is exactly three "
            f"files: {list(TASK_FILES)}."
        )

    stage_id = f"{name}-{uuid.uuid4().hex[:8]}"
    root = staging_root(data_dir) / stage_id
    package = root / CUSTOM_PACKAGE / TASKS_DIRNAME / name
    package.mkdir(parents=True, exist_ok=True)

    # Package markers so the mirrored import path resolves.
    (root / CUSTOM_PACKAGE / "__init__.py").write_text("")
    (root / CUSTOM_PACKAGE / TASKS_DIRNAME / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")

    destinations: Dict[str, str] = {}
    for filename, content in files.items():
        (package / filename).write_text(content)
        destinations[f"{CUSTOM_PACKAGE}/{TASKS_DIRNAME}/{name}/{filename}"] = content

    return Stage(id=stage_id, kind="task", name=name, root=root, files=destinations)


def stage_tool(name: str, content: str, data_dir=None) -> Stage:
    """Write a generated external wrapper into a staging tree."""
    stage_id = f"{name}-{uuid.uuid4().hex[:8]}"
    root = staging_root(data_dir) / stage_id
    package = root / CUSTOM_PACKAGE / TOOLS_DIRNAME
    package.mkdir(parents=True, exist_ok=True)

    (root / CUSTOM_PACKAGE / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / f"{name}.py").write_text(content)

    return Stage(
        id=stage_id, kind="tool", name=name, root=root,
        files={f"{CUSTOM_PACKAGE}/{TOOLS_DIRNAME}/{name}.py": content},
    )


def load_stage(stage_id: str, data_dir=None) -> Optional[Stage]:
    """Reconstruct a stage from disk.

    Staging directories outlive the process that created them — deliberately: the user is
    invited to open the staged files in an editor before approving, which may well happen
    in a later session.
    """
    root = staging_root(data_dir) / stage_id
    if not root.is_dir():
        return None

    tasks_dir = root / CUSTOM_PACKAGE / TASKS_DIRNAME
    if tasks_dir.is_dir():
        for package in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
            files = {
                f"{CUSTOM_PACKAGE}/{TASKS_DIRNAME}/{package.name}/{f.name}": f.read_text()
                for f in sorted(package.iterdir())
                if f.is_file() and f.name != "__init__.py"
            }
            if files:
                return Stage(id=stage_id, kind="task", name=package.name,
                             root=root, files=files)

    tools_dir = root / CUSTOM_PACKAGE / TOOLS_DIRNAME
    if tools_dir.is_dir():
        for module in sorted(tools_dir.glob("*.py")):
            if module.name == "__init__.py":
                continue
            return Stage(
                id=stage_id, kind="tool", name=module.stem, root=root,
                files={f"{CUSTOM_PACKAGE}/{TOOLS_DIRNAME}/{module.name}": module.read_text()},
            )

    return None


def list_stages(data_dir=None) -> List[Stage]:
    root = staging_root(data_dir)
    if not root.is_dir():
        return []

    stages = [load_stage(entry.name, data_dir) for entry in sorted(root.iterdir()) if entry.is_dir()]
    return [stage for stage in stages if stage is not None]


def land(stage: Stage) -> List[str]:
    """Copy a staged artifact into `<repo>/adaptrna_custom/`. The gated step."""
    written = []
    for destination, content in sorted(stage.files.items()):
        target = REPO_ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        written.append(destination)

    # Package markers in the real tree (they exist already, except for a new task dir).
    if stage.kind == "task":
        init = CUSTOM_ROOT / TASKS_DIRNAME / stage.name / "__init__.py"
        if not init.exists():
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text("")
            written.append(str(init.relative_to(REPO_ROOT)))

    return written


def discard(stage: Stage) -> None:
    shutil.rmtree(stage.root, ignore_errors=True)
