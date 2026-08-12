"""The ToolHub manifest: registry state on disk (`toolhub_data/tools.json`).

Pure data + JSON I/O — this module never imports the engine, and imports torch only
inside the `auto` device/dtype resolution helpers. Decisions recorded in
plans/PHASE_2_TOOLHUB_CORE.md §2: JSON store with atomic writes; paths stored
repo-root-relative (or absolute); `toolhub_data/` resolved from the explicit argument,
then `ADAPTRNA_TOOLHUB_DIR`, then `<repo>/toolhub_data`.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import contextlib
import json
import os
import tempfile

from adaptrna_agentic.settings import REPO_ROOT

FORMAT_VERSION = 1

DATA_DIR_VAR = "ADAPTRNA_TOOLHUB_DIR"

TOOL_STATES = ("active", "disabled")


def resolve_data_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser()

    env = os.environ.get(DATA_DIR_VAR)
    if env:
        return Path(env).expanduser()

    return REPO_ROOT / "toolhub_data"


def resolve_path(path_str: Union[str, Path]) -> Path:
    """Stored paths are `~`-expandable and, when relative, repo-root-relative."""
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class BackboneConfig:
    """One backbone per hub: every adapter tool must match `lm_config`."""

    lm_config: str = "giga"
    weights: Optional[str] = "weights/giga-v1.pt"
    device: str = "auto"   # auto -> cuda if available, else cpu
    dtype: str = "auto"    # auto -> the model default (fp32) -- see resolved_dtype()

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def resolved_dtype(self):
        """A torch dtype, or None to keep the model's default (fp32).

        `auto` deliberately resolves to fp32 even on CUDA: casting the whole engine to
        bf16 for *non-autocast* inference trips a dtype promotion in the engine's
        TokenDropout (an fp32 scalar promotes the activations to fp32, which then hit
        bf16 layer-norm weights) — discovered 2026-08-12, recorded in MASTER_PLAN §7.
        Half-precision serving needs an engine fix or an autocast wrapper; until then,
        an explicit `dtype: bfloat16` is user-opt-in and will fail on that engine path.
        """
        import torch

        if self.dtype == "auto":
            return None

        dtype = getattr(torch, self.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown dtype '{self.dtype}' in the ToolHub backbone config")
        return dtype


@dataclass
class ToolEntry:
    name: str
    type: str                       # "adapter" | "external"
    state: str                      # active | disabled
    description: str
    # Adapter-only fields (None for external tools) — a Phase 3 additive change; v1
    # manifests written before it load unchanged (missing keys take these defaults).
    task: Optional[str] = None
    lm_config: Optional[str] = None
    artifact: Optional[str] = None
    serving: Dict[str, Any] = field(default_factory=lambda: {"batch_size": None})
    test: Dict[str, Any] = field(default_factory=lambda: {"sequences": [], "expected": None})
    provenance: Dict[str, Any] = field(default_factory=dict)
    #: External-only: {"module": ..., "function": ..., "package": {pip, import_name,
    #: installed_version}}.
    external: Optional[Dict[str, Any]] = None

    @property
    def active(self) -> bool:
        return self.state == "active"

    def artifact_path(self) -> Optional[Path]:
        return resolve_path(self.artifact) if self.artifact else None


@dataclass
class Manifest:
    data_dir: Path
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    tools: Dict[str, ToolEntry] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.data_dir / "tools.json"

    @classmethod
    def load(cls, data_dir: Optional[Union[str, Path]] = None) -> "Manifest":
        data_dir = resolve_data_dir(data_dir)
        path = data_dir / "tools.json"

        if not path.exists():
            return cls(data_dir=data_dir)

        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"ToolHub manifest '{path}' is not valid JSON: {exc}") from exc

        if not isinstance(payload, dict) or "format_version" not in payload:
            raise ValueError(f"'{path}' is not a ToolHub manifest")

        version = payload["format_version"]
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Manifest '{path}' has format version {version}; this build reads {FORMAT_VERSION}"
            )

        return cls(
            data_dir=data_dir,
            backbone=BackboneConfig(**payload.get("backbone", {})),
            # The tool name is the dict key on disk; inject it back into the entry.
            tools={
                name: ToolEntry(name=name, **entry)
                for name, entry in payload.get("tools", {}).items()
            },
        )

    def save(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": FORMAT_VERSION,
            "backbone": asdict(self.backbone),
            "tools": {
                name: {k: v for k, v in asdict(entry).items() if k != "name"}
                for name, entry in sorted(self.tools.items())
            },
        }

        # Atomic: write a sibling temp file, then replace. A crash mid-write can never
        # leave a half-written manifest behind.
        fd, tmp_name = tempfile.mkstemp(dir=self.data_dir, prefix=".tools.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.replace(tmp_name, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise

        return self.path
