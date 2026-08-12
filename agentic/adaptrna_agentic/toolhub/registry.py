"""Tool lifecycle over the manifest: register / activate / deactivate / remove / list.

The engine is imported lazily and only to *validate adapter files* at registration —
nothing in this module ever loads a backbone (that is the runtime's job, and it is lazy
by decision; see plans/PHASE_2_TOOLHUB_CORE.md §2).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import json
import shutil

from adaptrna_agentic.settings import REPO_ROOT
from adaptrna_agentic.toolhub.manifest import (
    BackboneConfig,
    Manifest,
    ToolEntry,
    resolve_data_dir,
)

#: Tasks whose head is pad-sensitive: predictions depend on batch composition (padding
#: reaches the head through biases + InstanceNorm), so the hub serves them one sequence
#: at a time. This is MASTER_PLAN §7's per-tool serving policy.
PAD_SENSITIVE_TASKS = {"mrl"}

DEFAULT_TEST_SEQUENCES = [
    "GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC",
    "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC",
]

_NULL_STRINGS = ("null", "none", "")


class ToolHubError(RuntimeError):
    """Anything the ToolHub refuses to do, with the reason and the fix in the message."""


def _engine_load_adapter(path: Path) -> Dict[str, Any]:
    try:
        from rinalmo_hub.adapter import load_adapter
    except ImportError as exc:
        raise ToolHubError(
            "The engine package is not installed in this environment. "
            "Run `pip install -e ./engine` from the repo root."
        ) from exc

    return load_adapter(path)


def _jsonable_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(metadata, default=str))


class Registry:
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = resolve_data_dir(data_dir)
        self.manifest = Manifest.load(self.data_dir)

    # ------------------------------------------------------------------ queries

    def list(self) -> List[ToolEntry]:
        return sorted(self.manifest.tools.values(), key=lambda entry: entry.name)

    def get(self, name: str) -> ToolEntry:
        if name not in self.manifest.tools:
            raise KeyError(
                f"No tool named '{name}'. Known tools: {sorted(self.manifest.tools)}"
            )
        return self.manifest.tools[name]

    # ------------------------------------------------------------------ lifecycle

    def register(
        self,
        adapter_path: Union[str, Path],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        batch_size: Optional[int] = None,
        test_sequences: Optional[Sequence[str]] = None,
        link: bool = False,
    ) -> ToolEntry:
        source = Path(adapter_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Adapter file '{adapter_path}' not found")

        payload = _engine_load_adapter(source)
        task = payload["task"]
        name = name or task

        if name in self.manifest.tools:
            existing = self.manifest.tools[name].provenance.get("source", "?")
            raise ToolHubError(
                f"'{name}' is already registered (from '{existing}'). "
                f"Remove it first, or register under a different --name."
            )

        # LoRA-only (MASTER_PLAN §3.6). The engine's hub enforces this too; checking here
        # fails before anything is copied, with the rationale up front.
        metadata = dict(payload.get("metadata") or {})
        if payload["lora"] is None and metadata.get("arm") == "full_ft":
            raise ToolHubError(
                f"'{adapter_path}' is a full fine-tuning export: only its head travelled "
                f"with the file, so serving it would silently pair a fine-tuned head with "
                f"the pretrained backbone. Adapter tools are LoRA-only; evaluate full-FT "
                f"exports with the engine's `rinalmo_hub.cli.evaluate --init_params`."
            )

        backbone = self.manifest.backbone
        if payload["lm_config"] != backbone.lm_config:
            raise ToolHubError(
                f"Adapter '{adapter_path}' was trained on the '{payload['lm_config']}' "
                f"backbone, but this ToolHub serves '{backbone.lm_config}'. Backbone sizes "
                f"are not interchangeable. (An empty hub can be switched with "
                f"`toolhub config --lm-config {payload['lm_config']}`.)"
            )

        if link:
            artifact = str(source)
        else:
            dest = self.data_dir / "adapters" / f"{name}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            try:
                artifact = str(dest.relative_to(REPO_ROOT))
            except ValueError:
                artifact = str(dest)

        serving_batch = batch_size
        pad_note = ""
        if serving_batch is None and task in PAD_SENSITIVE_TASKS:
            serving_batch = 1
            pad_note = (
                " Served one sequence at a time: the task head is pad-sensitive, so batch "
                "composition would change predictions."
            )

        entry = ToolEntry(
            name=name,
            type="adapter",
            state="active",
            description=(description or f"{task} adapter (arm: {metadata.get('arm', '?')})")
            + pad_note,
            task=task,
            lm_config=payload["lm_config"],
            artifact=artifact,
            serving={"batch_size": serving_batch},
            test={"sequences": list(test_sequences or DEFAULT_TEST_SEQUENCES), "expected": None},
            provenance={
                "source": str(source),
                "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "adapter_metadata": _jsonable_metadata(metadata),
            },
        )

        self.manifest.tools[name] = entry
        self.manifest.save()
        return entry

    def activate(self, name: str) -> ToolEntry:
        return self._set_state(name, "active")

    def deactivate(self, name: str) -> ToolEntry:
        return self._set_state(name, "disabled")

    def _set_state(self, name: str, state: str) -> ToolEntry:
        entry = self.get(name)
        entry.state = state
        self.manifest.save()
        return entry

    def remove(self, name: str, *, keep_artifact: bool = False) -> None:
        entry = self.get(name)
        artifact = entry.artifact_path()

        del self.manifest.tools[name]
        self.manifest.save()

        # Only delete copies the registry owns; a --link'ed source is never touched.
        if not keep_artifact and self._owns(artifact) and artifact.exists():
            artifact.unlink()

    def _owns(self, path: Path) -> bool:
        try:
            path.resolve().relative_to((self.data_dir / "adapters").resolve())
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------ backbone config

    def configure_backbone(
        self,
        *,
        weights: Optional[str] = None,
        lm_config: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> BackboneConfig:
        """Update the backbone section. `weights="null"` clears the weights path
        (a random backbone — only sensible for tests)."""
        backbone = self.manifest.backbone

        if lm_config and lm_config != backbone.lm_config and self.manifest.tools:
            raise ToolHubError(
                f"Cannot change lm_config to '{lm_config}': {len(self.manifest.tools)} "
                f"tool(s) are registered for '{backbone.lm_config}'. Remove them first."
            )

        if weights is not None:
            backbone.weights = None if weights.lower() in _NULL_STRINGS else weights
        if lm_config:
            backbone.lm_config = lm_config
        if device:
            backbone.device = device
        if dtype:
            backbone.dtype = dtype

        self.manifest.save()
        return backbone
