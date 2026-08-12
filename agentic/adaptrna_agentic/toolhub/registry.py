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

# Re-exported here for backward compatibility; the class lives in errors.py so the
# external-tool contract module can share it without an import cycle.
from adaptrna_agentic.toolhub.errors import ToolHubError  # noqa: E402, F401


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

        # Copy to a temp name first and only move it into place once the manifest write
        # has succeeded: a failure between the two would otherwise leave an orphaned
        # artifact (copy-first) or an entry pointing at nothing (manifest-first).
        dest = staged_copy = None
        if link:
            artifact = str(source)
        else:
            dest = self.data_dir / "adapters" / f"{name}.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            staged_copy = dest.with_suffix(".pt.incoming")
            shutil.copy2(source, staged_copy)
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
        try:
            self.manifest.save()
        except BaseException:
            self.manifest.tools.pop(name, None)
            if staged_copy is not None:
                staged_copy.unlink(missing_ok=True)
            raise

        if staged_copy is not None:
            staged_copy.replace(dest)

        return entry

    def register_external(
        self,
        module_path: str,
        *,
        only: Optional[Sequence[str]] = None,
    ) -> List[ToolEntry]:
        """Register every function of an external wrapper module (or the `only` subset)
        as one tool entry each, named `<family>_<function>`.

        The wrapped package must already be importable — the CLI owns the approval-gated
        install; this method only refuses with the exact command when it is missing.
        """
        from adaptrna_agentic.toolhub.external import contract

        spec, _module = contract.load_spec(module_path)

        if not contract.is_available(spec.package):
            command = " ".join(contract.install_command(spec.package))
            raise ToolHubError(
                f"Package '{spec.package.pip}' (import '{spec.package.import_name}') is "
                f"not installed. Install it with `{command}`, or rerun "
                f"`toolhub register-external {module_path} --yes` for the gated install."
            )

        available = {fn.name: fn for fn in spec.functions}
        if only is not None:
            unknown = sorted(set(only) - set(available))
            if unknown:
                raise ToolHubError(
                    f"Unknown function(s) {unknown} in '{module_path}'. "
                    f"Available: {sorted(available)}"
                )
            selected = [available[name] for name in only]
        else:
            selected = list(spec.functions)

        # Refuse the whole batch before writing anything.
        for function in selected:
            tool_name = f"{spec.name}_{function.name}"
            if tool_name in self.manifest.tools:
                raise ToolHubError(
                    f"'{tool_name}' is already registered. Remove it first "
                    f"(`toolhub remove {tool_name}`)."
                )

        version = contract.installed_version(spec.package)
        registered_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        entries = []
        for function in selected:
            entry = ToolEntry(
                name=f"{spec.name}_{function.name}",
                type="external",
                state="active",
                description=function.description,
                serving={},
                # Golden cases are copied in, so the manifest entry is self-contained
                # even if the wrapper module's SPEC changes later.
                test={"golden": contract.golden_as_dicts(function)},
                external={
                    "module": module_path,
                    "function": function.name,
                    "package": {
                        "pip": spec.package.pip,
                        "import_name": spec.package.import_name,
                        "installed_version": version,
                    },
                },
                provenance={
                    "source": module_path,
                    "registered_at": registered_at,
                    "family": spec.name,
                    "family_description": spec.description,
                },
            )
            self.manifest.tools[entry.name] = entry
            entries.append(entry)

        self.manifest.save()
        return entries

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

        # Only delete copies the registry owns; a --link'ed source is never touched, and
        # external tools have no artifact at all.
        if (
            not keep_artifact
            and artifact is not None
            and self._owns(artifact)
            and artifact.exists()
        ):
            artifact.unlink()

    def _owns(self, path: Path) -> bool:
        try:
            path.resolve().relative_to((self.data_dir / "adapters").resolve())
            return True
        except ValueError:
            return False

    def verify(self) -> Dict[str, Any]:
        """Manifest ↔ disk consistency, in both directions. Read-only; feeds `doctor`."""
        missing, orphans = [], []

        for entry in self.list():
            path = entry.artifact_path()
            if entry.type == "adapter" and (path is None or not path.exists()):
                missing.append({"tool": entry.name, "artifact": str(path)})

        referenced = {
            str(e.artifact_path().resolve())
            for e in self.list()
            if e.artifact_path() is not None and e.artifact_path().exists()
        }
        adapters_dir = self.data_dir / "adapters"
        if adapters_dir.is_dir():
            for candidate in sorted(adapters_dir.glob("*.pt")):
                if str(candidate.resolve()) not in referenced:
                    orphans.append({
                        "artifact": str(candidate),
                        "size_bytes": candidate.stat().st_size,
                    })

        return {"missing_artifacts": missing, "orphan_artifacts": orphans}

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
