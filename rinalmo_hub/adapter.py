"""Adapter file format v2 -- save, load and inspect.

An adapter file is everything a fine-tuning run learned, minus the shared backbone:

    {
      "format_version": 2,
      "task": "mrl",                 # so the hub can dispatch without a config file
      "lm_config": "giga",           # guard: refuse to load onto a different backbone size
      "lora": {...} | None,          # None => head-only / full-FT export
      "head_config": {...},          # kwargs to rebuild the head with no YAML present
      "state_dict": {...},           # lora_* + head.* + ADAPTER_EXTRA_PREFIXES tensors only
      "extra": {...},                # non-tensor state (sec-struct threshold, ...)
      "metadata": {...},             # created / git_sha / train_metrics / seed
    }

`task` and `head_config` are the additions over the v1 format the source repo used; both are
needed so `RiNALMoHub.register(path)` can rebuild a task from the file alone.

Run `python -m rinalmo_hub.adapter <path>` to print a summary of one.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
import subprocess

import torch

ADAPTER_FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (2,)

REQUIRED_FIELDS = ("format_version", "task", "lm_config", "lora", "head_config", "state_dict")


def git_sha(repo_root: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Short SHA of the working tree, or None outside a git checkout."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None

    return sha or None


def build_metadata(
    train_metrics: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    **extra: Any,
) -> Dict[str, Any]:
    metadata = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "train_metrics": train_metrics or {},
        "seed": seed,
    }
    metadata.update(extra)

    return metadata


def save_adapter(
    path: Union[str, Path],
    *,
    task: str,
    lm_config: str,
    lora: Optional[dict],
    head_config: dict,
    state_dict: Dict[str, torch.Tensor],
    extra: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> Path:
    """Write an adapter file. Tensors are detached and moved to CPU first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": ADAPTER_FORMAT_VERSION,
        "task": task,
        "lm_config": lm_config,
        "lora": lora,
        "head_config": dict(head_config),
        "state_dict": {k: v.detach().cpu().clone() for k, v in state_dict.items()},
        "extra": dict(extra or {}),
        "metadata": dict(metadata or build_metadata()),
    }
    torch.save(payload, path)

    return path


def load_adapter(path: Union[str, Path], lm_config: Optional[str] = None) -> Dict[str, Any]:
    """
    Read and validate an adapter file.

    `lm_config` is the backbone the caller intends to load this onto; passing it turns a
    silent shape mismatch deep inside `load_state_dict` into a clear message.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError(f"'{path}' is not a RiNALMo-Hub adapter file")

    version = payload["format_version"]
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Adapter '{path}' has format version {version}; this build reads "
            f"{SUPPORTED_FORMAT_VERSIONS}"
        )

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"Adapter '{path}' is missing required field(s): {missing}")

    if lm_config is not None and payload["lm_config"] != lm_config:
        raise ValueError(
            f"Adapter '{path}' was trained on the '{payload['lm_config']}' backbone, but "
            f"'{lm_config}' was requested. Backbone sizes are not interchangeable."
        )

    payload.setdefault("extra", {})
    payload.setdefault("metadata", {})

    return payload


def _tensor_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in state_dict.values())


def describe_adapter(path: Union[str, Path]) -> str:
    """Human-readable summary of an adapter file."""
    path = Path(path)
    payload = load_adapter(path)

    state_dict = payload["state_dict"]
    lora_keys = [k for k in state_dict if "lora_" in k]
    head_keys = [k for k in state_dict if k.startswith("head.")]
    other_keys = [k for k in state_dict if k not in set(lora_keys) | set(head_keys)]

    lines = [
        f"{path}  ({path.stat().st_size / 1e6:.2f} MB on disk)",
        f"  format_version : {payload['format_version']}",
        f"  task           : {payload['task']}",
        f"  lm_config      : {payload['lm_config']}",
        f"  head_config    : {payload['head_config']}",
        f"  lora           : {payload['lora']}",
        f"  tensors        : {len(state_dict)} "
        f"({len(lora_keys)} lora, {len(head_keys)} head, {len(other_keys)} other)"
        f" / {_tensor_bytes(state_dict) / 1e6:.2f} MB",
        f"  parameters     : {sum(t.numel() for t in state_dict.values()):,}",
    ]

    if other_keys:
        lines.append(f"  other tensors  : {sorted(other_keys)}")
    if payload["extra"]:
        lines.append(f"  extra          : {payload['extra']}")

    lines.append("  metadata       :")
    for key, value in payload["metadata"].items():
        lines.append(f"    {key}: {value}")

    return "\n".join(lines)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a RiNALMo-Hub adapter file.")
    parser.add_argument("path", nargs="+", help="Adapter '.pt' file(s) to describe")
    args = parser.parse_args(argv)

    for path in args.path:
        print(describe_adapter(path))
        print()


if __name__ == "__main__":
    main()
