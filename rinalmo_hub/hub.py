"""`RiNALMoHub` -- several task adapters resident in one loaded backbone.

    hub = RiNALMoHub(backbone_weights="weights/giga-v1.pt", lm_config="giga", device="cuda")
    hub.register("adapters/splice_site_donor.pt")   # task name is read from the file
    hub.register("adapters/mrl.pt")
    hub.register("adapters/sec_struct.pt")

    hub.available()                        # ["mrl", "sec_struct", "splice_site"]
    hub.predict("mrl", ["GGCAUUAC...", ...])

One `RiNALMo` instance holds every adapter; peft keeps a per-adapter dict inside each
`lora.Linear` and its forward skips any active adapter a layer does not hold, so adapters
with *different* geometries coexist happily and switching is a dictionary lookup rather than
a 2.6 GB reload.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch

from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.model import RiNALMo

import rinalmo_hub.tasks  # noqa: F401  -- registers every shipped task
from rinalmo_hub.adapter import load_adapter
from rinalmo_hub.lora import (
    LoRASpec,
    disable_adapters,
    inject_lora,
    rename_adapter_in_state_dict,
    set_active_adapter,
)
from rinalmo_hub.registry import get_task


class RiNALMoHub:
    def __init__(
        self,
        backbone_weights: Optional[Union[str, Path]] = None,
        lm_config: str = "giga",
        device: Union[str, torch.device] = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        self.lm_config = lm_config
        self.device = torch.device(device)
        self.dtype = dtype

        self.config = model_config(lm_config)
        self.alphabet = Alphabet(**self.config["alphabet"])

        self.backbone = RiNALMo(self.config)
        if backbone_weights is not None:
            self.backbone.load_state_dict(
                torch.load(backbone_weights, map_location="cpu", weights_only=False)
            )

        self.backbone.to(device=self.device, dtype=self.dtype)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self._modules: Dict[str, torch.nn.Module] = {}
        self._sources: Dict[str, str] = {}

    # ------------------------------------------------------------------ registration

    def register(self, path: Union[str, Path], name: Optional[str] = None) -> str:
        """
        Load an adapter file and make its task available under `name` (default: the task name).

        Returns the name it was registered under.
        """
        payload = load_adapter(path, lm_config=self.lm_config)
        task = payload["task"]
        name = name or task

        if name in self._modules:
            raise ValueError(f"'{name}' is already registered (from '{self._sources[name]}')")

        arm = payload["metadata"].get("arm")
        if payload["lora"] is None and arm == "full_ft":
            raise ValueError(
                f"'{path}' is a full fine-tuning export: only its head travelled with the "
                f"file, and the backbone it was trained with is not this one. Serving it "
                f"here would silently pair a fine-tuned head with the pretrained backbone. "
                f"Load it with `rinalmo_hub.cli.evaluate --init_params` instead."
            )

        task_cls = get_task(task)
        module = task_cls(
            lm_config=self.lm_config,
            head_config=payload["head_config"],
            lora=payload["lora"],
            attach_backbone=False,
        )
        # One shared backbone for every task: assigning it here is the whole point of
        # `attach_backbone=False`.
        module.backbone = self.backbone
        module.lora_adapter_name = name

        state_dict = payload["state_dict"]
        if payload["lora"] is not None:
            inject_lora(self.backbone, LoRASpec.from_dict(payload["lora"]), adapter_name=name, verbose=False)
            state_dict = rename_adapter_in_state_dict(state_dict, name)

        module.load_adapter_state_dict(state_dict, source=str(path))
        module.load_adapter_extra(payload["extra"])

        module.to(device=self.device, dtype=self.dtype)
        module.eval()
        module.requires_grad_(False)

        self._modules[name] = module
        self._sources[name] = str(path)

        return name

    def available(self) -> List[str]:
        return sorted(self._modules)

    def module(self, task: str):
        """The `BaseDownstreamModule` serving `task`, for anything `predict` does not cover."""
        if task not in self._modules:
            raise KeyError(f"'{task}' is not registered. Available: {self.available()}")

        return self._modules[task]

    def source(self, task: str) -> str:
        return self._sources[task]

    # ------------------------------------------------------------------ inference

    def activate(self, task: str) -> None:
        """Make `task`'s adapter the active one on every layer that carries it."""
        module = self.module(task)
        if module.use_lora:
            set_active_adapter(self.backbone, module.lora_adapter_name)

    @contextmanager
    def no_adapter(self):
        """Run the bare pretrained backbone, for base-model baselines."""
        with disable_adapters(self.backbone):
            yield self

    def tokenize(self, sequences: Sequence[str]) -> torch.Tensor:
        return torch.tensor(self.alphabet.batch_tokenize(list(sequences)), dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict(self, task: str, sequences, batch_size: Optional[int] = None):
        """
        Run `task` over `sequences` and return its native output type.

        Splice site returns per-sequence probabilities, MRL returns mean ribosome load on
        the original scale, secondary structure returns one base-pairing matrix per sequence.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
        sequences = list(sequences)

        module = self.module(task)
        self.activate(task)

        batch_size = batch_size or type(module).DEFAULT_PREDICT_BATCH_SIZE
        outputs = []

        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start:start + batch_size]
            tokens = self.tokenize(chunk)
            raw = module(tokens)
            outputs.append(module.postprocess_predictions(raw, tokens, chunk))

        return _concat(outputs)

    @torch.no_grad()
    def embed(self, sequences, batch_size: int = 8) -> torch.Tensor:
        """Backbone representations with no adapter active -- the base model's view."""
        if isinstance(sequences, str):
            sequences = [sequences]
        sequences = list(sequences)

        chunks = []
        with self.no_adapter():
            for start in range(0, len(sequences), batch_size):
                tokens = self.tokenize(sequences[start:start + batch_size])
                chunks.append(self.backbone(tokens)["representation"])

        return torch.cat(chunks, dim=0)

    def __repr__(self):
        registered = ", ".join(f"{k} <- {Path(v).name}" for k, v in sorted(self._sources.items()))
        return f"RiNALMoHub(lm_config={self.lm_config!r}, device={str(self.device)!r}, [{registered}])"


def _concat(chunks):
    if not chunks:
        return []
    if isinstance(chunks[0], torch.Tensor):
        return torch.cat(chunks, dim=0)

    flattened = []
    for chunk in chunks:
        flattened.extend(chunk)

    return flattened
