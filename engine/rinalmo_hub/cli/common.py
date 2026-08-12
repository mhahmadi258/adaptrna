"""Shared plumbing for the three CLI entry points.

Model construction order is load-bearing and identical everywhere:

    build module -> load pretrained backbone -> inject LoRA -> load adapter

Injection moves each target's weight to `<target>.base_layer.weight`, so the pretrained
load has to happen first; and an adapter file's key names are the post-injection ones, so
it has to happen last.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import argparse

import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

import rinalmo_hub.tasks  # noqa: F401  -- registers every shipped task
from rinalmo.utils.finetune_callback import GradualUnfreezing
from rinalmo_hub.adapter import load_adapter
from rinalmo_hub.config import Config, resolve_config, save_config
from rinalmo_hub.registry import available_tasks, get_task


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--task", type=str, default=None,
                        help=f"Task name. Available: {available_tasks()}")
    parser.add_argument("--config", type=str, default=None,
                        help="Task YAML, layered on top of configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Dotted config override, repeatable (e.g. --set optim.lr=3e-4)")
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Fine-tune with LoRA adapters instead of updating the whole backbone")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Path to the pretrained RiNALMo backbone ('giga-v1.pt')")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Adapter '.pt' to load. Supplies the task and LoRA geometry, so "
                             "--use_lora and the lora.* config keys are taken from the file")
    parser.add_argument("--init_params", type=str, default=None,
                        help="Full state dict to start from (a full-FT export)")
    parser.add_argument("--lm_config", type=str, default=None,
                        help="Backbone size: nano | micro | mega | giga")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where logs, metrics.csv, the resolved config and adapters go")

    return parser


def resolve_run_config(args) -> Tuple[Config, str]:
    """Layer the configs, fold the CLI flags in, and work out which task this is."""
    adapter_payload = load_adapter(args.adapter) if args.adapter else None

    config_path = args.config
    if config_path is None and args.task:
        default_path = Path(__file__).resolve().parent.parent.parent / "configs" / "tasks" / f"{args.task}.yaml"
        config_path = default_path if default_path.exists() else None

    cfg = resolve_config(config_path, args.overrides)

    if args.task:
        cfg["task"] = args.task
    if args.lm_config:
        cfg["lm_config"] = args.lm_config
    if args.pretrained_weights:
        cfg["pretrained_weights"] = args.pretrained_weights
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.use_lora:
        cfg["use_lora"] = True
    if getattr(args, "prepare_data", False):
        cfg["data"]["prepare"] = True

    if adapter_payload is not None:
        # The file is the authority on everything needed to rebuild the trained model.
        cfg["task"] = adapter_payload["task"]
        cfg["lm_config"] = adapter_payload["lm_config"]
        cfg["head"] = adapter_payload["head_config"]
        cfg["lora"] = adapter_payload["lora"]
        cfg["use_lora"] = adapter_payload["lora"] is not None

    task = cfg.get("task")
    if not task:
        raise SystemExit(f"No task given. Pass --task (one of {available_tasks()}) or a --config that sets it.")
    if task not in available_tasks():
        raise SystemExit(f"Unknown task '{task}'. Available: {available_tasks()}")

    return cfg, task


def build_module(cfg: Config, task: str, args, verbose: bool = True):
    """Construct the task module and get weights into it in the one correct order."""
    task_cls = get_task(task)
    lora_cfg = cfg.get("lora") if cfg.get("use_lora") else None

    module = task_cls(
        lm_config=cfg["lm_config"],
        head_config=cfg.get("head") or {},
        lora=lora_cfg,
        optim=cfg.get("optim") or {},
        finetune=cfg.get("finetune") or {},
        task_config=cfg.get("task_config") or {},
    )

    weights = cfg.get("pretrained_weights")
    if weights:
        if not Path(weights).exists():
            raise SystemExit(
                f"Pretrained backbone '{weights}' not found. Download `giga-v1.pt` (see the "
                f"README's Setup section) or pass --set pretrained_weights=null to start from "
                f"a random backbone."
            )
        module.load_pretrained_backbone(weights)
        if verbose:
            print(f"Loaded pretrained backbone from '{weights}'")

    if module.use_lora:
        module.apply_lora(verbose=verbose)

    if args.adapter:
        module.load_adapter(args.adapter)

    if args.init_params:
        module.load_state_dict(torch.load(args.init_params, map_location="cpu", weights_only=False))
        if verbose:
            print(f"Loaded full state dict from '{args.init_params}'")

    return module


def _multi_device(devices) -> bool:
    """Resolve `devices` -- including 'auto' -- to "is this a multi-GPU run?"."""
    if devices in ("auto", None):
        return torch.cuda.device_count() > 1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    if isinstance(devices, str):
        if "," in devices:
            return True
        devices = int(devices)

    return int(devices) > 1 or int(devices) == -1


def build_trainer(cfg: Config, output_dir: Optional[str], extra_callbacks: Optional[List] = None) -> pl.Trainer:
    trainer_cfg = dict(cfg.get("trainer") or {})
    callbacks = list(extra_callbacks or [])
    loggers = []

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Metrics land on disk as the run goes, which is what makes a long job watchable
        # with `tail -f` and survives a crash.
        loggers.append(CSVLogger(save_dir=output_dir, name="metrics"))

    # Lightning's default RichProgressBar detects a non-TTY and buffers, so a redirected log
    # stays empty for the whole run -- an 8-hour job looked frozen at 3855 bytes. tqdm
    # flushes on every refresh, so the bar survives redirection.
    callbacks.append(TQDMProgressBar(refresh_rate=int(trainer_cfg.pop("progress_refresh_rate", 50))))
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Off by default: a full-FT checkpoint of the giga backbone is 7.8 GB per epoch. Note
    # Lightning installs a *default* `ModelCheckpoint` when `enable_checkpointing` is left
    # alone, so opting out has to be explicit -- passing no checkpoint callback is not enough.
    checkpoint_every_epoch = trainer_cfg.pop("checkpoint_every_epoch", False)
    if checkpoint_every_epoch:
        callbacks.append(ModelCheckpoint(dirpath=output_dir, filename="{epoch}-{step}",
                                         every_n_epochs=1, save_top_k=-1))

    finetune_cfg = cfg.get("finetune") or {}
    if finetune_cfg.get("schedule"):
        if cfg.get("use_lora"):
            raise SystemExit(
                "finetune.schedule and --use_lora are mutually exclusive: a gradual "
                "unfreezing schedule would unfreeze the backbone LoRA is meant to keep frozen."
            )
        callbacks.append(GradualUnfreezing(
            unfreeze_schedule_path=finetune_cfg["schedule"],
            initial_denom_lr=float(finetune_cfg.get("initial_denom_lr", 1.0)),
        ))

    devices = trainer_cfg.pop("devices", "auto")
    # `RiNALMo.forward` always runs `lm_mask_head`, whose output never reaches the loss, so
    # DDP has to be told to expect parameters that get no gradient. Note that 'auto' has to
    # be resolved here too, or a multi-GPU box crashes on default arguments.
    strategy = DDPStrategy(find_unused_parameters=True) if _multi_device(devices) else "auto"

    return pl.Trainer(
        accelerator=trainer_cfg.pop("accelerator", "auto"),
        devices=devices,
        max_epochs=trainer_cfg.pop("max_epochs", -1),
        max_steps=trainer_cfg.pop("max_steps", -1),
        precision=trainer_cfg.pop("precision", "bf16-mixed"),
        gradient_clip_val=trainer_cfg.pop("gradient_clip_val", None),
        log_every_n_steps=trainer_cfg.pop("log_every_n_steps", 50),
        default_root_dir=output_dir,
        enable_checkpointing=checkpoint_every_epoch,
        strategy=strategy,
        logger=loggers or False,
        callbacks=callbacks,
        **trainer_cfg,
    )


def prepare_run(cfg: Config, output_dir: Optional[str], filename: str = "resolved_config.yaml") -> None:
    if cfg.get("seed") is not None:
        pl.seed_everything(int(cfg["seed"]), workers=True)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_config(cfg, Path(output_dir) / filename)
        print(f"Resolved config written to '{Path(output_dir) / filename}'")
