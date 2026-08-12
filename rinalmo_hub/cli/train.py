"""Train one task, full fine-tuning or LoRA.

    python -m rinalmo_hub.cli.train --task mrl --config configs/tasks/mrl.yaml \
        --use_lora --set optim.lr=3e-4 --set lora.layer_stride=3 \
        --seed 42 --output_dir outputs/mrl_lora

The only thing that differs between the two arms of a comparison is `--use_lora` and the
learning rate; everything else stays in the task YAML so it cannot drift between runs.
"""

from pathlib import Path
import argparse

import torch

from rinalmo_hub.adapter import build_metadata
from rinalmo_hub.cli.common import (
    add_common_arguments,
    build_module,
    build_trainer,
    prepare_run,
    resolve_run_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rinalmo_hub.cli.train",
        description="Train a RiNALMo downstream task (full fine-tuning or LoRA).",
    )
    add_common_arguments(parser)

    parser.add_argument("--adapter_out", type=str, default=None,
                        help="Where to write the trained adapter (LoRA runs). "
                             "Defaults to '<output_dir>/<task>_adapter.pt'")
    parser.add_argument("--save_full_weights", action="store_true", default=False,
                        help="Full-FT runs: write the whole state dict (~2.6 GB for 'giga') "
                             "to '<output_dir>/<task>_full.pt'. Off by default")
    parser.add_argument("--prepare_data", action="store_true", default=False,
                        help="Download and prepare the dataset before training")
    parser.add_argument("--no_test", action="store_true", default=False,
                        help="Skip the test pass after training")
    parser.add_argument("--test_only", action="store_true", default=False,
                        help="Skip training; evaluate the loaded weights on the test set")

    return parser


def _save_artifacts(args, cfg, task, module, trainer) -> None:
    """
    LoRA runs get a ~6 MB adapter; full-FT runs get nothing unless asked.

    A full-FT run's adapter file would carry only the head, and the backbone it was trained
    with would not travel with it -- pairing that head with the pretrained backbone silently
    produces nonsense, so `--save_full_weights` (~2.6 GB) is the honest artifact.
    """
    metadata = build_metadata(
        train_metrics={k: float(v) for k, v in trainer.callback_metrics.items()},
        seed=cfg.get("seed"),
    )

    if module.use_lora:
        adapter_out = args.adapter_out
        if adapter_out is None and args.output_dir:
            adapter_out = Path(args.output_dir) / f"{task}_adapter.pt"

        if adapter_out:
            path = module.save_adapter(adapter_out, metadata=metadata)
            print(f"Saved adapter to '{path}' ({path.stat().st_size / 1e6:.2f} MB)")
        return

    if args.save_full_weights and args.output_dir:
        path = Path(args.output_dir) / f"{task}_full.pt"
        torch.save(module.state_dict(), path)
        print(f"Saved full state dict to '{path}' ({path.stat().st_size / 1e9:.2f} GB)")
    elif args.adapter_out:
        path = module.save_adapter(args.adapter_out, metadata=metadata)
        print(f"Saved head-only export to '{path}'. Note this is a full fine-tuning run: the "
              f"backbone it was trained with is NOT in this file.")
    else:
        print("Full fine-tuning run: no artifact written. Pass --save_full_weights to keep "
              "the trained weights (~2.6 GB for 'giga').")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg, task = resolve_run_config(args)
    prepare_run(cfg, args.output_dir)

    module = build_module(cfg, task, args)
    datamodule = type(module).build_datamodule(cfg)
    trainer = build_trainer(cfg, args.output_dir)

    if not args.test_only:
        trainer.fit(model=module, datamodule=datamodule)

        if trainer.is_global_zero:
            _save_artifacts(args, cfg, task, module, trainer)

    if not args.no_test:
        trainer.test(model=module, datamodule=datamodule)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
