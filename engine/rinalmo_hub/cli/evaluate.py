"""Evaluate a backbone + adapter on a task's test set.

    python -m rinalmo_hub.cli.evaluate --adapter outputs/mrl_lora/mrl_adapter.pt \
        --pretrained_weights weights/giga-v1.pt --config configs/tasks/mrl.yaml \
        --output_dir outputs/mrl_lora_eval

The adapter file carries the task, the backbone size, the LoRA geometry and the head config,
so `--task`, `--use_lora` and the `lora.*` keys are all read from it. The `--config` is still
needed for the data paths.
"""

import argparse
import json

from rinalmo_hub.cli.common import (
    add_common_arguments,
    build_module,
    build_trainer,
    prepare_run,
    resolve_run_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rinalmo_hub.cli.evaluate",
        description="Evaluate a trained adapter (or full checkpoint) on a task's test set.",
    )
    add_common_arguments(parser)

    parser.add_argument("--split", type=str, default="test", choices=("test", "validate"),
                        help="Which split to run")
    parser.add_argument("--prepare_data", action="store_true", default=False)
    parser.add_argument("--metrics_out", type=str, default=None,
                        help="Write the resulting metrics to this JSON file")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.adapter and not args.init_params:
        raise SystemExit("Nothing to evaluate: pass --adapter or --init_params.")

    cfg, task = resolve_run_config(args)
    prepare_run(cfg, args.output_dir, filename="resolved_config_eval.yaml")

    module = build_module(cfg, task, args)
    datamodule = type(module).build_datamodule(cfg)
    trainer = build_trainer(cfg, args.output_dir)

    runner = trainer.test if args.split == "test" else trainer.validate
    results = runner(model=module, datamodule=datamodule)

    metrics = results[0] if results else {}
    print(json.dumps(metrics, indent=2, default=float))

    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"Metrics written to '{args.metrics_out}'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
