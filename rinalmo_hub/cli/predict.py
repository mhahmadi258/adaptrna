"""Predict from raw sequences with a backbone + one or more adapters.

    # one task
    python -m rinalmo_hub.cli.predict --pretrained_weights weights/giga-v1.pt \
        --adapter outputs/mrl_lora/mrl_adapter.pt \
        --sequences GGCAUUACGGCUUAA AUGCAUGCAUGCAUG

    # several tasks in one loaded backbone, sequences from a file
    python -m rinalmo_hub.cli.predict --pretrained_weights weights/giga-v1.pt \
        --adapter adapters/mrl.pt --adapter adapters/splice_site_donor.pt \
        --input sequences.txt --output predictions.json --device cuda
"""

from pathlib import Path
import argparse
import json

import torch

from rinalmo_hub.hub import RiNALMoHub


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rinalmo_hub.cli.predict",
        description="Run RNA sequences through one or more trained adapters.",
    )
    parser.add_argument("--adapter", action="append", default=[], required=True,
                        help="Adapter '.pt' file, repeatable (one per task)")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="Path to the pretrained RiNALMo backbone ('giga-v1.pt')")
    parser.add_argument("--lm_config", type=str, default="giga")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default=None, choices=("float32", "bfloat16", "float16"),
                        help="Backbone dtype. FlashAttention needs bfloat16 or float16 on CUDA")

    parser.add_argument("--sequences", nargs="+", default=None, help="RNA sequences, inline")
    parser.add_argument("--input", type=str, default=None,
                        help="File with one sequence per line (lines starting with '>' or '#' are skipped)")
    parser.add_argument("--task", action="append", default=None,
                        help="Only run these tasks. Default: every registered adapter")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Write predictions to this JSON file")

    return parser


def read_sequences(args) -> list:
    sequences = list(args.sequences or [])

    if args.input:
        for line in Path(args.input).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith((">", "#")):
                sequences.append(line)

    if not sequences:
        raise SystemExit("No sequences given: pass --sequences or --input.")

    return sequences


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()

    return value


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sequences = read_sequences(args)

    dtype = getattr(torch, args.dtype) if args.dtype else None
    hub = RiNALMoHub(
        backbone_weights=args.pretrained_weights,
        lm_config=args.lm_config,
        device=args.device,
        dtype=dtype,
    )

    for path in args.adapter:
        name = hub.register(path)
        print(f"Registered '{name}' from '{path}'")

    tasks = args.task or hub.available()
    predictions = {task: _jsonable(hub.predict(task, sequences, batch_size=args.batch_size))
                   for task in tasks}

    payload = {"sequences": sequences, "predictions": predictions}
    text = json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(text)
        print(f"Predictions written to '{args.output}'")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
