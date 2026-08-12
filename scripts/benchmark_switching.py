#!/usr/bin/env python
"""Measure what multi-adapter residency actually buys: latency and peak GPU memory.

Compares three ways of serving N tasks from one machine:

  1. `set_adapter` switching   -- one backbone, N adapters resident, switch by name
  2. reload adapter each time  -- one backbone, adapters loaded on demand
  3. reload full model each time -- what you must do with N full fine-tuned checkpoints

Written but not executed at build time: it needs `giga-v1.pt`, real adapters and a GPU.

    python scripts/benchmark_switching.py \
        --backbone weights/giga-v1.pt \
        --adapter adapters/splice_site_donor.pt \
        --adapter adapters/mrl.pt \
        --adapter adapters/sec_struct.pt \
        --full-checkpoint weights/rinalmo_giga_splice_donor_ft.pt \
        --repeats 20 --device cuda
"""

from pathlib import Path
import argparse
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rinalmo.config import model_config  # noqa: E402
from rinalmo.model.model import RiNALMo  # noqa: E402
from rinalmo_hub.adapter import load_adapter  # noqa: E402
from rinalmo_hub.hub import RiNALMoHub  # noqa: E402

SEQUENCES = [
    "GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCCUUAGCAUCGAUCGAUCGGAUCG",
    "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAU",
    "GGGAAACCCUUUGGGAAACCCUUUGGGAAACCCUUUGGGAAACCCUUUGGGAAA",
]


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _reset_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_gb(device) -> float:
    if torch.device(device).type != "cuda":
        return float("nan")

    return torch.cuda.max_memory_allocated() / 1e9


def _timeit(fn, repeats, device):
    """Median and p90 wall-clock in milliseconds, after one warm-up call."""
    fn()
    _sync(device)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p90_ms": samples[int(0.9 * (len(samples) - 1))],
        "min_ms": samples[0],
    }


def benchmark_resident_switching(args, tasks):
    """One backbone, every adapter resident, switching by name."""
    _reset_peak_memory(args.device)

    start = time.perf_counter()
    hub = RiNALMoHub(backbone_weights=args.backbone, lm_config=args.lm_config,
                     device=args.device, dtype=args.torch_dtype)
    for path in args.adapter:
        hub.register(path)
    _sync(args.device)
    load_seconds = time.perf_counter() - start

    results = {}

    for task in tasks:
        results[task] = _timeit(lambda t=task: hub.predict(t, SEQUENCES), args.repeats, args.device)

    # Switching on its own, with no forward pass, is the number that matters for an agent
    # that activates a task per request.
    results["_switch_only"] = _timeit(
        lambda: [hub.activate(t) for t in tasks], args.repeats, args.device
    )
    results["_peak_memory_gb"] = _peak_memory_gb(args.device)
    results["_load_seconds"] = load_seconds

    del hub
    if torch.device(args.device).type == "cuda":
        torch.cuda.empty_cache()

    return results


def benchmark_adapter_reload(args, tasks):
    """One backbone kept loaded, but the adapter is re-registered for every request."""
    _reset_peak_memory(args.device)

    paths = {load_adapter(p)["task"]: p for p in args.adapter}

    def one_request(task):
        hub = RiNALMoHub(backbone_weights=None, lm_config=args.lm_config,
                         device=args.device, dtype=args.torch_dtype)
        hub.backbone.load_state_dict(
            torch.load(args.backbone, map_location=args.device, weights_only=False)
        )
        hub.register(paths[task])

        return hub.predict(task, SEQUENCES)

    results = {task: _timeit(lambda t=task: one_request(t), max(1, args.repeats // 4), args.device)
               for task in tasks}
    results["_peak_memory_gb"] = _peak_memory_gb(args.device)

    return results


def benchmark_full_model_reload(args):
    """The full fine-tuning baseline: a fresh 2.6 GB backbone per task, from disk."""
    if not args.full_checkpoint:
        return None

    _reset_peak_memory(args.device)

    def one_request():
        model = RiNALMo(model_config(args.lm_config))
        model.load_state_dict(
            torch.load(args.full_checkpoint, map_location="cpu", weights_only=False)
        )
        model.to(device=args.device, dtype=args.torch_dtype).eval()

        del model
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()

    results = _timeit(one_request, max(1, args.repeats // 10), args.device)
    results["_peak_memory_gb"] = _peak_memory_gb(args.device)
    results["_checkpoint_gb"] = Path(args.full_checkpoint).stat().st_size / 1e9

    return results


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, help="Path to giga-v1.pt")
    parser.add_argument("--adapter", action="append", required=True,
                        help="Adapter '.pt', repeatable")
    parser.add_argument("--full-checkpoint", default=None,
                        help="A full fine-tuned checkpoint, for the reload baseline")
    parser.add_argument("--lm_config", default="giga")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default=None, help="Write the results to this JSON file")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.torch_dtype = getattr(torch, args.dtype)

    tasks = [load_adapter(p)["task"] for p in args.adapter]
    print(f"Benchmarking {len(tasks)} tasks on {args.device} ({args.dtype}): {tasks}\n")

    results = {
        "config": {"device": args.device, "dtype": args.dtype, "repeats": args.repeats,
                   "tasks": tasks},
        "resident_switching": benchmark_resident_switching(args, tasks),
        "adapter_reload": benchmark_adapter_reload(args, tasks),
        "full_model_reload": benchmark_full_model_reload(args),
    }

    print(json.dumps(results, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to '{args.output}'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
