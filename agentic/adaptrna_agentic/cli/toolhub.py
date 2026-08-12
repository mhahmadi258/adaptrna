"""ToolHub management CLI.

    python -m adaptrna_agentic.cli.toolhub list
    python -m adaptrna_agentic.cli.toolhub register outputs/run/<task>_adapter.pt
    python -m adaptrna_agentic.cli.toolhub predict splice_site --sequences ACGU...

Registry operations are instant; `predict`, `test` and `warmup` load the backbone (lazy
by design). Residency is per process: each CLI invocation pays its own backbone load —
the long-lived process holding one warm `AdapterRuntime` arrives with the Phase 4 chat
and the Phase 8 service.
"""

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import argparse
import json
import sys

from adaptrna_agentic.toolhub.registry import Registry, ToolHubError
from adaptrna_agentic.toolhub.runtime import AdapterRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adaptrna_agentic.cli.toolhub",
        description="Manage and run AdaptRNA tools (adapters in one shared backbone).",
    )
    parser.add_argument("--data-dir", type=str, default=None,
                        help="ToolHub state dir (default: $ADAPTRNA_TOOLHUB_DIR or <repo>/toolhub_data)")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List registered tools").set_defaults(func=cmd_list)

    p = sub.add_parser("register", help="Register a LoRA adapter file as a tool")
    p.add_argument("adapter", help="Path to the adapter '.pt' file")
    p.add_argument("--name", default=None, help="Tool name (default: the adapter's task)")
    p.add_argument("--description", default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Serving batch size (default: the task's own default)")
    p.add_argument("--test-sequences", nargs="+", default=None,
                   help="Sequences for `toolhub test`")
    p.add_argument("--test-input", default=None,
                   help="File with one test sequence per line ('>'/'#' lines skipped)")
    p.add_argument("--link", action="store_true",
                   help="Reference the file in place instead of copying it into toolhub_data/")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("register-external",
                       help="Register an external wrapper module's functions as tools")
    p.add_argument("module",
                   help="Import path, e.g. adaptrna_agentic.toolhub.external.vienna")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of the module's functions")
    p.add_argument("--yes", action="store_true",
                   help="Approve the package install without prompting")
    p.set_defaults(func=cmd_register_external)

    p = sub.add_parser("call", help="Invoke an external tool function")
    p.add_argument("name")
    p.add_argument("kv", nargs="*", metavar="KEY=VALUE",
                   help="Function arguments, e.g. sequence=GGGGAAAACCCC")
    p.add_argument("--args", default=None, help="Arguments as a JSON object")
    p.set_defaults(func=cmd_call)

    for verb in ("activate", "deactivate"):
        p = sub.add_parser(verb, help=f"{verb.capitalize()} a tool")
        p.add_argument("name")
        p.set_defaults(func=cmd_activate if verb == "activate" else cmd_deactivate)

    p = sub.add_parser("remove", help="Remove a tool (deletes registry-owned artifact copies)")
    p.add_argument("name")
    p.add_argument("--keep-artifact", action="store_true")
    p.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("info", help="Show a tool's manifest entry and adapter summary")
    p.add_argument("name")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("test", help="Smoke-test a tool (loads the backbone)")
    p.add_argument("name")
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("predict", help="Run sequences through a tool (loads the backbone)")
    p.add_argument("name")
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--input", default=None,
                   help="File with one sequence per line ('>'/'#' lines skipped)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", default=None, help="Write predictions to this JSON file")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("config", help="Show or update the backbone configuration")
    p.add_argument("--weights", default=None, help="Checkpoint path ('null' clears it)")
    p.add_argument("--lm-config", default=None, help="nano | micro | mega | giga")
    p.add_argument("--device", default=None, help="auto | cpu | cuda")
    p.add_argument("--dtype", default=None, help="auto | float32 | bfloat16 | float16")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("doctor", help="Check the install and report what is wrong")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("prune", help="Delete unreferenced state (dry run unless --yes)")
    p.add_argument("what", choices=("staging", "sessions", "jobs", "runs", "artifacts"))
    p.add_argument("--older-than", type=float, default=None, metavar="DAYS")
    p.add_argument("--yes", action="store_true", help="Actually delete")
    p.set_defaults(func=cmd_prune)

    sub.add_parser("warmup", help="Eagerly load the backbone + active adapters (this process)"
                   ).set_defaults(func=cmd_warmup)
    sub.add_parser("rebuild", help="Drop the resident hub (this process)"
                   ).set_defaults(func=cmd_rebuild)

    return parser


# ---------------------------------------------------------------------- commands

def cmd_list(args) -> int:
    entries = Registry(args.data_dir).list()
    if not entries:
        print("No tools registered. Add one with: toolhub register <adapter.pt>")
        return 0

    rows = [("NAME", "STATE", "TYPE", "TASK", "BATCH", "SOURCE")]
    for e in entries:
        batch = e.serving.get("batch_size")
        batch_label = str(batch) if batch else ("task default" if e.type == "adapter" else "-")
        rows.append((e.name, e.state, e.type, e.task or "-", batch_label,
                     e.provenance.get("source", e.artifact or "-")))

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        lead = "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
        print(f"{lead}  {row[-1]}")
    return 0


def cmd_register(args) -> int:
    test_sequences = list(args.test_sequences or [])
    if args.test_input:
        test_sequences.extend(_read_sequences_file(args.test_input))

    entry = Registry(args.data_dir).register(
        args.adapter,
        name=args.name,
        description=args.description,
        batch_size=args.batch_size,
        test_sequences=test_sequences or None,
        link=args.link,
    )
    print(f"Registered '{entry.name}' (task: {entry.task}, state: {entry.state})")
    print(json.dumps(asdict(entry), indent=2))
    return 0


def cmd_activate(args) -> int:
    entry = Registry(args.data_dir).activate(args.name)
    print(f"'{entry.name}' is now {entry.state}")
    return 0


def cmd_deactivate(args) -> int:
    entry = Registry(args.data_dir).deactivate(args.name)
    print(f"'{entry.name}' is now {entry.state}")
    return 0


def cmd_remove(args) -> int:
    registry = Registry(args.data_dir)
    entry = registry.get(args.name)

    if not args.yes:
        reply = input(f"Remove tool '{entry.name}'"
                      f"{'' if args.keep_artifact else ' and its artifact copy'}? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    registry.remove(args.name, keep_artifact=args.keep_artifact)
    print(f"Removed '{args.name}'")
    return 0


def cmd_register_external(args) -> int:
    from adaptrna_agentic.toolhub.external import contract

    spec, _module = contract.load_spec(args.module)

    if not contract.is_available(spec.package):
        command = " ".join(contract.install_command(spec.package))
        print(f"Package '{spec.package.pip}' (import '{spec.package.import_name}') "
              f"is not installed.")
        print(f"Would run: {command}")

        if args.yes:
            approved = True
        elif sys.stdin.isatty():
            approved = input("Proceed with the install? [y/N] ").strip().lower() in ("y", "yes")
        else:
            approved = False

        if not approved:
            raise ToolHubError(
                f"Install not approved. Install it yourself with `{command}`, "
                f"or rerun with --yes."
            )

        version = contract.install(spec.package)
        print(f"Installed {spec.package.pip} {version}")

    only = [part.strip() for part in args.only.split(",")] if args.only else None
    entries = Registry(args.data_dir).register_external(args.module, only=only)
    for entry in entries:
        print(f"Registered '{entry.name}' — {entry.description}")
    return 0


def cmd_call(args) -> int:
    registry = Registry(args.data_dir)
    entry = registry.get(args.name)

    if entry.type != "external":
        raise ToolHubError(
            f"'{args.name}' is an {entry.type} tool; use `toolhub predict {args.name} ...`."
        )
    if not entry.active:
        raise ToolHubError(
            f"Tool '{args.name}' is disabled. Enable it with `toolhub activate {args.name}`."
        )

    arguments = {}
    if args.args:
        arguments.update(json.loads(args.args))
    for pair in args.kv:
        if "=" not in pair:
            raise ToolHubError(f"Expected KEY=VALUE, got '{pair}'")
        key, _, value = pair.partition("=")
        arguments[key] = value

    from adaptrna_agentic.toolhub.external.contract import call_entry

    result = call_entry(entry, arguments)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_info(args) -> int:
    entry = Registry(args.data_dir).get(args.name)
    print(json.dumps(asdict(entry), indent=2))

    if entry.type == "adapter":
        from rinalmo_hub.adapter import describe_adapter

        print()
        print(describe_adapter(entry.artifact_path()))
    return 0


def cmd_test(args) -> int:
    registry = Registry(args.data_dir)
    entry = registry.get(args.name)

    if entry.type == "external":
        from adaptrna_agentic.toolhub.external.contract import run_golden

        report = run_golden(entry)
    else:
        report = AdapterRuntime(registry).smoke_test(args.name)

    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


def cmd_predict(args) -> int:
    sequences = list(args.sequences or [])
    if args.input:
        sequences.extend(_read_sequences_file(args.input))
    if not sequences:
        raise ToolHubError("No sequences given: pass --sequences or --input.")

    runtime = AdapterRuntime(Registry(args.data_dir))
    outputs = runtime.predict(args.name, sequences, batch_size=args.batch_size)

    payload = {"sequences": sequences, "predictions": {args.name: _jsonable(outputs)}}
    text = json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(text)
        print(f"Predictions written to '{args.output}'")
    else:
        print(text)
    return 0


def cmd_config(args) -> int:
    registry = Registry(args.data_dir)
    if any(v is not None for v in (args.weights, args.lm_config, args.device, args.dtype)):
        backbone = registry.configure_backbone(
            weights=args.weights, lm_config=args.lm_config,
            device=args.device, dtype=args.dtype,
        )
        print("Backbone configuration updated:")
    else:
        backbone = registry.manifest.backbone
    print(json.dumps(asdict(backbone), indent=2))
    return 0


def cmd_doctor(args) -> int:
    from adaptrna_agentic.toolhub import doctor

    report = doctor.run_checks(args.data_dir)
    print(json.dumps(report, indent=2, default=str) if args.json
          else doctor.format_report(report))

    return 1 if report["status"] == "fail" else 0


def cmd_prune(args) -> int:
    from adaptrna_agentic.toolhub import prune as prune_module

    report = prune_module.prune(
        args.what, older_than=args.older_than, apply=args.yes, data_dir=args.data_dir
    )
    print(prune_module.format_report(report))

    return 0


def cmd_warmup(args) -> int:
    runtime = AdapterRuntime(Registry(args.data_dir))
    problems = runtime.warmup()
    resident = sorted(runtime._resident)
    print(f"Backbone loaded; resident adapters: {resident or 'none'}")
    for problem in problems:
        print(f"  skipped: {problem}", file=sys.stderr)
    return 0


def cmd_rebuild(args) -> int:
    runtime = AdapterRuntime(Registry(args.data_dir))
    runtime.rebuild()
    print("Hub state dropped for this process. (Residency is per process; a fresh "
          "invocation always starts empty.)")
    return 0


# ---------------------------------------------------------------------- helpers

def _read_sequences_file(path: str) -> List[str]:
    sequences = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith((">", "#")):
            sequences.append(line)
    return sequences


def _jsonable(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ToolHubError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        print(f"error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
