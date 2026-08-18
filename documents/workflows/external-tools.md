# External Tools (flow E)

Classical, non-neural packages wrapped as typed functions, sharing the same manifest and the
same lifecycle as adapter tools.

---

## Contents

1. [What an external tool is](#1-what-an-external-tool-is)
2. [Registering a wrapper](#2-registering-a-wrapper)
3. [The approval-gated install](#3-the-approval-gated-install)
4. [Using it](#4-using-it)
5. [Writing a wrapper by hand](#5-writing-a-wrapper-by-hand)
6. [Generating one](#6-generating-one)
7. [Where the pieces live](#7-where-the-pieces-live)
8. [What can go wrong](#8-what-can-go-wrong)

---

## 1. What an external tool is

Three parts:

* a **Python wrapper module** exposing typed functions with JSON-scalar keyword arguments
  and dict returns;
* an **install spec** (`pip` distribution name + the top-level module it provides), which is
  approval-gated;
* **golden smoke tests** — a known input and its expected output, captured against the
  installed version.

Each *function* becomes its own tool entry, named `<family>_<function>`, so a wrapper family
named `fold` with functions `predict` and `predict_pair` would register as `fold_predict` and
`fold_predict_pair`. From there they are indistinguishable from adapter tools: same manifest,
same `activate`/`deactivate`/`test`, same appearance in `list_tools`, same binding into the
orchestrator.

Wrapped packages are **tool dependencies**: installed into the venv through the gated flow,
recorded in the manifest provenance, and deliberately absent from every `pyproject.toml`.

## 2. Registering a wrapper

Once you have a wrapper module — written by hand (§5) or generated (§6) and landed at
`adaptrna_custom/tools/<name>.py` — register its functions as tools:

```bash
python -m adaptrna_agentic.cli.toolhub register-external adaptrna_custom.tools.<your_tool>
```

What happens:

1. `contract.load_spec(module_path)` imports the module and validates the contract — a
   missing `SPEC`, or a declared function the module does not define, fails here.
2. If the package is not importable, the **gate** fires (next section).
3. Every function in the spec — or the `--only a,b` subset — becomes a manifest entry, with
   its golden cases **copied in** so the entry stays self-contained even if the module's
   `SPEC` changes later.
4. The whole batch is refused before anything is written if any tool name already exists.

```
Registered 'fold_predict' — Predict the minimum-free-energy secondary structure of one
                             sequence. Returns the dot-bracket structure and its free energy…
Registered 'fold_predict_pair' — Predict the minimum-free-energy structure of two strands…
```

## 3. The approval-gated install

```
Package 'YourPackage' (import 'your_import') is not installed.
Would run: /home/you/adaptrna/.venv/bin/python -m pip install YourPackage
Proceed with the install? [y/N]
```

Same "Would run:" discipline as the training gate — the exact command, before anything runs.
`--yes` approves non-interactively. A non-TTY without `--yes` **declines**:

```
error: Install not approved. Install it yourself with `… -m pip install YourPackage`,
       or rerun with --yes.
```

`contract.install_command()` builds the argv; `contract.install()` runs it. Only ever the
spec-named package, only ever into this venv. The installed version is recorded in the
manifest entry's `external.package.installed_version`, which is what makes a golden test
meaningful later.

Registering through the `Registry` API directly does **not** install anything — it refuses
with the exact command to run, because the CLI owns the gate.

## 4. Using it

```bash
$toolhub call fold_predict sequence=GGGGAAAACCCC
# {"structure": "((((....))))", "mfe": -5.4}

$toolhub call fold_predict --args '{"sequence": "GGGGAAAACCCC"}'
$toolhub test fold_predict        # the golden cases; exit code 1 if any fail
```

In chat, they are simply tools:

```
you> Fold GGGGAAAACCCC and tell me if it's a hairpin.
  → fold_predict({sequence: "GGGGAAAACCCC"})
    = {"structure": "((((....))))", "mfe": -5.4}
Yes — four base pairs closing a four-nucleotide loop, at −5.4 kcal/mol.
```

Over HTTP, external tools use `/call` (not `/predict`):

```bash
curl -s -X POST localhost:8000/api/tools/fold_predict/call \
     -H 'content-type: application/json' -d '{"args": {"sequence": "GGGGAAAACCCC"}}'
```

The tool's LangChain schema is inferred from the wrapper function's **real signature** —
`_external_tool` wraps it with `functools.wraps`, which is the contract's typed JSON-scalar
kwargs paying off.

## 5. Writing a wrapper by hand

There is no shipped wrapper to imitate any more — the contract in
[`contract.py`](../../agentic/adaptrna_agentic/toolhub/external/contract.py) is the whole
teaching surface, and it stands on its own:

```python
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class PackageSpec:
    pip: str            # distribution name, e.g. "YourPackage"
    import_name: str    # top-level module it provides, e.g. "your_import"

@dataclass(frozen=True)
class GoldenCase:
    args: Dict[str, Any]
    expect: Dict[str, Any]

@dataclass(frozen=True)
class FunctionSpec:
    name: str
    description: str
    golden: Tuple[GoldenCase, ...] = ()

@dataclass(frozen=True)
class ExternalToolSpec:
    name: str                              # family prefix: tools register as <name>_<fn>
    description: str
    package: PackageSpec
    functions: Tuple[FunctionSpec, ...]
```

A wrapper module must:

1. **define `SPEC: ExternalToolSpec` at module level** — the declarative description of the
   tool family: its package, its functions, and their golden test cases;
2. **define one module-level callable per declared function**, taking JSON-scalar keyword
   arguments and returning a JSON-serialisable dict — `contract.load_spec` checks every
   `FunctionSpec.name` in `SPEC.functions` resolves to a real, callable module attribute, and
   refuses to register the module otherwise;
3. **validate inputs before importing the wrapped package**, so a missing package fails at
   the call boundary with the install hint (rather than a bare `ImportError` from deep
   inside), and validation tests can run without the package installed at all.

`agentic/tests/fixtures/validating_external.py` is a small, fully contract-compliant wrapper
that exists to demonstrate exactly this shape end to end — worth reading as a template for
the mechanics, independent of what package you are actually wrapping:

```python
from adaptrna_agentic.toolhub.external.contract import (
    ExternalToolSpec, FunctionSpec, GoldenCase, PackageSpec)

def _clean(text: str, label: str = "text") -> str:
    """Validate + normalise. Runs BEFORE any import of the wrapped package."""
    value = (text or "").strip()
    if not value:
        raise ValueError(f"Empty {label}: provide at least one character.")
    if not value.isprintable():
        raise ValueError(f"Non-printable characters in {label}.")
    return value

def checksum(text: str) -> dict:
    """CRC32 checksum of one string."""
    value = _clean(text)

    import zlib                      # AFTER validation — deliberately

    return {"crc32": zlib.crc32(value.encode("utf-8"))}

SPEC = ExternalToolSpec(
    name="checksum",
    description="CRC32 checksums over text, via the stdlib zlib module.",
    package=PackageSpec(pip="stdlib", import_name="zlib"),
    functions=(
        FunctionSpec(
            name="checksum",
            description="CRC32 checksum of one string.",
            golden=(GoldenCase(args={"text": "hello"}, expect={"crc32": 907060870}),),
        ),
    ),
)
```

### Golden cases

`expect` values compare exactly; `{"approx": x, "tol": t}` compares within tolerance. They
are **captured against the installed version, never guessed** — the strongest ones mix two
kinds of evidence:

| Kind | Example | Why it is good |
|---|---|---|
| *a priori* | a value you can derive by hand or reason about without running the package | True of any correct implementation |
| captured | a value read off one real run of the installed package | Pins the actual behaviour of the version you have |

Function `description` is what the orchestrator reads when deciding whether to call the tool
— write it for that audience.

## 6. Generating one

```
you> Wrap YourPackage so I can call it.
  → create_external_tool({name: "fold", package: "YourPackage",
                          description: "MFE structure prediction"})
```

The ToolSmith is given the **full text of `contract.py`** as the specification to satisfy —
there is no shipped wrapper handed over as a reference implementation any more (the same
reasoning that removed the shipped task shown to the task generator, D2 in
[the phase plan](../../plans/PHASE_13_COLD_START_SINGLE_CSV.md)) — plus the instruction that
golden cases must be values it is confident about a priori, never invented numbers.

Verification differs from the task flow, deliberately:

* it runs **in process**, not in the sandbox — *wrapper modules are small and import a
  package the user already approved, so process isolation buys little here; the contract
  loader is the gate*;
* `contract.load_spec` is check one, and a failure there ends the attempt;
* if the wrapped package is not installed, the golden check is recorded as **`skip` and the
  attempt still succeeds** — the package's absence is a property of the environment, not of
  the code. (This is the opposite of the task flow, where a skipped required check is a
  failure, because there the missing thing is *your data*.)

Same bounded loop (≤ 3 attempts), same staging, same **approval gate**. The approval window
displays the generated code in a read-only Monaco editor — with a tab per file if multiple
files are involved — so you can review it inline before approving.

When you approve `land_generated_code`, two things happen atomically:

1. The wrapper is written to `adaptrna_custom/tools/<name>.py`.
2. `Registry.register_external` is called immediately — the tools are **active and usable at
   once**, with no manual follow-up step.

```
approve → write adaptrna_custom/tools/fold.py
        → register fold_predict, fold_predict_pair
        → both tools are live
```

**Caveat:** the wrapped package must be installed before approval. If it is not, the
registration step surfaces the exact `pip install` command as the tool result (the same
message as the CLI's refusal). Install the package and land again.

If you prefer to manage registration manually (or need to re-register after modifying a landed
wrapper), the CLI remains available:

```bash
python -m adaptrna_agentic.cli.toolhub register-external adaptrna_custom.tools.<name>
```

## 7. Where the pieces live

| Path | Role |
|---|---|
| [`toolhub/external/contract.py`](../../agentic/adaptrna_agentic/toolhub/external/contract.py) | The contract, the loader, install helpers, the golden runner |
| [`agentic/tests/fixtures/validating_external.py`](../../agentic/tests/fixtures/validating_external.py) | A minimal, tested, contract-compliant wrapper — worth reading for the mechanics |
| `Registry.register_external` | Spec → one manifest entry per function |
| `cli/toolhub.py::cmd_register_external` | The install gate |
| `agents/tool_factory.py::_external_tool` | Manifest entry → LangChain tool |
| `api/routers/tools.py::call_tool` | `POST /api/tools/{name}/call` |
| `adaptrna_custom/tools/` | Where wrappers — hand-written or generated — land |

The manifest entry for an external tool records `{module, function, package: {pip,
import_name, installed_version}}` plus the copied golden cases —
[../configuration.md §5](../configuration.md#5-toolhub_datatoolsjson--the-manifest).

## 8. What can go wrong

| Symptom | Meaning | Fix |
|---|---|---|
| `does not define SPEC: ExternalToolSpec` | Not a wrapper module | Follow the contract in §5 |
| `SPEC … declares function 'x' but the module does not define it` | Spec and code disagree | Define it, or drop it from `SPEC` |
| `Package 'X' (import 'y') is not installed` | The gate, refusing to install unasked | Run the printed command, or rerun with `--yes` |
| `Install failed (…)` | pip failed; the last 2000 chars of stderr are included | Usually a build dependency |
| `'x_fn' is already registered` | Name collision | `toolhub remove x_fn` first, or use `--only` for a subset |
| `toolhub test` fails after an upgrade | The goldens were captured against an older version | Re-verify the values by hand, then update the wrapper's `SPEC` **and** re-register, since goldens are copied into the manifest at registration |
| `doctor` reports `external_tools` FAIL | A registered tool's package is no longer importable | Reinstall the package, or remove the tool |
| `Tool 'x' is disabled` | Routing-level deactivation | `toolhub activate x`, or just ask in chat |
