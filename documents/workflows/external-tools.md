# External Tools (flow E)

Classical, non-neural bioinformatics packages wrapped as typed functions, sharing the same
manifest and the same lifecycle as adapter tools.

---

## Contents

1. [What an external tool is](#1-what-an-external-tool-is)
2. [Registering the reference tool](#2-registering-the-reference-tool)
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

Each *function* becomes its own tool entry, named `<family>_<function>`, so ViennaRNA's two
functions register as `vienna_fold` and `vienna_cofold`. From there they are indistinguishable
from adapter tools: same manifest, same `activate`/`deactivate`/`test`, same appearance in
`list_tools`, same binding into the orchestrator.

Wrapped packages are **tool dependencies**: installed into the venv through the gated flow,
recorded in the manifest provenance, and deliberately absent from every `pyproject.toml`.

## 2. Registering the reference tool

```bash
python -m adaptrna_agentic.cli.toolhub register-external \
    adaptrna_agentic.toolhub.external.vienna
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
Registered 'vienna_fold' — Predict the minimum-free-energy secondary structure of one RNA
                           sequence. Returns the dot-bracket structure and its free energy…
Registered 'vienna_cofold' — Predict the minimum-free-energy structure of two RNA strands…
```

## 3. The approval-gated install

```
Package 'ViennaRNA' (import 'RNA') is not installed.
Would run: /home/you/adaptrna/.venv/bin/python -m pip install ViennaRNA
Proceed with the install? [y/N]
```

Same "Would run:" discipline as the training gate — the exact command, before anything runs.
`--yes` approves non-interactively. A non-TTY without `--yes` **declines**:

```
error: Install not approved. Install it yourself with `… -m pip install ViennaRNA`,
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
$toolhub call vienna_fold sequence=GGGGAAAACCCC
# {"structure": "((((....))))", "mfe": -5.4}

$toolhub call vienna_cofold sequence_a=GGGGGGGG sequence_b=CCCCCCCC
$toolhub call vienna_fold --args '{"sequence": "GGGGAAAACCCC"}'

$toolhub test vienna_fold        # the golden cases; exit code 1 if any fail
```

In chat, they are simply tools:

```
you> Fold GGGGAAAACCCC and tell me if it's a hairpin.
  → vienna_fold({sequence: "GGGGAAAACCCC"})
    = {"structure": "((((....))))", "mfe": -5.4}
Yes — four base pairs closing a four-nucleotide loop, at −5.4 kcal/mol.
```

Over HTTP, external tools use `/call` (not `/predict`):

```bash
curl -s -X POST localhost:8000/api/tools/vienna_fold/call \
     -H 'content-type: application/json' -d '{"args": {"sequence": "GGGGAAAACCCC"}}'
```

The tool's LangChain schema is inferred from the wrapper function's **real signature** —
`_external_tool` wraps it with `functools.wraps`, which is the contract's typed JSON-scalar
kwargs paying off.

## 5. Writing a wrapper by hand

[`toolhub/external/vienna.py`](../../agentic/adaptrna_agentic/toolhub/external/vienna.py) is
the reference. The contract is in
[`contract.py`](../modules/toolhub.md#5-external--non-neural-tools):

```python
from adaptrna_agentic.toolhub.external.contract import (
    ExternalToolSpec, FunctionSpec, GoldenCase, PackageSpec)

_VALID_BASES = set("ACGU")

def _clean(sequence: str, label: str = "sequence") -> str:
    """Validate + normalise. Runs BEFORE any import of the wrapped package."""
    seq = (sequence or "").strip().upper().replace("T", "U")
    if not seq:
        raise ValueError(f"Empty {label}: provide at least one base (A, C, G, U or T).")
    invalid = sorted(set(seq) - _VALID_BASES)
    if invalid:
        raise ValueError(f"Invalid characters {invalid} in {label}; expected only A, C, G, U or T.")
    return seq

def fold(sequence: str) -> dict:
    """Minimum-free-energy secondary structure of one RNA sequence."""
    seq = _clean(sequence)

    import RNA                       # AFTER validation — deliberately

    structure, mfe = RNA.fold(seq)
    return {"structure": structure, "mfe": round(float(mfe), 2)}

SPEC = ExternalToolSpec(
    name="vienna",                                   # the family prefix
    description="Thermodynamic RNA secondary-structure prediction via ViennaRNA.",
    package=PackageSpec(pip="ViennaRNA", import_name="RNA"),
    functions=(
        FunctionSpec(
            name="fold",
            description="…what the MODEL reads to decide whether to call this…",
            golden=(
                GoldenCase(args={"sequence": "AAAAAAAAAAAA"},
                           expect={"structure": "............",
                                   "mfe": {"approx": 0.0, "tol": 0.01}}),
                GoldenCase(args={"sequence": "GGGGAAAACCCC"},
                           expect={"structure": "((((....))))",
                                   "mfe": {"approx": -5.4, "tol": 0.5}}),
            ),
        ),
    ),
)
```

### The three rules

1. **`SPEC` at module level**, typed as `ExternalToolSpec`.
2. **One module-level callable per declared function**, taking JSON-scalar keyword arguments
   and returning a JSON-serialisable dict.
3. **Validate inputs before importing the wrapped package**, so a missing package fails at
   the call boundary with the install hint, and validation tests run without it installed.

### Golden cases

`expect` values compare exactly; `{"approx": x, "tol": t}` compares within tolerance. They
are **captured against the installed version, never guessed** — and the strongest ones mix
two kinds of evidence:

| Kind | Example | Why it is good |
|---|---|---|
| *a priori* | `AAAA…` → all dots, 0.0 | A homopolymer cannot pair with itself. True of any correct implementation. |
| captured | `GGGGAAAACCCC` → `((((....))))` at −5.4 | Pins the actual behaviour of ViennaRNA 2.7.2 |

The cofold golden documents a non-obvious detail worth copying: the returned structure spans
both strands **with the `&` dropped**, so 8+8 bases give a 16-character string.

Function `description` is what the orchestrator reads when deciding whether to call the tool
— write it for that audience.

## 6. Generating one

```
you> Wrap the ViennaRNA package so I can fold sequences.
  → create_external_tool({name: "vienna", package: "ViennaRNA",
                          description: "MFE structure prediction"})
```

The ToolSmith is given the **full text of `contract.py` and of `vienna.py`** as the reference
to imitate, plus the instruction that golden cases must be values it is confident about a
priori — never invented numbers.

Verification differs from the task flow, deliberately:

* it runs **in process**, not in the sandbox — *wrapper modules are small and import a
  package the user already approved, so process isolation buys little here; the contract
  loader is the gate*;
* `contract.load_spec` is check one, and a failure there ends the attempt;
* if the wrapped package is not installed, the golden check is recorded as **`skip` and the
  attempt still succeeds** — the package's absence is a property of the environment, not of
  the code. (This is the opposite of the task flow, where a skipped required check is a
  failure, because there the missing thing is *your data*.)

Same bounded loop (≤ 3 attempts), same staging, same approval gate. Landed wrappers go to
`adaptrna_custom/tools/<name>.py`, and the tool result reminds you of the next step:

> *"Register its functions as tools with the external-tool flow."*

```bash
$toolhub register-external adaptrna_custom.tools.<name>
```

## 7. Where the pieces live

| Path | Role |
|---|---|
| [`toolhub/external/contract.py`](../../agentic/adaptrna_agentic/toolhub/external/contract.py) | The contract, the loader, install helpers, the golden runner |
| [`toolhub/external/vienna.py`](../../agentic/adaptrna_agentic/toolhub/external/vienna.py) | The hand-written reference, and the codegen template |
| `Registry.register_external` | Spec → one manifest entry per function |
| `cli/toolhub.py::cmd_register_external` | The install gate |
| `agents/tool_factory.py::_external_tool` | Manifest entry → LangChain tool |
| `api/routers/tools.py::call_tool` | `POST /api/tools/{name}/call` |
| `adaptrna_custom/tools/` | Where generated wrappers land |

The manifest entry for an external tool records `{module, function, package: {pip,
import_name, installed_version}}` plus the copied golden cases —
[../configuration.md §5](../configuration.md#5-toolhub_datatoolsjson--the-manifest).

## 8. What can go wrong

| Symptom | Meaning | Fix |
|---|---|---|
| `does not define SPEC: ExternalToolSpec` | Not a wrapper module | Follow the contract; see `vienna.py` |
| `SPEC … declares function 'x' but the module does not define it` | Spec and code disagree | Define it, or drop it from `SPEC` |
| `Package 'X' (import 'y') is not installed` | The gate, refusing to install unasked | Run the printed command, or rerun with `--yes` |
| `Install failed (…)` | pip failed; the last 2000 chars of stderr are included | Usually a build dependency |
| `'vienna_fold' is already registered` | Name collision | `toolhub remove vienna_fold` first, or use `--only` for a subset |
| `toolhub test` fails after an upgrade | The goldens were captured against an older version | Re-verify the values by hand, then update the wrapper's `SPEC` **and** re-register, since goldens are copied into the manifest at registration |
| `doctor` reports `external_tools` FAIL | A registered tool's package is no longer importable | Reinstall the package, or remove the tool |
| `Tool 'x' is disabled` | Routing-level deactivation | `toolhub activate x`, or just ask in chat |
