# Phase 3 — External tools + the ViennaRNA reference (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 3; design contract in §4.
> **Definition of done:** `toolhub test vienna_fold` passes; enable/disable works.
> Status: planned · not started

---

## 1. Context and goal

Phase 2 gave the ToolHub one tool kind (adapters on the shared backbone). Phase 3 adds the
second: **external tools** — classical, non-neural bioinformatics packages wrapped as typed
Python functions — sharing the *same* manifest, lifecycle and CLI, so `list`, `activate`,
`deactivate`, `test` and `info` treat both kinds uniformly (MASTER_PLAN §3.5).

The first external tool, **ViennaRNA**, is **hand-written on purpose**: it is the reference
implementation *and* the template Phase 6's ToolSmith will imitate when generating wrappers
for packages the user introduces later. That dual role drives the design — the wrapper
contract must be declarative, mechanical, and verifiable by code, not by convention.

Still deterministic, still no LLM. Verified at plan time: ViennaRNA 2.7.2 on PyPI (py3.12
wheels), `import RNA` currently **absent** from the venv — so the DoD run exercises the
approval-gated install flow for real, from a clean state.

**Out of scope:** generated wrappers and the ToolSmith (Phase 6), the agent integration
(Phase 4), sandboxing generated code (Phase 6 — Phase 3 code is first-party and trusted),
non-pip installers (conda/apt noted as a documented fallback only).

## 2. Decisions this plan fixes

| Decision | Choice | Rationale |
|---|---|---|
| **Tool granularity** | One manifest entry **per function**, named `<family>_<function>` (`vienna_fold`, `vienna_cofold`); one wrapper module per family registers all of its functions at once | Matches the DoD name exactly; each callable becomes one agent tool in Phase 4; finest activate/disable control ("disable vienna" in chat later = disable all `vienna_*`) |
| **Wrapper location** | First-party wrappers live in the package: `adaptrna_agentic/toolhub/external/vienna.py` | This is trusted, reviewed code. Where Phase 6's *generated* wrappers land stays an open §10 decision — the plan does not pre-empt it |
| **Wrapper contract** | A module-level `SPEC: ExternalToolSpec` (dataclass: name, description, `PackageSpec(pip, import_name)`, `FunctionSpec`s with golden cases) + plain typed functions; a loader validates the contract at registration | Declarative enough for ToolSmith to generate and for the Verifier to check mechanically; no inheritance, no magic |
| **Manifest evolution** | **Additive, stays format_version 1**: `task`/`lm_config`/`artifact` become Optional (adapter-only); new optional `external` dict ({module, function, package: {pip, import_name, installed_version}}) | Existing manifests load unchanged (missing keys → dataclass defaults); nothing that exists gets reinterpreted |
| **Install flow** | `register-external` checks importability (`importlib.util.find_spec`); if absent it prints the exact `pip install` command and requires **explicit approval** — interactive `[y/N]` or `--yes`; install runs in the current venv via `python -m pip install <pip-spec>`; installed version recorded in provenance | The approval gate is the human at this phase; Phase 4 wires the same gate through `interrupt()`. Only the spec-named package is ever installed, and the command is shown before it runs |
| **Golden tests** | Captured **at implementation time** against the installed version, then pinned into the SPEC: exact match for dot-bracket structures, tolerance (±0.5 kcal/mol) for energies; the installed version in provenance explains any future drift | The plan does not guess ViennaRNA outputs; it defines the capture procedure (§6) with a plausibility check (balanced brackets) before pinning |

## 3. The wrapper contract (the template ToolSmith will imitate)

`adaptrna_agentic/toolhub/external/contract.py`:

```python
@dataclass(frozen=True)
class PackageSpec:
    pip: str                      # "ViennaRNA"
    import_name: str              # "RNA"

@dataclass(frozen=True)
class GoldenCase:
    args: dict                    # {"sequence": "GGGGAAAACCCC"}
    expect: dict                  # {"structure": "((((....))))", "mfe": {"approx": -3.1, "tol": 0.5}}

@dataclass(frozen=True)
class FunctionSpec:
    name: str                     # must be a module-level callable
    description: str              # becomes the agent-tool description in Phase 4
    golden: tuple[GoldenCase, ...]

@dataclass(frozen=True)
class ExternalToolSpec:
    name: str                     # family prefix: "vienna"
    description: str
    package: PackageSpec
    functions: tuple[FunctionSpec, ...]
```

Contract rules, enforced by a loader (`load_spec(module_path)`): the module defines `SPEC`
of this type; every `FunctionSpec.name` is a module-level callable; function inputs are
JSON-scalar kwargs and outputs are JSON-serializable dicts (agent-tool-ready); **input
validation runs before the package import**, so a missing package fails at the call
boundary with the install hint, and validation tests run without the package installed.
`expect` values: plain values compare exactly; `{"approx": x, "tol": t}` compares within
tolerance. These rules go in `contract.py`'s docstring — they are the Phase 6 checklist.

### The ViennaRNA reference (`external/vienna.py`)

Two functions (enough to prove the multi-function shape):

- `fold(sequence: str) -> {"structure": str, "mfe": float}` — MFE dot-bracket via
  `RNA.fold`; input cleaned (upper-case, `T`→`U`, alphabet-validated) *before* `import RNA`.
- `cofold(sequence_a: str, sequence_b: str) -> {"structure": str, "mfe": float}` — dimer
  MFE via `RNA.cofold("a&b")`.

Golden candidates (pinned after capture, §6): a strong hairpin (`GGGGAAAACCCC` →
expected shape `((((....))))`), and the degenerate case `AAAAAAAAAAAA` → structure
`............`, mfe `0.0` (predictable a priori); one cofold duplex case.

## 4. Component changes

```
agentic/adaptrna_agentic/toolhub/
├── external/
│   ├── __init__.py
│   ├── contract.py        # NEW — dataclasses, load_spec(), golden runner, install helpers
│   └── vienna.py          # NEW — the reference wrapper (SPEC + fold + cofold)
├── manifest.py            # ToolEntry: task/lm_config/artifact Optional; + external field
├── registry.py            # + register_external(); remove() tolerates artifact=None
└── runtime.py             # unchanged (already refuses non-adapter types)
agentic/adaptrna_agentic/cli/toolhub.py    # + register-external, call; test/info dispatch by type
```

- **`contract.py`** additionally provides: `is_available(spec) -> bool`;
  `install_command(spec) -> list[str]` (constructed, not executed — testable);
  `install(spec) -> str` (runs it, returns installed version via
  `importlib.metadata.version`); `run_golden(entry) -> report` with the same report shape
  as Phase 2's `smoke_test` ({name, ok, checks, outputs}).
- **`registry.register_external(module_path, only=None, install=False, assume_yes=False)`**
  → loads + validates SPEC; availability check → approval-gated install path; creates one
  entry per function (or the `only` subset): `type="external"`, `state="active"`,
  description from the FunctionSpec, `external={module, function, package+installed_version}`,
  `test={"golden": [...]}` copied from the SPEC (the manifest is self-contained — `test`
  works even if the module's SPEC later changes). Duplicate names rejected as for adapters.
- **CLI**:
  - `register-external MODULE [--only fold,cofold] [--yes]` — the `[y/N]` prompt prints
    the exact pip command first; declining aborts with instructions.
  - `call NAME key=value ... | --args '{"...": ...}'` — invoke an external function;
    refuses disabled tools with the `activate` hint (same routing-level semantics as
    adapters). `predict` on an external tool errors pointing to `call`, and vice versa.
  - `test NAME` — dispatches by entry type: adapter → `AdapterRuntime.smoke_test`,
    external → `run_golden`. `info` shows the SPEC-derived fields for externals.
  - `list` already shows TYPE; SOURCE column shows the module path for externals.

## 5. Tests (all deterministic; **no test installs anything or requires ViennaRNA**)

A tests-local fixture module (`tests/fixtures/dummy_external.py`) with a `SPEC` whose
package is something always importable (`import_name="json"`, pip name irrelevant) and two
functions (`echo`, `add`) incl. golden cases — so every code path runs without touching
PyPI.

| test file | asserts |
|---|---|
| `test_external_contract.py` | `load_spec` on the dummy passes; module without `SPEC`, SPEC naming a missing function, non-callable → distinct actionable errors; golden runner: pass, exact-mismatch fail, approx within/outside tolerance; `install_command` builds `python -m pip install ViennaRNA` from the spec without executing |
| `test_external_registry.py` | register_external creates `dummy_echo`+`dummy_add` (active, type external, provenance has module+package); `--only` subset; duplicate rejected; activate/deactivate/remove round-trip (remove with `artifact=None` works — the Phase 2 `remove` gets its None-guard); adapters and externals coexist in one `list`; **manifest from Phase 2 (adapter-only, no `external` keys) still loads** |
| `test_external_cli.py` | `register-external` on the dummy via CLI; missing package path: availability monkeypatched False → without `--yes` (non-interactive) exits 1 printing the pip command; with `--yes` calls the (monkeypatched) installer; `call dummy_add --args ...` returns JSON; key=value arg form; disabled tool refuses `call`; `predict` on external errors pointing to `call`; `test dummy_echo` exit 0 / doctored golden → exit 1 |
| `test_vienna_wrapper.py` | validation-only tests run **everywhere** (invalid characters, empty input — they precede `import RNA`); the fold/cofold behavior tests are `skipif find_spec("RNA") is None` and assert golden agreement once vienna is installed (they turn on automatically after the DoD run) |

Phase 0+2's 52 tests stay green; engine's 135 untouched.

## 6. Implementation order

1. `contract.py` + dummy fixture + `test_external_contract.py`.
2. `manifest.py` additive fields + backward-compat test; `registry.register_external` +
   `remove` None-guard + `test_external_registry.py`.
3. CLI (`register-external`, `call`, type-dispatching `test`/`info`) + `test_external_cli.py`.
4. `vienna.py` with SPEC (goldens provisional) + validation tests.
5. **DoD run** (§7): install via the gated flow, then **capture goldens** — run
   `fold`/`cofold` on the candidates, check plausibility (balanced brackets, hairpin
   shape, `AAAA…` → no structure), pin the outputs + tolerance into `SPEC`, and re-run
   `toolhub test` so the manifest carries the final goldens (re-register or update entry).
6. Docs: agentic README external-tools section; `contract.py` docstring is the ToolSmith
   template reference. Close-out: MASTER_PLAN §8 tick (no §10 items belong to Phase 3).

## 7. Verification / definition of done

1. `cd agentic && pytest` → 52 + ~22 new, green, no network/install/GPU/key.
2. **DoD demo (live, this machine):**
   ```bash
   python -m adaptrna_agentic.cli.toolhub register-external adaptrna_agentic.toolhub.external.vienna
   #   → prints: package 'ViennaRNA' (import 'RNA') is not installed;
   #     would run: .../python -m pip install ViennaRNA   — Proceed? [y/N]
   #   (non-interactive first run declines → exit 1 with instructions; rerun --yes installs)
   python -m adaptrna_agentic.cli.toolhub list                 # adapter + vienna_* side by side
   python -m adaptrna_agentic.cli.toolhub test vienna_fold     # ← the DoD: golden tests pass
   python -m adaptrna_agentic.cli.toolhub call vienna_fold sequence=GGGGAAAACCCC
   python -m adaptrna_agentic.cli.toolhub deactivate vienna_fold   # call now refuses
   python -m adaptrna_agentic.cli.toolhub activate vienna_fold     # ← enable/disable DoD
   ```
3. `test splice_site` still passes (adapter path undisturbed by the dispatch change).
4. `git status`: agentic additions + plan/master-plan edits only; no engine changes;
   ViennaRNA appears in the venv but in no `pyproject.toml` (it is a *tool* dependency,
   installed through the ToolHub flow — exactly the model for user-introduced packages).

## 8. Risks and notes

- **Wheel availability**: ViennaRNA 2.7.2 publishes manylinux wheels; if resolution fails,
  pin 2.6.4 or document the conda fallback — the contract records whatever version landed.
- **Golden drift across versions**: goldens are tied to the installed version recorded in
  provenance; a future upgrade re-runs the capture step. Energies use ±0.5 kcal/mol
  tolerance; structures compare exactly.
- **Determinism**: ViennaRNA MFE folding is deterministic — right for golden-pair testing
  (contrast with the adapter smoke tests, which check form/range).
- **Install surface**: pip only, current venv only, spec-named package only, command shown
  before running. Anything fancier (extras, version pins from user input) waits for
  Phase 6's flow with the Verifier in the loop.
