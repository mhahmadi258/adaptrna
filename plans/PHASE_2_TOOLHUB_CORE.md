# Phase 2 — ToolHub core, no LLM (detailed plan)

> Parent: [MASTER_PLAN.md](MASTER_PLAN.md) §8, Phase 2; design contract in §4.
> **Definition of done:** the existing splice-site adapter
> (`outputs/splice_donor_lora/splice_site_adapter.pt`) registered and predicting from the
> management CLI; nano tests green.
> Status: planned · not started

---

## 1. Context and goal

Phase 2 builds the deterministic heart of the platform: the **manifest registry** (what
tools exist, in what state) and the **adapter runtime** (one shared backbone serving every
active adapter via `RiNALMoHub`), exposed through a management CLI. No LLM anywhere — this
layer must behave identically every time, and Phase 4 will simply wrap its operations as
agent tools, so every operation returns plain data (dicts/lists/strings), never prints as
its only output.

**Out of scope:** ExternalTool + ViennaRNA (Phase 3), agent integration (Phase 4),
training/jobs (Phase 5), any engine code change (forbidden by MASTER_PLAN §3.4 — the
engine is consumed through its public API only).

Verified at plan time (2026-08-12): `giga-v1.pt` at `~/.cache/rinalmo_pretrained/`, an
H200 GPU, flash-attn 2.3.2 in the venv, and real 400 nt donor test sequences at
`~/bio2/RiNALMo/dataset/test_data/Danio/` — so the DoD demo is executable on this machine.

## 2. Decisions this plan fixes (from MASTER_PLAN §10)

| Decision | Choice | Rationale |
|---|---|---|
| **Manifest store** | **JSON file** (`toolhub_data/tools.json`), atomic writes (tmp + `os.replace`), `format_version: 1` | Single user, single process until Phase 8; human-readable and diffable; trivially testable. SQLite buys nothing at ~tens of tools; revisit at Phase 8 if concurrent writers appear. |
| **Backbone load policy** | **Lazy** — the backbone loads on the first call that needs a forward pass (`predict`/`test`); registry operations (list/register/activate/…) never pay the 2.6 GB load. Explicit `warmup` op for eager loading; `rebuild` drops the hub for full cleanup. | Management commands must be instant; Phase 4's chat startup stays fast and can call `warmup` itself. Registration validates the adapter *file* (engine `load_adapter`), which needs no backbone. |
| Artifact ownership | `register` **copies** the adapter into `toolhub_data/adapters/<name>.pt` by default (`--link` records the source path instead) | Tool lifetime decoupled from run directories; 6 MB copies are free. `remove` deletes owned copies, never linked sources. |
| Engine dependency | **Not** declared in `agentic/pyproject.toml`; both packages are installed editable into one venv, and toolhub modules import the engine **lazily inside functions** with an actionable ImportError | A local-path dependency is brittle; lazy imports keep `adaptrna_agentic` import (and the chat CLI) torch-free. |
| `toolhub_data/` | Lives at the repo root (CWD convention like `outputs/`), **git-ignored** | Runtime state, reproducible from `register` commands; contains binaries. |
| Tool naming | Defaults to the adapter's task name; `--name` allows e.g. `splice_site_donor` (engine `hub.register(path, name)` supports this) | Several adapters of one task type (donor + acceptor) must coexist. |

## 3. Data layout and manifest schema (v1)

```
toolhub_data/                  # repo root, git-ignored
├── tools.json                 # the manifest (below)
└── adapters/<name>.pt         # registry-owned artifact copies
```

```json
{
  "format_version": 1,
  "backbone": {
    "lm_config": "giga",
    "weights": "~/.cache/rinalmo_pretrained/giga-v1.pt",
    "device": "auto",                 // auto -> cuda if available, else cpu
    "dtype": "auto"                   // auto -> bfloat16 on cuda, float32 on cpu
  },
  "tools": {
    "splice_site": {
      "type": "adapter",
      "state": "active",              // active | disabled
      "description": "Donor splice-site probability for 400 nt sequences",
      "task": "splice_site",
      "lm_config": "giga",
      "artifact": "toolhub_data/adapters/splice_site.pt",
      "serving": {"batch_size": null},     // null -> the task's DEFAULT_PREDICT_BATCH_SIZE
      "test": {"sequences": ["ACGU..."], "expected": null},
      "provenance": {
        "source": "outputs/splice_donor_lora/splice_site_adapter.pt",
        "registered_at": "2026-08-12T...Z",
        "adapter_metadata": { "created": "...", "seed": 42, "arm": "lora",
                              "train_metrics": {"val/f1_score": 98.58} }
      }
    }
  }
}
```

- Relative paths are repo-root-relative; `~` is expanded. One `lm_config` per hub — a
  registration whose adapter `lm_config` differs from `backbone.lm_config` is rejected at
  register time (fail early; the engine would reject it at load time anyway).
- `serving.batch_size` is the MASTER_PLAN §7 serving policy slot. Registration fills the
  MRL caveat in automatically: for tasks whose head is pad-sensitive (`mrl`), default it
  to **1** with a note in `description`; others stay `null` (task default).
- `test.sequences` seeds the smoke test; supplied at registration (`--test-sequences` /
  `--test-input FILE`), defaulting to two generic ACGU strings.

## 4. Components

```
agentic/adaptrna_agentic/toolhub/
├── __init__.py        # re-exports: Registry, AdapterRuntime, Manifest, errors
├── manifest.py        # dataclasses + JSON I/O (no engine imports at all)
├── registry.py        # lifecycle ops (engine imported lazily, for file validation only)
└── runtime.py         # AdapterRuntime over RiNALMoHub (engine imported lazily)
agentic/adaptrna_agentic/cli/toolhub.py    # python -m adaptrna_agentic.cli.toolhub
```

### 4.1 `manifest.py`

`BackboneConfig`, `ToolEntry`, `Manifest` dataclasses mirroring §3; `Manifest.load(dir)` /
`.save()` with atomic replace; unknown `format_version` and malformed JSON produce
actionable errors. Path helpers: `data_dir` resolution order = explicit argument →
`ADAPTRNA_TOOLHUB_DIR` env → `<repo root>/toolhub_data`. Resolution of `auto`
device/dtype lives here too (single place).

### 4.2 `registry.py` — lifecycle

```python
class Registry:
    def __init__(self, data_dir: Path | None = None): ...
    def register(self, adapter_path, *, name=None, description=None, batch_size=None,
                 test_sequences=None, link=False) -> ToolEntry
    def activate(self, name) / deactivate(self, name) -> ToolEntry
    def remove(self, name, *, keep_artifact=False) -> None
    def list(self) -> list[ToolEntry]
    def get(self, name) -> ToolEntry          # KeyError names the known tools
```

`register` validates via the engine's public `rinalmo_hub.adapter.load_adapter(path)`
(format version, required fields) and then enforces, *before* anything is copied:

- **LoRA-only** (MASTER_PLAN §3.6): `payload["lora"] is None` and `metadata.arm ==
  "full_ft"` → reject with the engine's own rationale (a full-FT export carries only the
  head; serving it would pair a fine-tuned head with the pretrained backbone).
- `lm_config` must equal `backbone.lm_config`.
- Duplicate names rejected; `description` defaults to task + provenance summary;
  provenance copied from the adapter's own `metadata`.

### 4.3 `runtime.py` — `AdapterRuntime`

```python
class AdapterRuntime:
    def __init__(self, registry: Registry): ...
    @property
    def loaded(self) -> bool                  # backbone resident?
    def warmup(self) -> None                  # eager load + register all active tools
    def rebuild(self) -> None                 # drop the hub (full cleanup op)
    def predict(self, name, sequences, batch_size=None) -> list | Tensor
    def smoke_test(self, name) -> dict        # report: ok, checks, outputs summary
```

- First forward-pass call builds one `RiNALMoHub(backbone_weights, lm_config, device,
  dtype)` and `hub.register(artifact, name)`s every **active** adapter tool; tools
  activated after the build are hub-registered on demand (tracked by a resident-name
  set). A **disabled tool refuses `predict`** even while resident — deactivation is
  routing-level, exactly as MASTER_PLAN §4 specifies; `rebuild` is the full cleanup.
- Missing weights fail with an actionable message naming the manifest's `backbone.weights`
  and the engine README's download instructions.
- `smoke_test(name)`: runs `predict` on `test.sequences`; generic checks (one output per
  sequence, values finite) plus a per-task validator table — `splice_site`: probabilities
  in [0, 1]; `mrl`: floats ≥ 0; `sec_struct`: square binary matrix matching the sequence
  length — with a generic fallback for unknown tasks. If `test.expected` is set, compare
  within tolerance (forward passes are deterministic per device/dtype, but cross-device
  exactness is not promised — default tolerance 1e-4).

### 4.4 `cli/toolhub.py` — management CLI

`python -m adaptrna_agentic.cli.toolhub <cmd>`, global `--data-dir` (else env/default):

| command | behavior |
|---|---|
| `list` | table: name, type, state, task, batch policy, source |
| `register PATH [--name --description --batch-size --test-sequences … --test-input F --link]` | §4.2; prints the created entry |
| `activate NAME` / `deactivate NAME` | flip state |
| `remove NAME [--keep-artifact] [--yes]` | delete entry (+ owned copy); confirms unless `--yes` |
| `info NAME` | manifest entry + the engine's `describe_adapter` summary |
| `test NAME` | smoke test; exit code 0/1 by report |
| `predict NAME (--sequences … \| --input FILE) [--batch-size N] [--output out.json]` | JSON `{sequences, predictions}` — same shape as the engine's predict CLI |
| `config [--weights P --lm-config C --device D --dtype T]` | show / update the backbone section |
| `warmup` / `rebuild` | eager load / drop the resident hub (single-process semantics) |

### 4.5 Engine contract used (public API only, zero engine edits)

`rinalmo_hub.hub.RiNALMoHub` (`register`, `predict`, `available`), `rinalmo_hub.adapter`
(`load_adapter`, `describe_adapter`), `rinalmo_hub.registry.get_task` (for
`DEFAULT_PREDICT_BATCH_SIZE` and task-native output docs). The engine's own guards
(construction order, geometry checks, full-FT refusal) remain the authority — registry
checks only *duplicate* them earlier for better errors, never replace them.

## 5. Tests (nano backbone, CPU, no weights, no key — engine's test philosophy)

A conftest fixture builds a real **nano** adapter file the way the engine's own tests do,
via public API: `get_task("splice_site")(lm_config="nano", head_config={"head_embed_dim":
16}, lora={"r": 4, "alpha": 8, "dropout": 0.0, "layer_stride": 3})` → `apply_lora()` →
randomize trainable tensors → `save_adapter(tmp)`. A second fixture makes an `mrl` nano
adapter (scaler buffers included) and a fake full-FT export for the refusal test. All
toolhub tests point `--data-dir`/`Registry(data_dir=tmp_path)` at pytest tmp dirs; the
backbone config is `lm_config="nano", weights=None` (random init — fine for structural
checks).

| test file | asserts |
|---|---|
| `test_manifest.py` | round trip; atomic save; unknown `format_version` rejected; malformed JSON error names the file; `auto` device/dtype resolution |
| `test_registry.py` | register creates entry + copies artifact (and `--link` doesn't); name defaults to task, `--name` overrides; duplicate name rejected; lm_config mismatch rejected; **full-FT export rejected** with the pairing rationale; mrl gets `serving.batch_size == 1` by default; activate/deactivate/remove (owned copy deleted, linked source kept); unknown name errors list known tools |
| `test_runtime.py` | backbone is **not** loaded by registry ops (`runtime.loaded is False`); first `predict` loads it once and serves both registered nano tools; splice_site outputs are per-sequence probabilities in [0,1]; disabled tool refuses predict while resident; tool activated after warmup becomes servable without rebuild; `rebuild` drops residency; smoke_test report ok on the nano tool and failing (clear message) on absurd `expected` |
| `test_toolhub_cli.py` | each subcommand via `main(argv)` against a tmp data dir: list/register/activate/deactivate/info/test/predict/config round-trips; `predict --output` writes the JSON shape above; missing weights message is actionable (weights path set to a nonexistent file) |

Expected total: ~25 tests, seconds on CPU. `cd agentic && pytest` must also keep the
Phase 0 16 green; `cd engine && pytest` stays untouched at 135.

## 6. Implementation order

1. `manifest.py` + `test_manifest.py`.
2. Nano-adapter conftest fixtures (the everything-else prerequisite).
3. `registry.py` + `test_registry.py`.
4. `runtime.py` + `test_runtime.py`.
5. `cli/toolhub.py` + `test_toolhub_cli.py`; `.gitignore` entry for `toolhub_data/`.
6. Real-adapter DoD run (below); README updates (`agentic/README.md` gains a ToolHub
   section; root `README.md` example line).
7. Close-out: MASTER_PLAN §8 tick; §10 rows "Manifest store" and "Backbone load policy"
   marked decided with one-line rationales.

## 7. Verification / definition of done

1. **Deterministic:** `cd agentic && pytest` → Phase 0's 16 + ~25 new, all green, no
   weights/GPU/key. `cd engine && pytest` → 135 (untouched).
2. **DoD demo (this machine has GPU + weights):**
   ```bash
   python -m adaptrna_agentic.cli.toolhub config --weights ~/.cache/rinalmo_pretrained/giga-v1.pt --lm-config giga
   python -m adaptrna_agentic.cli.toolhub register outputs/splice_donor_lora/splice_site_adapter.pt \
       --description "Donor splice-site probability (Spliceator, 400 nt windows)"
   python -m adaptrna_agentic.cli.toolhub list
   python -m adaptrna_agentic.cli.toolhub predict splice_site --input <400nt-sequences.txt>
   python -m adaptrna_agentic.cli.toolhub test splice_site
   ```
   Sequences: take a couple of real 400 nt donor windows from
   `~/bio2/RiNALMo/dataset/test_data/Danio/SA_sequences_donor_400_Final_3.fasta`
   (positives should score near 1). Expected wall-clock: one-time backbone load
   (~10–30 s), then sub-second predictions.
3. `deactivate splice_site` → `predict` refuses with a clear message; `activate` restores.
4. `git status`: only `agentic/` additions, `.gitignore` line, plan/master-plan edits —
   `toolhub_data/` ignored, no engine changes.

## 8. Risks and notes

- **Giga memory on GPU:** one resident backbone (~5 GB bf16) — trivial for the H200; CPU
  fallback works but slowly. The runtime never loads two backbones.
- **flash-attn 2.3.2 present** → fast path on CUDA; its absence would silently use the
  reference attention path (correct, slow) — worth one line in the README.
- **Meaningfulness vs mechanics:** predictions on arbitrary short strings are mechanical
  smoke only; the splice model was trained on 400 nt windows — the DoD uses real windows
  so the number is also *plausible*, not just well-formed.
- **Single-process semantics:** `warmup`/`rebuild`/`predict` residency lives per process;
  the CLI pays the backbone load per invocation. That is acceptable for Phase 2 (the
  long-lived process arrives with Phase 4's chat and Phase 8's service, which hold one
  `AdapterRuntime` for their lifetime). Note it in the CLI help.
- **Engine coupling surface** stays the §4.5 list — if a need arises beyond it, the answer
  is a design change here, not an engine edit.
