# `rinalmo/` — the vendored backbone

`engine/rinalmo/`

The RiNALMo RNA language model as vendored into this repository: the transformer itself, its
tokeniser, the prediction heads the shipped tasks use, the benchmark datamodules, and a few
utilities. **Treat this package as third-party code.** The framework in
[`rinalmo_hub/`](engine-hub.md) wraps it; the agentic layer never touches it directly except
through `Alphabet` and `model_config`.

---

## Contents

1. [`config.py` — model sizes](#1-configpy--model-sizes)
2. [`model/` — the network](#2-model--the-network)
3. [`data/` — tokenisation and benchmarks](#3-data--tokenisation-and-benchmarks)
4. [`utils/`](#4-utils)
5. [`pretrained.py` and `resources/`](#5-pretrainedpy-and-resources)
6. [What the rest of the project actually uses](#6-what-the-rest-of-the-project-actually-uses)

---

## 1. `config.py` — model sizes

```python
model_config(name) -> ml_collections.ConfigDict
```

Four sizes, differing only in width and depth:

| `lm_config` | `embed_dim` | `num_blocks` | `num_heads` | Used for |
|---|---:|---:|---:|---|
| `nano` | 320 | 6 | 20 | **Every CPU test and the verification harness** — random init, no weights, instant |
| `micro` | 480 | 12 | 20 | — |
| `mega` | 640 | 30 | 20 | — |
| `giga` | 1280 | 33 | 20 | The real model; what every shipped config assumes |

The returned `ConfigDict` has four sections — `globals`, `alphabet`, `training`, `model` —
and `model` carries `embedding`, `token_dropout`, `transformer` and `lm_mask_head` blocks.
Defaults worth knowing: `use_rot_emb: True`, `attn_qkv_bias: False`, `use_flash_attn: True`,
`attention_dropout: 0.1`, `residual_dropout: 0.1`, `transition_factor: 4`.

`any_tokenizer_discrepancies()` asserts at construction that the alphabet and the
`globals.*_tkn_idx` fields agree — a cheap guard against a config that would tokenise
inconsistently with the pretrained weights.

## 2. `model/` — the network

### `model.py` — `RiNALMo`

```python
forward(tokens, need_attn_weights=False) -> {"logits", "representation", "attentions"?}
```

```
tokens → embedding → token_dropout → transformer → representation
                                                 └→ lm_mask_head → logits
```

Downstream tasks consume `["representation"]` only. **The masked-LM head always runs**, and
its output never reaches a downstream loss — which is why the engine configures DDP with
`find_unused_parameters=True` and why the codegen harness excludes `lm_mask_head` from its
"every trainable tensor got a gradient" check.

### `modules.py`

| Class | Role |
|---|---|
| `TokenDropout` | Scales embeddings to compensate for masked tokens. **The module behind the fp32-only serving constraint**: it holds an fp32 scalar that promotes activations to fp32, which then meet bf16 layer-norm weights when the whole model is cast for non-autocast half-precision inference. |
| `Transformer` | The block stack plus a final layer norm; returns `(representation, attn_weights)` |
| `TransformerBlock` | Attention + SwiGLU transition, with residual dropout |
| `SwiGLU` | The transition non-linearity, with a learnable beta |
| `MaskedLanguageModelHead` | Pretraining head, always evaluated |

### `attention.py`

Two paths behind one interface:

* `FlashMultiHeadSelfAttention` / `FlashAttention` — the fast path, using
  `flash_attn_qkvpacked_func` (or the `varlen` variant with `unpad_input`/`pad_input` when a
  padding mask is present) and flash-attn's own `RotaryEmbedding`. Uses a **single fused
  `Wqkv` projection**, which is why LoRA's default target list has one entry covering q, k
  and v jointly.
* `MultiHeadSelfAttention` / `dot_product_attention` — the plain-PyTorch fallback, with
  `apply_reference_rotary` matching flash-attn's `interleaved=False` convention so the two
  paths agree numerically.

The flash imports are guarded at module level, so the package imports and the CPU test suite
runs without `flash_attn` installed.

`rope.py` holds the rotary position embedding used by the fallback path.

### `downstream.py` — the prediction heads

| Head | Used by | Shape |
|---|---|---|
| `SpliceSitePredictionHead(c_in, embed_dim)` | `splice_site` | CLS embedding → one logit |
| `RibosomeLoadingPredictionHead(c_in, embed_dim, num_blocks, dropout)` | `mrl` | 1D ResNet over the sequence, masked pooling → one scalar |
| `SecStructPredictionHead(embed_dim, num_blocks, conv_dim, kernel_size)` | `sec_struct` | `_outer_concat` → 2D ResNet → L×L logits |
| `ncRNAClassificationHead(c_in, embed_dim, n_classes)` | the `examples/` task | CLS → class logits |
| `RibonanzaPredictionHead` | — | Present but unused by any shipped task |
| `ResNet1D` / `ResNet2D` / `ResNet1DBlock` / `ResNet2DBlock` | building blocks | |

A generated task is free to define its own head — `adaptrna_custom/tasks/splice_simple`
defines `SpliceSimpleHead` inline rather than reusing one of these.

## 3. `data/` — tokenisation and benchmarks

### `alphabet.py` + `constants.py`

```python
Alphabet(standard_tkns=RNA_TOKENS, special_tkns=[CLS, PAD, EOS, UNK, MASK])
    len(alphabet); get_tkn(idx); get_idx(tkn)
    encode(seq, pad_to_len=-1) -> [idx]        # wraps with CLS … EOS
    batch_tokenize(seqs) -> [[idx]]            # pads to the longest
```

Every datamodule and every prediction path tokenises through this. `batch_tokenize` is what
`RiNALMoHub.tokenize` and the harness both call.

### `data/downstream/`

Three benchmark packages, each `dataset.py` + `datamodule.py`:

| Package | Dataset | Notes |
|---|---|---|
| `splice_site_prediction/` | Spliceator, via the SpliceBERT release | `<root>/GS_1/db_N/{Train,Val}_{donor,acceptor}_400.csv`, `;`-separated and **headerless**; a separate `test_root` holds the benchmark species (`Danio`, `Fly`, `Thaliana`, `Worm`). This is a **cross-species** benchmark. |
| `ribosome_loading/` | Synthetic human 5'UTR library | One gzipped CSV, `GSM4084997_varying_length_25to100.csv.gz`; `train_eval_split` reserves the top 100 sequences per length for `Random7600` |
| `secondary_structure/` | bpRNA (SPOT-RNA split) or ArchiveII family folds | `.ct`/`.bpseq` structure files; **one structure per batch** because the head is O(L²) |

Each datamodule downloads its own data when `prepare_data` is enabled
(`--prepare_data` sets `data.prepare`).

## 4. `utils/`

| Module | Purpose |
|---|---|
| `finetune_callback.py` | `GradualUnfreezing(BaseFinetuning)` — reads an `ft_schedules/*.yaml` mapping epoch → list of module-name regexes, merges each epoch's regexes into one pattern, and unfreezes matching modules at that epoch. Skips children of an already-unfrozen parent, so `backbone.*` unfreezes the whole model in one go. `initial_denom_lr` defaults to **1.0** here, against Lightning's 10.0, which had silently trained the backbone at 1e-6 instead of the requested 1e-5 for an entire MRL run. |
| `scaler.py` | `StandardScaler(nn.Module)` with `_mean`/`_std` as **buffers** — which is what lets them travel inside an adapter file via `ADAPTER_EXTRA_PREFIXES = ("scaler.",)`. `partial_fit`, `transform`, `inverse_transform`. |
| `sec_struct.py` | `prob_mat_to_sec_struct` (probabilities → a valid structure), `ss_precision`/`ss_recall`/`ss_f1`, `save_to_ct` |
| `splice_site_metrics.py` | `accuracy`, `precision`, `recall`, `specificity`, `f1_score` — returning **percentages rounded to two decimals**, matching the numbers reported in the source project |
| `download.py` | Google Drive / HTTP acquisition for weights and datasets |

## 5. `pretrained.py` and `resources/`

```python
from rinalmo.pretrained import get_pretrained_model
get_pretrained_model("giga-v1")      # caches into ~/.cache/rinalmo_pretrained/
```

`resources/model2gdisk.json` maps model names to Google Drive ids; `resources/remote_data.json`
does the same for datasets. Both are shipped as package data (the engine's `pyproject.toml`
declares `"rinalmo" = ["resources/*.json"]`).

## 6. What the rest of the project actually uses

Everything outside `engine/` touches this package through a very small surface:

| Import | Used by |
|---|---|
| `rinalmo.data.alphabet.Alphabet` | every task's `build_datamodule`, `RiNALMoHub.tokenize`, the harness's `_predict` |
| `rinalmo.config.model_config` | `BaseDownstreamModule.__init__`, `RiNALMoHub.__init__` |
| `rinalmo.model.model.RiNALMo` | `BaseDownstreamModule.__init__`, `RiNALMoHub.__init__` |
| `rinalmo.model.downstream.*` | the three shipped tasks and the example |
| `rinalmo.utils.scaler.StandardScaler` | the `mrl` task |
| `rinalmo.utils.sec_struct.*`, `rinalmo.utils.splice_site_metrics` | their respective tasks |
| `rinalmo.utils.finetune_callback.GradualUnfreezing` | `cli/common.build_trainer` |
| `rinalmo.data.downstream.*` | the three shipped tasks' datamodules |

A **generated** task normally needs only `Alphabet` — `adaptrna_custom/tasks/splice_simple`
imports nothing else from this package, and defines its own head.

If you are adding a task, you should not need to modify anything here. If you find yourself
wanting to, that is a signal the abstraction in [`rinalmo_hub/`](engine-hub.md) is wrong —
which is exactly what `engine/tests/test_new_task_acceptance.py` asserts by checking that no
core file so much as mentions the example task's name.
