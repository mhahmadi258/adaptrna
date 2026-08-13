# Inference and Tool Management (flows A and B)

Asking a question and getting a prediction, and the lifecycle operations around the tools
that answer it.

---

## Flow A — inference

### From chat

```bash
python -m adaptrna_agentic.cli.chat --session work
```

```
you> What tools are available?
  → list_tools({})
    = [{"name": "splice_site", "type": "adapter", "state": "active", …}, …]

you> Is TTTTATAAGCGGTCAGAAACT…AAATGAANN a donor splice site?
  → splice_site({sequences: ["TTTTATAAGC…"]})
    = [0.9998]
The model gives a probability of 0.9998 that this window contains a donor splice site.
```

### From the CLI, no API key needed

```bash
toolhub="python -m adaptrna_agentic.cli.toolhub"

$toolhub list
$toolhub predict splice_site --sequences ACGUACGU... 
$toolhub predict splice_site --input windows.txt --output preds.json
$toolhub call vienna_fold sequence=GGGGAAAACCCC
```

`--input` reads one sequence per line, skipping `>` and `#` lines, so a FASTA file works
as-is.

### Over HTTP

```bash
curl -s localhost:8000/api/tools | jq
curl -s -X POST localhost:8000/api/tools/splice_site/predict \
     -H 'content-type: application/json' -d '{"sequences": ["ACGU..."]}'
curl -s -X POST localhost:8000/api/tools/vienna_fold/call \
     -H 'content-type: application/json' -d '{"args": {"sequence": "GGGGAAAACCCC"}}'
```

Adapter tools use `/predict`; external tools use `/call`. Getting it the wrong way round
returns a 409 pointing at the other endpoint.

### What happens underneath

```mermaid
sequenceDiagram
    participant U as You
    participant O as Orchestrator
    participant TF as tool_factory
    participant RT as AdapterRuntime
    participant H as RiNALMoHub

    U->>O: question with a sequence
    O->>TF: splice_site(sequences=[...])
    TF->>TF: _check_active — disabled? ToolException with the fix
    TF->>RT: predict("splice_site", seqs)
    RT->>RT: inference_lock
    RT->>RT: entry checks: type · state · artifact exists
    opt first foundation-model call in this process
        RT->>RT: discovery.load_all() then load the backbone (~2.6 GB)
    end
    opt adapter not resident
        RT->>H: register(artifact, name=tool)
    end
    RT->>H: predict(name, seqs, batch_size=serving policy)
    H->>H: activate(name) across every tuner layer
    H-->>RT: task-native output
    RT-->>O: [0.9998]
    O-->>U: an answer in the task's own terms
```

Details: [../modules/toolhub.md](../modules/toolhub.md),
[../architecture.md §5](../architecture.md#5-inference-data-flow).

### Output types per task

| Task | `predict` returns |
|---|---|
| `splice_site` | One probability in `[0, 1]` per sequence |
| `mrl` | One mean ribosome load per sequence, **on the original scale**, clamped at 0 |
| `sec_struct` | One base-pairing matrix per sequence (`L × L`) — large; served one at a time |
| `splice_simple` (generated) | `{"probability": float, "label": 0|1}` per sequence |
| `vienna_fold` / `vienna_cofold` | `{"structure": dot-bracket, "mfe": kcal/mol}` |

### First call is slow

The backbone is loaded **lazily**, on the first foundation-model call in a process. To pay
that cost up front:

```bash
python -m adaptrna_agentic.cli.chat --warmup
python -m adaptrna_agentic.cli.serve --warmup
python -m adaptrna_agentic.cli.toolhub warmup      # per-invocation; mostly diagnostic
```

`warmup` reports problems rather than raising — one tool with a missing artifact must not
stop a chat from starting. That tool still fails, with the same message, when it is used.

---

## Flow B — tool management

### The lifecycle

```mermaid
stateDiagram-v2
    [*] --> active: register / register-external
    active --> disabled: deactivate
    disabled --> active: activate
    active --> [*]: remove
    disabled --> [*]: remove
```

All six operations are available three ways — CLI, chat, HTTP — over the same manifest.

| Operation | CLI | In chat | HTTP |
|---|---|---|---|
| list | `toolhub list` | *"what tools are available?"* | `GET /api/tools` |
| inspect | `toolhub info <n>` | *"tell me about X"* | `GET /api/tools/{n}` |
| activate | `toolhub activate <n>` | *"enable X"* — **asks; needs your approval** | `POST /api/tools/{n}/activate` |
| deactivate | `toolhub deactivate <n>` | *"disable X"* — **asks; needs your approval** | `POST /api/tools/{n}/deactivate` |
| test | `toolhub test <n>` | *"test X"* | `POST /api/tools/{n}/test` |
| remove | `toolhub remove <n>` | — *(not an agent tool)* | — *(no delete surface)* |

**Removal is deliberately human-only**: it is not bound as an agent tool and has no HTTP
endpoint. Deletion stays a CLI action.

### Deactivation is routing-level

A disabled tool stays registered and its weights stay resident — peft cannot cleanly uninject
an adapter, and a resident adapter costs megabytes. What changes is that every call path
refuses it:

```
Tool 'vienna_fold' is disabled. Enable it with `toolhub activate vienna_fold`.
```

In chat the model receives that same string as a tool result and can act on it — which is
why disabled tools are still *bound*, with a `(currently DISABLED — only the user can enable
it …)` note in their description. Being bound is what lets the model **mention** a disabled
tool accurately. It is not permission to switch it on.

### The switch is yours (Phase 10)

`activate_tool` and `deactivate_tool` are approval-gated. The model can put the request to
you; only your answer changes anything:

```
you> Disable the fold tool, then fold GGGGAAAACCCC anyway.
  → deactivate_tool({name: "vienna_fold"})   ⏸ approval: Disable the tool 'vienna_fold' (currently active)
you> [approve]                               = 'vienna_fold' is now disabled
  → vienna_fold({sequence: "GGGGAAAACCCC"})  = Tool 'vienna_fold' is disabled. Only the user can enable it…
ai> It is off now, so I cannot fold that. Want me to ask you to turn it back on?
you> yes
  → activate_tool({name: "vienna_fold"})     ⏸ approval: Enable the tool 'vienna_fold' (currently disabled)
you> [decline]                               = The user declined. Do not retry.
```

Decline and the manifest is untouched — `toolhub_data/tools.json` still reads
`"state": "disabled"`. Approve and the tool flips and the fold runs
(`{"structure": "((((....))))", "mfe": -5.4}`).

Until Phase 10 this flow ran without stopping: the model hit the refusal, called
`activate_tool` itself, and folded the sequence — a documented feature that turned out to
make the switch meaningless in practice. See
[PHASE_10 §1](../../plans/PHASE_10_SESSION_RAIL_AND_TOOL_GATE.md) for the reversal, and
[agents.md §4](../modules/agents.md#4-the-approval-gate) for why this is gated for authority
rather than for cost.

For a full cleanup, `AdapterRuntime.rebuild()` (or `toolhub rebuild`) drops the resident hub.

### Testing a tool

```bash
$toolhub test splice_site      # adapter → smoke test; exit code 1 if not ok
$toolhub test vienna_fold      # external → golden cases
```

| Kind | What runs |
|---|---|
| Adapter | The stored `test.sequences` through the tool, then: one output per sequence, a per-task output-form validator (probabilities in `[0,1]`, non-negative MRL, square `L×L` matrices), and an optional exact comparison against `test.expected` within `test.tolerance` |
| External | Every golden case from the manifest entry, comparing exactly or within `{"approx", "tol"}` |

Reports share one shape — `{name, ok, checks: [...], outputs}` — so the CLI, the chat and the
browser render the same thing.

### Registering an adapter by hand

```bash
$toolhub register outputs/my_run/splice_site_adapter.pt \
    --name donor \
    --description "Donor splice-site probability (Spliceator, 400 nt windows)" \
    --test-input windows.txt
```

| Flag | Effect |
|---|---|
| `--name` | Defaults to the adapter's task name |
| `--description` | Shown to the user **and to the model** — worth writing carefully |
| `--batch-size` | Serving policy; `mrl` is forced to 1 automatically |
| `--test-sequences` / `--test-input` | What `toolhub test` will run |
| `--link` | Reference the file in place instead of copying into `toolhub_data/adapters/` |

Refused, with the reason up front: a duplicate name, a **full fine-tuning export** (only its
head travels in the file), or an adapter trained on a different backbone size.

Registering the *result of a training run* is normally done through the chat instead, behind
an approval gate — see [finetuning.md](finetuning.md#step-6--register).

### Where tool state lives

`toolhub_data/tools.json` — human-diffable JSON, atomic writes, an in-file revision counter.
Schema: [../configuration.md §5](../configuration.md#5-toolhub_datatoolsjson--the-manifest).

Two chat processes are fine; if both write, the second gets:

```
'…/tools.json' changed on disk since it was read — another process registered or changed a
tool. Nothing was written; retry the operation so it applies on top of their change.
```

Over HTTP that is a `409` with `retryable: true`. Retry; nothing was lost.
