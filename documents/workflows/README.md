# Workflows

End-to-end procedures, step by step, naming the modules involved at each step.

| Document | Flow | Start here when you want to… |
|---|---|---|
| [inference-and-tools.md](inference-and-tools.md) | A + B | Ask a question and get a prediction; list, enable, disable or test tools |
| [finetuning.md](finetuning.md) | C | Turn your data into a new servable adapter tool |
| [new-task-codegen.md](new-task-codegen.md) | D | Get a task written for data no shipped task can read |
| [external-tools.md](external-tools.md) | E | Wrap a classical bioinformatics package as a tool |
| [operations.md](operations.md) | — | Diagnose a broken install, reclaim disk, recover from a failure |

Setup and installation are in [../setup.md](../setup.md). The engine's own training,
evaluation and prediction commands — used directly, outside the agent platform — are
documented in [`../../engine/README.md`](../../engine/README.md) and summarised in
[../modules/engine-hub.md](../modules/engine-hub.md#8-cli).

---

## The shape shared by every creation flow

```mermaid
flowchart LR
    D["describe<br/><i>profile / inspect</i>"] --> P["propose<br/><i>from validated knowledge</i>"]
    P --> G1{{"APPROVAL"}}
    G1 --> E["execute<br/><i>GPU run · codegen · install</i>"]
    E --> J["judge<br/><i>analyze / verify / golden tests</i>"]
    J --> G2{{"APPROVAL"}}
    G2 --> R["register / land"]
```

Two invariants hold across all of them:

1. **Approval in one step never implies approval for the next.** Starting a training run and
   registering its result are two separate gates, and a decline at either is final for that
   turn — the orchestrator is instructed not to retry the same action.
2. **What you are shown is what will happen.** The gate renders the exact command, the exact
   file list, or the exact diff — not a paraphrase. `tests/test_ui_contract.py` and the
   browser suite verify the browser's rendering against the terminal's own renderer.

## Choosing a flow

```mermaid
flowchart TD
    Q["What do you want?"] --> A{"A prediction?"}
    A -->|"yes"| A1["Flow A — inference-and-tools.md"]
    A -->|"no"| B{"A tool that does not exist yet?"}
    B -->|"a classical package"| E1["Flow E — external-tools.md"]
    B -->|"a model capability"| C{"Can a shipped task read your data?"}
    C -->|"profile_dataset says yes<br/>(layout_match is set)"| C1["Flow C — finetuning.md"]
    C -->|"no"| D1["Flow D — new-task-codegen.md<br/>then Flow C"]
```

`profile_dataset` answers the branch at the bottom for you: its `layout_match` field is
either a task name (flow C) or `null` with a `layout_reason` explaining what the nearest
shape expects (flow D).
