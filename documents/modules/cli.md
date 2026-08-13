# `cli/` — the three executables

`agentic/adaptrna_agentic/cli/`

| Module | Command | Needs an API key |
|---|---|---|
| `chat.py` | `python -m adaptrna_agentic.cli.chat` | ✅ |
| `toolhub.py` | `python -m adaptrna_agentic.cli.toolhub <subcommand>` | – |
| `serve.py` | `python -m adaptrna_agentic.cli.serve` | ✅ (for chat endpoints) |

Run every one of them **from the repository root**.

---

## `chat.py`

The terminal REPL against the orchestrator, and the reference renderer for the approval
gate.

```bash
python -m adaptrna_agentic.cli.chat                      # REPL, session 'default'
python -m adaptrna_agentic.cli.chat --session paper      # a named persistent session
python -m adaptrna_agentic.cli.chat --new-session        # timestamp-named: session-YYYYmmdd-HHMMSS
python -m adaptrna_agentic.cli.chat --once "PROMPT"      # one exchange, then exit
python -m adaptrna_agentic.cli.chat --list-sessions
python -m adaptrna_agentic.cli.chat --warmup             # load the backbone at startup
python -m adaptrna_agentic.cli.chat --model anthropic:claude-sonnet-5   # override every role
python -m adaptrna_agentic.cli.chat --data-dir /tmp/hub  # a different ToolHub state dir
```

### Startup order

1. `--list-sessions` short-circuits: read `DISTINCT thread_id` from `checkpoints` and exit.
2. `Settings.from_env()`, with `--model` replacing every role via `dataclasses.replace`.
3. **`build_chat_model("orchestrator", settings)` eagerly** — so a missing API key fails
   here, with the actionable message, rather than three turns in.
4. `Registry` + `AdapterRuntime`; `--warmup` loads the backbone and prints any skipped tools
   and the resident adapter list.
5. `SqliteSaver` on `chat_data/sessions.sqlite` (`ADAPTRNA_CHAT_DIR` overrides the
   directory), `check_same_thread=False`.
6. `build_orchestrator_graph(model, registry, runtime, checkpointer)` with
   `config = {"configurable": {"thread_id": session}}`.

One chat process holds **one** runtime: the backbone loads once, on the first foundation-model
tool call. Tool lifecycle changes made outside this process are picked up at the next turn,
because the orchestrator rebuilds its tool list on every model call.

### `run_turn(graph, config, user_text)`

History lives in the checkpointer, so **only the new message is sent**:

```python
payload = {"messages": [HumanMessage(content=user_text)]}
while True:
    for state in graph.stream(payload, config, stream_mode="values"):
        ...                                     # print tool calls and results as they land
    request = _pending_interrupt(graph, config)
    if request is None: break
    payload = Command(resume=_prompt_approval(request))
```

Progress is printed live: `  → tool_name({args})` for each call and `    = result …` for
each result, previewed to 200 characters. `seen` tracking skips the checkpointed history on
the first pass, so a resumed session does not reprint everything.

### `_prompt_approval` — the terminal gate

```
  ┌─ approval required ──────────────────────────────────────────
  │ Train splice_site (lora) — ETA ~7 min, output outputs/…
  │   would run: /…/python -m adaptrna_agentic.jobs.train_entrypoint --task splice_site …
  │   output:    outputs/splice_site_acceptor_lora_20260812_185058
  │   note:      Dataset download required (MRL is ~431 MB).
  │   write:     adaptrna_custom/tasks/splice_simple/task.py  (166 lines)
  │   staged in: toolhub_data/staging/splice_simple-ab12cd34
  │   (open it in your editor to read the code before approving)
  │   ! Quick run: capped at 200 steps. …
  └──────────────────────────────────────────────────────────────
  approve? [y/N]
```

Only `y`/`yes` approves. `EOF` or `Ctrl-C` counts as a decline (with the note *"the user
declined at the approval prompt"*), so a piped or interrupted session cannot accidentally
approve. This function is the **reference implementation** of the gate: the browser modal was
verified byte-for-byte against its output.

## `toolhub.py`

The management CLI. Fifteen subcommands, no API key, and every error printed as
`error: <message>` with exit code 1.

```bash
toolhub="python -m adaptrna_agentic.cli.toolhub"     # optionally --data-dir DIR before the subcommand
```

| Subcommand | Arguments | Effect |
|---|---|---|
| `list` | | Aligned table: NAME, STATE, TYPE, TASK, BATCH, SOURCE |
| `register` | `<adapter.pt>` `--name --description --batch-size --test-sequences --test-input --link` | Register a LoRA adapter. `--link` references the file in place instead of copying. |
| `register-external` | `<module>` `--only a,b` `--yes` | Load the wrapper's `SPEC`; if the package is missing, print `Would run: <pip command>` and prompt (or accept `--yes`); then register each function. |
| `call` | `<name>` `KEY=VALUE …` `--args '<json>'` | Invoke an external tool |
| `activate` / `deactivate` | `<name>` | Flip state |
| `remove` | `<name>` `--keep-artifact --yes` | Confirms unless `--yes` |
| `info` | `<name>` | The manifest entry as JSON, plus `describe_adapter()` for adapter tools |
| `test` | `<name>` | Smoke test (adapter) or golden test (external). **Exit code 1 if the report is not ok.** |
| `predict` | `<name>` `--sequences … --input FILE --batch-size N --output FILE` | Loads the backbone |
| `config` | `--weights --lm-config --device --dtype` | Prints the backbone config; updates it if any flag is given. `--weights null` clears it. |
| `doctor` | `--json` | The health report. **Exit code 1 on any failing check.** |
| `prune` | `staging\|sessions\|jobs\|runs\|artifacts` `--older-than DAYS --yes` | Dry run unless `--yes` |
| `warmup` | | Load the backbone + active adapters **in this process** |
| `rebuild` | | Drop the resident hub in this process |

`warmup` and `rebuild` are per-invocation and therefore mostly diagnostic — residency is per
process, and a fresh CLI invocation always starts empty. The long-lived warm runtime is what
`chat` and `serve` provide.

`--input` and `--test-input` read one sequence per line, skipping lines that start with `>`
or `#`, so a FASTA file works as-is.

There are **no job subcommands** here — job state is reachable through the chat's
`job_status` tool, `GET /api/jobs/{id}`, or `JobRunner` in Python. (The `doctor` remedy that
suggests `toolhub job-status` is wrong; see
[../README.md gap #1](../README.md#known-documentation-gaps).)

## `serve.py`

```bash
python -m adaptrna_agentic.cli.serve                        # 127.0.0.1:8000
python -m adaptrna_agentic.cli.serve --open                 # …and open the UI in a browser
python -m adaptrna_agentic.cli.serve --port 8077 --warmup   # preload the backbone
python -m adaptrna_agentic.cli.serve --reload               # development auto-reload
ADAPTRNA_API_TOKEN=secret python -m adaptrna_agentic.cli.serve --host 0.0.0.0
```

### The binding refusal

```python
check_binding(host, token) -> str | None
    None                                  if is_loopback(host) or token
    "Refusing to bind to '<host>' without a token: this API can start training runs and
     write code into the repository. Set ADAPTRNA_API_TOKEN to a secret value, or bind to
     127.0.0.1 (the default)."
```

Checked **before** uvicorn is even imported, and it returns exit code 1. That refusal is the
point: the dangerous configuration should not be reachable by accident.

### Startup output

```
AdaptRNA API on http://127.0.0.1:8000  [no token (loopback only)]
  web UI:   http://127.0.0.1:8000/
  docs:     http://127.0.0.1:8000/docs
  sessions: shared with the terminal chat at /path/to/chat_data/sessions.sqlite
```

`--warmup` loads the backbone before the server starts listening, printing any skipped tools
to stderr. `--open` schedules the browser launch on a **daemon** `threading.Timer` — `uvicorn.run`
blocks, so the browser cannot be opened after it, and a daemon thread means a failed launch
can never keep the process alive.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | `toolhub`: a `ToolHubError`, `FileNotFoundError`, `ValueError` or `KeyError` (printed as `error: …` on stderr); a failing `test` or `doctor` report; a declined `remove` · `serve`: an unsafe binding |

`chat` always returns 0; failures surface inside the conversation.
