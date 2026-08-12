"""Process-wide singletons for the HTTP service.

This is the whole concurrency story in one file. The service holds exactly one
`AdapterRuntime` — the backbone is 2.6 GB and belongs in memory once — one SQLite
checkpointer shared with the terminal chat, and one compiled graph.

Inference safety lives in `AdapterRuntime` itself (its `inference_lock`), not here, so
that every caller is covered rather than each entry point having to remember.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os
import sqlite3

from adaptrna_agentic.settings import REPO_ROOT
from adaptrna_agentic.toolhub.registry import Registry
from adaptrna_agentic.toolhub.runtime import AdapterRuntime

TOKEN_VAR = "ADAPTRNA_API_TOKEN"

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass
class Services:
    """Everything a request handler may need, built once per process."""

    registry: Registry
    runtime: AdapterRuntime
    graph: Any
    checkpointer: Any
    db_path: Path
    token: Optional[str] = None

    @property
    def inference_lock(self):
        return self.runtime.inference_lock


def open_checkpointer(db_path: Optional[Path] = None):
    """A SqliteSaver on the same file the terminal chat uses.

    WAL and a busy timeout are set explicitly rather than assumed: a fresh database
    starts in rollback-journal mode, where a writer blocks every reader — and two
    processes sharing sessions is the entire point of this service.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    from adaptrna_agentic.cli.chat import chat_db_path

    path = Path(db_path) if db_path is not None else chat_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")

    return SqliteSaver(connection), path


def build_services(
    *,
    data_dir=None,
    db_path: Optional[Path] = None,
    model=None,
    token: Optional[str] = None,
) -> Services:
    """Construct the singletons. `model` is injectable so tests need no API key."""
    from adaptrna_agentic.agents.orchestrator import build_orchestrator_graph

    registry = Registry(data_dir)
    runtime = AdapterRuntime(registry)
    checkpointer, path = open_checkpointer(db_path)

    graph = build_orchestrator_graph(
        model=model, registry=registry, runtime=runtime, checkpointer=checkpointer
    )

    return Services(
        registry=registry, runtime=runtime, graph=graph, checkpointer=checkpointer,
        db_path=path, token=token if token is not None else os.environ.get(TOKEN_VAR),
    )


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS
