"""Configuration for the agentic layer.

Per-role model selection and API-key handling. Model specs are provider-prefixed strings
resolved through LangChain's `init_chat_model` (e.g. "anthropic:claude-opus-5") — that
string format is the entire provider abstraction, so swapping providers is a config edit
here, never a code change elsewhere.

The API key is checked at model-construction time, never at import time, so every
deterministic code path (the test suite, the later ToolHub) stays key-free.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union
import os

from dotenv import load_dotenv

#: The three judgment roles of the platform (MASTER_PLAN §5). Everything else is a
#: deterministic service and never talks to a model.
ROLES = ("orchestrator", "toolsmith", "verifier")

DEFAULT_MODEL = "anthropic:claude-opus-5"
DEFAULT_MAX_TOKENS = 8192

#: settings.py lives at <repo>/agentic/adaptrna_agentic/, so the repo root is three up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

API_KEY_VAR = "ANTHROPIC_API_KEY"
_GLOBAL_MODEL_VAR = "ADAPTRNA_MODEL"


@dataclass(frozen=True)
class Settings:
    """Resolved agentic-layer settings. Frozen: derive variants with `dataclasses.replace`."""

    models: Dict[str, str]
    max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_env(cls, env_file: Optional[Union[str, Path]] = None) -> "Settings":
        """
        Resolve settings from the environment.

        A repo-root `.env` (or `env_file`, the test seam) is loaded first when it exists;
        real environment variables always win over it. Per-role variables
        (`ADAPTRNA_MODEL_ORCHESTRATOR` / `_TOOLSMITH` / `_VERIFIER`) win over the global
        `ADAPTRNA_MODEL`, which wins over the default.
        """
        path = Path(env_file) if env_file is not None else REPO_ROOT / ".env"
        if path.exists():
            load_dotenv(path, override=False)

        global_spec = os.environ.get(_GLOBAL_MODEL_VAR, DEFAULT_MODEL)
        models = {
            role: os.environ.get(f"{_GLOBAL_MODEL_VAR}_{role.upper()}", global_spec)
            for role in ROLES
        }

        return cls(
            models=models,
            max_tokens=int(os.environ.get("ADAPTRNA_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )

    def model_for(self, role: str) -> str:
        if role not in self.models:
            raise KeyError(
                f"Unknown agent role '{role}'. Known roles: {sorted(self.models)}"
            )
        return self.models[role]


def require_api_key() -> None:
    """Fail fast — with the fix in the message — when no Anthropic credential is set."""
    if not os.environ.get(API_KEY_VAR):
        raise RuntimeError(
            f"{API_KEY_VAR} is not set. Export it in your shell or put it in "
            f"'{REPO_ROOT / '.env'}' (that file is git-ignored)."
        )
