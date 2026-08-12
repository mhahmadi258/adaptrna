"""Chat-model construction — the single provider-aware seam of the agentic layer.

LangChain's `init_chat_model` resolves provider-prefixed specs ("anthropic:claude-opus-5"),
so no other module in this package imports a provider integration — not even
`langchain_anthropic` directly. Swapping providers later is a `Settings` edit.
"""

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from adaptrna_agentic.settings import Settings, require_api_key


def build_chat_model(
    role: str, settings: Optional[Settings] = None, **overrides
) -> BaseChatModel:
    """The chat model configured for `role` ("orchestrator" | "toolsmith" | "verifier")."""
    settings = settings or Settings.from_env()
    spec = settings.model_for(role)

    # Checked here -- at construction, not at import -- so deterministic code paths and
    # the test suite never need a credential.
    if spec.startswith("anthropic"):
        require_api_key()

    kwargs = {"max_tokens": settings.max_tokens, **overrides}
    return init_chat_model(spec, **kwargs)
