"""Tool listing, lifecycle and invocation.

Handlers are `def`, not `async def`: Starlette runs them in its threadpool, so a
prediction that occupies the GPU for a second does not freeze the event loop — and
`/health` keeps answering.
"""

from dataclasses import asdict

from fastapi import APIRouter, Depends

from adaptrna_agentic.api.app import get_services
from adaptrna_agentic.api.deps import Services
from adaptrna_agentic.api.schemas import CallRequest, PredictRequest
from adaptrna_agentic.toolhub.errors import ToolHubError

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
def list_tools(services: Services = Depends(get_services)) -> list:
    return [
        {"name": e.name, "type": e.type, "state": e.state, "task": e.task,
         "description": e.description}
        for e in services.registry.list()
    ]


@router.get("/{name}")
def tool_info(name: str, services: Services = Depends(get_services)) -> dict:
    return asdict(services.registry.get(name))


@router.post("/{name}/activate")
def activate(name: str, services: Services = Depends(get_services)) -> dict:
    entry = services.registry.activate(name)
    return {"name": entry.name, "state": entry.state}


@router.post("/{name}/deactivate")
def deactivate(name: str, services: Services = Depends(get_services)) -> dict:
    entry = services.registry.deactivate(name)
    return {"name": entry.name, "state": entry.state}


@router.post("/{name}/test")
def test_tool(name: str, services: Services = Depends(get_services)) -> dict:
    entry = services.registry.get(name)

    if entry.type == "external":
        from adaptrna_agentic.toolhub.external.contract import run_golden

        return run_golden(entry)

    return services.runtime.smoke_test(name)


@router.post("/{name}/predict")
def predict(
    name: str, body: PredictRequest, services: Services = Depends(get_services)
) -> dict:
    """Adapter tools. Serialised inside the runtime — see AdapterRuntime.inference_lock."""
    entry = services.registry.get(name)
    if entry.type != "adapter":
        raise ToolHubError(
            f"'{name}' is an {entry.type} tool; POST to /api/tools/{name}/call instead."
        )

    outputs = services.runtime.predict(name, body.sequences, batch_size=body.batch_size)

    return {"tool": name, "sequences": body.sequences, "predictions": _jsonable(outputs)}


@router.post("/{name}/call")
def call_tool(
    name: str, body: CallRequest, services: Services = Depends(get_services)
) -> dict:
    """External tools (ViennaRNA and friends)."""
    entry = services.registry.get(name)
    if entry.type != "external":
        raise ToolHubError(
            f"'{name}' is an {entry.type} tool; POST to /api/tools/{name}/predict instead."
        )
    if not entry.active:
        raise ToolHubError(
            f"Tool '{name}' is disabled. Enable it with POST /api/tools/{name}/activate."
        )

    from adaptrna_agentic.toolhub.external.contract import call_entry

    return {"tool": name, "result": call_entry(entry, dict(body.args))}


def _jsonable(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()

    return value
