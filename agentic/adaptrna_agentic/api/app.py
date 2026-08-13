"""The AdaptRNA HTTP service.

Every endpoint wraps something the CLI already does — no new capability, no second
implementation to drift. What it adds is a second *front end*, sharing sessions with the
terminal through the same checkpointer, which is what Phase 9's browser UI talks to (it is
served from `/`, and is a pure client of the endpoints below).

Deliberately absent: any delete surface. Phase 7 kept deletion a human action at the CLI,
and an HTTP endpoint is a weaker boundary than a shell prompt.
"""

from typing import Optional
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from adaptrna_agentic.api.deps import Services, build_services
from adaptrna_agentic.toolhub.errors import ConcurrentModificationError, ToolHubError

API_PREFIX = "/api"


def create_app(services: Optional[Services] = None, **service_kwargs) -> FastAPI:
    """Build the app. `services` is injectable so tests need no API key and no GPU."""
    app = FastAPI(
        title="AdaptRNA",
        description="Conversational RNA analysis: adapters, external tools, fine-tuning.",
        version="0.1.0",
    )

    app.state.services = services or build_services(**service_kwargs)

    _install_error_handlers(app)
    _install_auth(app)

    from adaptrna_agentic.api.routers import jobs, sessions, system, tools, ui

    app.include_router(system.router)
    app.include_router(tools.router, prefix=API_PREFIX)
    app.include_router(jobs.router, prefix=API_PREFIX)
    app.include_router(sessions.router, prefix=API_PREFIX)
    ui.install(app)

    return app


def get_services(request: Request) -> Services:
    return request.app.state.services


# ---------------------------------------------------------------------- errors

def _install_error_handlers(app: FastAPI) -> None:
    """The CLI's messages, mapped onto status codes.

    Phase 7 standardised those messages precisely so a second front end would not need
    its own wording — the body a browser shows is the text the terminal prints.
    """

    @app.exception_handler(ConcurrentModificationError)
    def _conflict(_request: Request, exc: ConcurrentModificationError):
        return JSONResponse(status_code=409,
                            content={"error": str(exc), "retryable": True})

    @app.exception_handler(ToolHubError)
    def _refused(_request: Request, exc: ToolHubError):
        return JSONResponse(status_code=409,
                            content={"error": str(exc), "type": "ToolHubError"})

    @app.exception_handler(KeyError)
    def _not_found(_request: Request, exc: KeyError):
        message = exc.args[0] if exc.args else str(exc)
        return JSONResponse(status_code=404, content={"error": str(message)})

    @app.exception_handler(FileNotFoundError)
    def _bad_request(_request: Request, exc: FileNotFoundError):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(ValueError)
    def _invalid(_request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(Exception)
    def _internal(_request: Request, exc: Exception):
        # Never leak a traceback over HTTP; the id ties the response to the server log.
        request_id = uuid.uuid4().hex[:12]
        print(f"[{request_id}] unhandled {type(exc).__name__}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "internal error", "request_id": request_id},
        )


# ---------------------------------------------------------------------- auth

def _install_auth(app: FastAPI) -> None:
    """A bearer token, required only when one is configured.

    This API can spend GPU hours and write code into the repository, so the safe default
    is loopback-only with no token; binding anywhere else *requires* one (enforced where
    the server is started, in `cli/serve.py`).
    """

    @app.middleware("http")
    async def _check_token(request: Request, call_next):
        token = request.app.state.services.token
        if token and request.url.path != "/health":
            supplied = request.headers.get("authorization", "")
            if supplied != f"Bearer {token}":
                return JSONResponse(status_code=401,
                                    content={"error": "missing or invalid bearer token"})

        return await call_next(request)
