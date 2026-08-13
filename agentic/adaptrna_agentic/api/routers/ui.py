"""Serving the browser client.

No build step: the assets are plain ES modules the browser loads directly, so the UI ships
with `pip install -e ./agentic` and adds no second toolchain to a Python repo. See
plans/PHASE_9_WEB_UI.md §2 for why that beat React here — and for the tripwire that says
when to change course.

The client is a *renderer of server state*: conversations live in the checkpointer, tools
in the manifest, jobs in the job store. It holds no domain state of its own, which is what
keeps it this small.
"""

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles

from adaptrna_agentic.settings import REPO_ROOT

#: Repo-root `ui/`, per the master plan's three-directory split.
UI_DIR = REPO_ROOT / "ui"

STATIC_PREFIX = "/ui"

MISSING = """<!doctype html>
<title>AdaptRNA</title>
<body style="font-family: system-ui; padding: 3rem; line-height: 1.6">
<h1>UI assets not found</h1>
<p>Expected them in <code>{path}</code>.</p>
<p>The API itself is fine — see <a href="/docs">/docs</a>. This happens when the package
is installed away from its repository checkout; the browser client is served from the
repo's <code>ui/</code> directory.</p>
"""

router = APIRouter(tags=["ui"])


@router.get("/", include_in_schema=False)
def index():
    """The app shell. Everything else it needs comes from `/ui/*` and `/api/*`."""
    page = UI_DIR / "index.html"
    if not page.exists():
        return HTMLResponse(MISSING.format(path=UI_DIR), status_code=503)

    # The shell must never be cached: it names the asset files, and a stale copy would
    # reference modules that no longer exist after an edit.
    return FileResponse(page, media_type="text/html",
                        headers={"Cache-Control": "no-store"})


def install(app: FastAPI) -> None:
    """Mount the static assets and the shell.

    Mounted under `/ui` rather than `/`, which would shadow `/api` and `/health`.
    """
    app.include_router(router)

    if UI_DIR.is_dir():
        app.mount(STATIC_PREFIX, StaticFiles(directory=UI_DIR), name="ui")
