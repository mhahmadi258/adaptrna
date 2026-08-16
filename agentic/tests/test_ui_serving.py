"""The browser client is served, and it is genuinely self-contained.

The offline check is the one with teeth: this UI is meant to run on a workstation with no
internet access, and a single CDN `<script>` would break it there while passing every
other test in this file.
"""

import re

import pytest
from fastapi.testclient import TestClient

from adaptrna_agentic.api.routers.ui import UI_DIR
from api_helpers import build_test_app

ASSETS = ["app.js", "sse.js", "api.js", "render.js", "md.js", "dom.js", "style.css"]

#: Anything that would make the browser hit another host. Written as fetch *sites*
#: (`src=`, `href=`, `url(`, `from`) rather than a bare "http" search, so the SVG favicon's
#: `xmlns='http://www.w3.org/2000/svg'` — a namespace name, never resolved — does not
#: count as a network dependency.
EXTERNAL = re.compile(
    r"""(?:src|href)\s*=\s*["'](?:https?:)?//"""
    r"""|url\(\s*["']?(?:https?:)?//"""
    r"""|(?:^|\s)(?:from|import)\s*\(?\s*["'](?:https?:)?//""",
    re.MULTILINE,
)


@pytest.fixture
def client(nano_registry, tmp_path):
    app, _ = build_test_app(nano_registry, tmp_path / "s.sqlite")
    with TestClient(app) as test_client:
        yield test_client


def test_the_shell_is_served_at_the_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>AdaptRNA</title>" in response.text


def test_the_shell_loads_the_app_as_a_module(client):
    """No bundler: the browser resolves the imports itself, which needs type=module."""
    body = client.get("/").text

    assert re.search(r'<script[^>]+type="module"[^>]+src="/ui/app\.js"', body)


def test_the_shell_is_not_cached(client):
    """It names the asset files; a stale copy would import modules that no longer exist."""
    assert "no-store" in client.get("/").headers.get("cache-control", "")


@pytest.mark.parametrize("asset", ASSETS)
def test_assets_are_served(client, asset):
    response = client.get(f"/ui/{asset}")

    assert response.status_code == 200
    assert response.content, f"{asset} is empty"

    expected = "text/css" if asset.endswith(".css") else "javascript"
    assert expected in response.headers["content-type"]


@pytest.mark.parametrize("name", ASSETS + ["index.html"])
def test_no_asset_reaches_for_another_host(name):
    match = EXTERNAL.search((UI_DIR / name).read_text())

    assert match is None, f"{name} fetches from outside the app: {match.group(0)!r}"


def test_the_ui_mount_does_not_shadow_the_api(client):
    """`/` serving the shell must not swallow the endpoints underneath it."""
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/tools").status_code == 200


def test_a_missing_asset_is_404_not_the_shell(client):
    """A typo'd import should fail loudly rather than silently return HTML."""
    assert client.get("/ui/nope.js").status_code == 404


def test_vendored_monaco_loader_is_served(client):
    """Monaco is vendored locally — served from /ui/vendor/monaco/vs/loader.js, not a CDN.
    This test proves the static mount covers the vendor tree and Monaco is actually present."""
    response = client.get("/ui/vendor/monaco/vs/loader.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert len(response.content) > 1000    # not an empty file


def test_index_loads_monaco_from_local_vendor(client):
    """The shell must reference the locally vendored Monaco loader, never a CDN."""
    body = client.get("/").text

    assert "/ui/vendor/monaco/vs/loader.js" in body
    # Belt-and-suspenders: the EXTERNAL regex from the offline test must still pass
    # (the offline test covers all named ASSETS; this covers index.html explicitly).
    import re
    EXTERNAL = re.compile(
        r"""(?:src|href)\s*=\s*["'](?:https?:)?//"""
        r"""|url\(\s*["']?(?:https?:)?//"""
        r"""|(?:^|\s)(?:from|import)\s*\(?\s*["'](?:https?:)?//""",
        re.MULTILINE,
    )
    assert EXTERNAL.search(body) is None, "index.html fetches from an external host"
