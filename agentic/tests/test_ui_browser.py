"""The UI in a real browser. Opt-in: `pytest -m ui`.

    ../.venv/bin/python -m pip install playwright
    ../.venv/bin/python -m playwright install chromium
    ../.venv/bin/python -m pytest -m ui

Everything else in the suite tests the client's *contract*; only these tests prove that
the JavaScript actually runs — that the fetch-based SSE reader assembles frames, that the
modal renders, that a refresh restores a pending approval. A real server on a real port,
because the parts most worth testing here (streaming, `sessionStorage`, module imports)
are exactly the parts a mocked page would not exercise.

The model is scripted, so no API key and no GPU are involved and the assertions are exact.
"""

import socket
import sys
import threading
import time

import pytest
from langchain_core.messages import AIMessage

from adaptrna_agentic.profiling.recommender import PLAN_SOURCE
from api_helpers import build_test_app
from scripted_model import tool_call

pytestmark = pytest.mark.ui

playwright_api = pytest.importorskip(
    "playwright.sync_api",
    reason="pip install playwright && playwright install chromium",
)

COMMAND = ["/bin/echo", "trained", "--task", "splice_site", "--use_lora"]
WARNING = "Cross-species benchmark: test is a different organism"

#: Shaped like what the orchestrator actually writes — it reaches for a table whenever it
#: reports metrics, which is most of the time.
MARKDOWN_REPLY = """Hello from the scripted model.

| Metric | Value |
|---|---|
| F1 | **0.9663** |

- first point
- second point

Run `toolhub list` to see them."""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="function")
def server(nano_registry, nano_splice_adapter, tmp_path, monkeypatch):
    """A real uvicorn on a real port, with a scripted model."""
    import uvicorn

    monkeypatch.setenv("ADAPTRNA_JOBS_DIR", str(tmp_path / "jobs_data"))
    monkeypatch.setattr("adaptrna_agentic.jobs.runner.REPO_ROOT", tmp_path)
    monkeypatch.setattr("adaptrna_agentic.jobs.store.REPO_ROOT", tmp_path)
    nano_registry.register(nano_splice_adapter)

    plan = {
        "source": PLAN_SOURCE, "task": "splice_site", "arm": "lora",
        "output_dir": str(tmp_path / "outputs" / "browser_run"),
        "command": COMMAND, "overrides": {},
        "estimated_wall_clock": "~7 min", "warnings": [WARNING],
    }

    app, _ = build_test_app(nano_registry, tmp_path / "s.sqlite", script=[
        AIMessage(content=MARKDOWN_REPLY),
        AIMessage(content="", tool_calls=[tool_call("start_training", {"plan": plan})]),
        AIMessage(content="Training started."),
        AIMessage(content="Anything else?"),
    ])

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    deadline = time.time() + 20
    while not uvicorn_server.started and time.time() < deadline:
        time.sleep(0.05)
    assert uvicorn_server.started, "the test server never came up"

    yield f"http://127.0.0.1:{port}", tmp_path

    uvicorn_server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def page(server):
    base, tmp_path = server

    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        page = context.new_page()
        page.on("pageerror", lambda error: pytest.fail(f"uncaught JS error: {error}"))
        yield page, base, tmp_path
        context.close()
        browser.close()


def _open(page, base, session):
    page.goto(f"{base}/?session={session}")
    page.wait_for_selector("#composer-input:not([disabled])")


def _say(page, text):
    page.fill("#composer-input", text)
    page.click("#composer-send")


def _jobs(tmp_path):
    from adaptrna_agentic.jobs.store import JobStore

    return JobStore(tmp_path / "jobs_data").jobs


# ------------------------------------------------------------------ chat

def test_the_app_loads_and_streams_a_reply(page):
    page, base, _ = page
    _open(page, base, "browser_chat")

    _say(page, "hello")

    page.wait_for_selector(".msg-assistant")
    assert "scripted model" in page.inner_text(".msg-assistant")
    assert page.inner_text(".msg-user") == "hello"


def test_markdown_in_a_reply_is_rendered_not_shown_raw(page):
    """The model answers in Markdown and leans on tables; raw pipes are unreadable."""
    page, base, _ = page
    _open(page, base, "browser_md")

    _say(page, "hello")

    page.wait_for_selector(".msg-assistant .md-table")
    reply = page.locator(".msg-assistant").last

    assert reply.locator(".md-table th").first.inner_text() == "Metric"
    assert "0.9663" in reply.locator(".md-table").inner_text()
    assert reply.locator(".md-table strong").count() == 1, "**bold** survived into a cell"
    assert reply.locator(".md-list li").count() == 2
    assert reply.locator(".md-inline-code").first.inner_text() == "toolhub list"

    text = reply.inner_text()
    for raw in ("|---|", "**0.9663**", "`toolhub list`"):
        assert raw not in text, f"unrendered Markdown left in the reply: {raw}"


def test_the_panels_render_server_state(page):
    page, base, _ = page
    _open(page, base, "browser_panels")

    page.wait_for_selector(".tool-row")
    assert "splice_site" in page.inner_text("#tools-list")
    assert "No training runs yet" in page.inner_text("#jobs-list")


def test_the_monitor_keeps_polling_after_a_transient_conflict(page):
    """A 409 must not freeze the live monitor.

    Phase 7's optimistic concurrency answers 409 while the job store is mid-write, which
    is most likely just after a run starts — exactly when someone is watching this panel.
    The first version of `refreshJobs` returned without re-arming its timer and stopped
    updating for good; the live DoD found it.
    """
    page, base, _ = page
    calls = {"n": 0}

    def flaky(route):
        calls["n"] += 1
        if calls["n"] == 1:
            route.fulfill(status=409, content_type="application/json",
                          body='{"error": "changed on disk since it was read",'
                               ' "retryable": true}')
        else:
            route.continue_()

    page.route("**/api/jobs", flaky)
    _open(page, base, "browser_flaky")

    # The panel can only reach its empty state by polling again after the refusal.
    page.wait_for_selector("#jobs-list .msg-notice", timeout=15000)
    assert "No training runs yet" in page.inner_text("#jobs-list")
    assert calls["n"] >= 2, "the monitor gave up after one conflict"


def test_a_tool_can_be_disabled_and_re_enabled(page):
    page, base, _ = page
    _open(page, base, "browser_tools")
    page.wait_for_selector(".tool-row")

    row = page.locator(".tool-row", has_text="splice_site").first
    row.get_by_role("button", name="disable").click()

    page.wait_for_selector(".tool-row.is-disabled")
    row = page.locator(".tool-row", has_text="splice_site").first
    row.get_by_role("button", name="enable").click()

    page.wait_for_selector(".tool-row.is-active")


# ------------------------------------------------------------------ the gate

def test_the_modal_shows_the_exact_command_and_starts_nothing(page):
    """Gate 4: verbatim, and nothing runs before the human answers."""
    page, base, tmp_path = page
    _open(page, base, "browser_gate")

    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")
    _say(page, "train it")

    page.wait_for_selector("#approval:not([hidden])")

    shown = page.inner_text("#approval-body")
    assert " ".join(COMMAND) in shown, f"the modal did not show the command verbatim: {shown}"
    assert WARNING in shown
    assert not _jobs(tmp_path), "a job started BEFORE the human answered"


def test_declining_starts_nothing(page):
    page, base, tmp_path = page
    _open(page, base, "browser_decline")
    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")
    _say(page, "train it")
    page.wait_for_selector("#approval:not([hidden])")

    page.fill("#approval-note", "not now")
    page.click("#approval-decline")

    page.wait_for_selector("#approval", state="hidden")
    page.wait_for_function("!document.getElementById('composer-send').disabled")
    assert not _jobs(tmp_path), "a job ran despite the refusal"


def test_approving_starts_the_job_and_the_monitor_shows_it(page):
    page, base, tmp_path = page
    _open(page, base, "browser_approve")
    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")
    _say(page, "train it")
    page.wait_for_selector("#approval:not([hidden])")

    page.click("#approval-approve")

    page.wait_for_selector("#approval", state="hidden")
    page.wait_for_selector(".job-row", timeout=15000)
    assert "browser_run" in page.inner_text("#jobs-list")
    assert "browser_run" in _jobs(tmp_path)


def test_a_refresh_mid_approval_restores_the_dialog(page):
    """The suspended turn lives in the checkpointer; the modal must come back with it."""
    page, base, _ = page
    _open(page, base, "browser_refresh")
    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")
    _say(page, "train it")
    page.wait_for_selector("#approval:not([hidden])")

    page.reload()

    page.wait_for_selector("#approval:not([hidden])")
    assert " ".join(COMMAND) in page.inner_text("#approval-body")


# ------------------------------------------------------------------ sessions

def test_history_is_restored_on_reload(page):
    page, base, _ = page
    _open(page, base, "browser_history")
    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")

    page.reload()

    page.wait_for_selector(".msg-user")
    assert page.inner_text(".msg-user") == "hello"
    assert "scripted model" in page.inner_text(".msg-assistant")


def test_a_terminal_session_appears_in_the_picker(page):
    """The two front ends are one system: the picker is fed by the shared checkpointer."""
    page, base, _ = page
    _open(page, base, "browser_picker")
    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")

    page.goto(f"{base}/")
    # `<option>` is never "visible" to Playwright, so wait on the count instead.
    page.wait_for_function("document.querySelectorAll('#session-picker option').length > 0")

    assert "browser_picker" in page.inner_text("#session-picker")
