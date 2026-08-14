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


def _view(page, name):
    """Click one of the two activity-bar buttons. Both views render into `#rail-list`."""
    page.click(f"#activity-{name}")


def _approve_a_run(page, base, session):
    """Drive the scripted gate to completion so a real job with a real log exists."""
    _open(page, base, session)
    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")
    _say(page, "train it")
    page.wait_for_selector("#approval:not([hidden])")
    page.click("#approval-approve")
    page.wait_for_selector("#approval", state="hidden")


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

    _view(page, "jobs")
    page.wait_for_selector("#rail-list .msg-notice")
    assert "No training runs yet" in page.inner_text("#rail-list")


def test_the_monitor_keeps_polling_after_a_transient_conflict(page):
    """A 409 must not freeze the live monitor.

    Phase 7's optimistic concurrency answers 409 while the job store is mid-write, which
    is most likely just after a run starts — exactly when someone is watching. The first
    version of `refreshJobs` returned without re-arming its timer and stopped updating for
    good; the live DoD found it.

    Two refusals, not one: the first lands during boot and the second on the view switch,
    so only the re-armed *timer* can produce the render this asserts on. The empty state is
    no longer proof of anything — the rail paints it from an empty cache before the first
    successful poll — so the third response carries a job that can only have come over the
    wire.
    """
    page, base, _ = page
    calls = {"n": 0}

    def flaky(route):
        calls["n"] += 1
        if calls["n"] <= 2:
            route.fulfill(status=409, content_type="application/json",
                          body='{"error": "changed on disk since it was read",'
                               ' "retryable": true}')
        else:
            route.fulfill(status=200, content_type="application/json",
                          body='[{"id": "recovered_run", "task": "splice_site",'
                               ' "arm": "lora", "state": "succeeded"}]')

    page.route("**/api/jobs", flaky)
    _open(page, base, "browser_flaky")
    _view(page, "jobs")

    page.wait_for_selector(".job-row", timeout=15000)
    assert "recovered_run" in page.inner_text("#rail-list")
    assert calls["n"] >= 3, "the monitor gave up after a conflict"


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
    _approve_a_run(page, base, "browser_approve")

    _view(page, "jobs")
    page.wait_for_selector(".job-row", timeout=15000)
    assert "browser_run" in page.inner_text("#rail-list")
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


def test_a_terminal_session_appears_in_the_rail(page):
    """The two front ends are one system: the rail is fed by the shared checkpointer."""
    page, base, _ = page
    _open(page, base, "browser_rail")
    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")

    page.goto(f"{base}/")
    page.wait_for_selector(".session-row")

    assert "browser_rail" in page.inner_text("#rail-list")
    # The row is dated from the checkpoint id, not from anything the client stores.
    assert page.locator(".session-row .session-when").first.inner_text() != ""


def test_the_rail_creates_renames_and_deletes(page):
    page, base, _ = page
    _open(page, base, "browser_crud")
    page.wait_for_selector(".session-row")

    page.once("dialog", lambda dialog: dialog.accept("made-in-rail"))
    page.click("#rail-new")
    page.wait_for_selector("text=made-in-rail")

    row = page.locator(".session-row", has_text="made-in-rail")
    page.once("dialog", lambda dialog: dialog.accept("renamed-in-rail"))
    row.locator("button[title^='Rename']").click()
    page.wait_for_selector("text=renamed-in-rail")
    assert "made-in-rail" not in page.inner_text("#rail-list")

    page.once("dialog", lambda dialog: dialog.accept())  # the confirm()
    page.locator(".session-row", has_text="renamed-in-rail") \
        .locator("button[title^='Delete']").click()
    page.wait_for_function(
        "!document.getElementById('rail-list').innerText.includes('renamed-in-rail')"
    )


def test_the_rail_filter_narrows_the_list(page):
    page, base, _ = page
    _open(page, base, "browser_filter")
    page.wait_for_selector(".session-row")

    # Two real sessions: only a session with a checkpoint is listed, and `browser_filter`
    # has never taken a turn, so it is a client-side draft that the reload would drop.
    for name in ("zzz-unlikely", "aaa-other"):
        page.once("dialog", lambda dialog, n=name: dialog.accept(n))
        page.click("#rail-new")
        page.wait_for_selector(f"text={name}")

    page.wait_for_function("document.querySelectorAll('.session-row').length === 2")

    page.fill("#rail-filter", "zzz")
    page.wait_for_function("document.querySelectorAll('.session-row').length === 1")
    assert "zzz-unlikely" in page.inner_text("#rail-list")

    page.fill("#rail-filter", "")
    page.wait_for_function("document.querySelectorAll('.session-row').length === 2")


def test_the_rail_collapses_and_the_width_survives_a_reload(page):
    page, base, _ = page
    _open(page, base, "browser_geometry")
    page.wait_for_selector(".session-row")

    def rail_width():
        return page.evaluate("document.getElementById('rail').getBoundingClientRect().width")

    assert rail_width() > 0
    page.click("#rail-toggle")
    page.wait_for_function(
        "document.getElementById('rail').getBoundingClientRect().width === 0"
    )

    page.click("#rail-toggle")   # back open, then resize it
    page.evaluate("window.localStorage.setItem('adaptrna.rail.width', '22')")
    page.reload()
    page.wait_for_selector(".session-row")

    assert rail_width() == pytest.approx(22 * 16, abs=2)


# ------------------------------------------------------------------ activity bar + job log

def test_the_activity_bar_switches_the_rail_between_sessions_and_jobs(page):
    page, base, _ = page
    _open(page, base, "browser_views")
    page.wait_for_selector(".session-row")
    assert page.is_visible("#rail-new")

    _view(page, "jobs")
    page.wait_for_selector("#rail-list .msg-notice")
    assert page.locator(".session-row").count() == 0
    # There is no "＋ New run" and never will be: starting one is a gated action in the chat.
    assert not page.is_visible("#rail-new")
    assert "behind the approval gate" in page.inner_text("#rail-list")

    _view(page, "sessions")
    page.wait_for_selector(".session-row")
    assert page.is_visible("#rail-new")


def test_clicking_the_open_view_collapses_the_rail(page):
    """VS Code's behaviour — and the activity bar never collapses with it."""
    page, base, _ = page
    _open(page, base, "browser_recollapse")
    page.wait_for_selector(".session-row")

    page.click("#activity-sessions")
    page.wait_for_function("document.getElementById('rail').getBoundingClientRect().width === 0")
    assert page.evaluate(
        "document.getElementById('activity').getBoundingClientRect().width"
    ) > 0, "the activity bar collapsed with the rail; there would be no way back"

    page.click("#activity-sessions")
    page.wait_for_function("document.getElementById('rail').getBoundingClientRect().width > 0")


def test_selecting_a_job_replaces_the_chat_with_its_log(page):
    page, base, _ = page
    _approve_a_run(page, base, "browser_joblog")

    _view(page, "jobs")
    page.wait_for_selector(".job-row", timeout=15000)
    page.click(".job-open")

    page.wait_for_selector("#joblog:not([hidden])")
    assert not page.is_visible("#chat")
    # The composer belongs to a session, not a run.
    assert not page.is_visible("#composer-input")
    assert "browser_run" in page.inner_text("#joblog-head")

    # The scripted plan runs `/bin/echo trained …`, so that word reaches `train.log`.
    page.wait_for_function(
        "document.getElementById('joblog-body').innerText.includes('trained')",
        timeout=15000,
    )


def test_closing_the_job_log_returns_to_the_chat(page):
    page, base, _ = page
    _approve_a_run(page, base, "browser_jobclose")

    _view(page, "jobs")
    page.wait_for_selector(".job-row", timeout=15000)

    page.click(".job-open")
    page.wait_for_selector("#joblog:not([hidden])")
    page.click("#joblog .btn-close")
    page.wait_for_selector("#joblog", state="hidden")
    assert page.is_visible("#composer-input")

    # Picking a conversation is the other way back.
    page.click(".job-open")
    page.wait_for_selector("#joblog:not([hidden])")
    _view(page, "sessions")
    page.click(".session-open")
    page.wait_for_selector("#joblog", state="hidden")
    assert page.is_visible("#composer-input")


def test_the_open_view_survives_a_reload(page):
    page, base, _ = page
    _open(page, base, "browser_viewreload")
    page.wait_for_selector(".session-row")

    _view(page, "jobs")
    page.wait_for_selector("#rail-list .msg-notice")

    page.reload()
    page.wait_for_selector("#composer-input:not([disabled])")
    page.wait_for_selector("#rail-list .msg-notice")

    assert page.get_attribute("#activity-jobs", "aria-selected") == "true"
    assert page.locator(".session-row").count() == 0


def test_the_grip_resizes_from_the_rails_own_left_edge(page):
    """The drag measured `clientX` against the viewport, so an activity bar to the left of
    the rail made every drag report a width one bar too wide."""
    page, base, _ = page
    _open(page, base, "browser_grip")
    page.wait_for_selector(".session-row")

    bar = page.evaluate("document.getElementById('activity').getBoundingClientRect().width")
    assert bar > 0, "there is no activity bar to be offset by"

    box = page.locator("#rail-grip").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 40)
    page.mouse.down()
    page.mouse.move(360, box["y"] + 40, steps=8)
    page.mouse.up()

    width = page.evaluate("document.getElementById('rail').getBoundingClientRect().width")
    assert width == pytest.approx(360 - bar, abs=4), "the rail's edge did not land under the pointer"


def test_the_thinking_dots_are_dark_at_rest(page):
    """The dots animated from page load for the whole of Phase 9: `.thinking` set an
    author-origin `display`, which beats the user agent's `[hidden] { display: none }`."""
    page, base, _ = page
    _open(page, base, "browser_dots")

    assert page.locator("#thinking").is_visible() is False

    _say(page, "hello")
    page.wait_for_selector(".msg-assistant")
    page.wait_for_function(
        "getComputedStyle(document.getElementById('thinking')).display === 'none'"
    )
