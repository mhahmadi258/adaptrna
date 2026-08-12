"""External tools through the management CLI, including the approval-gated install."""

import json

import pytest

from adaptrna_agentic.cli.toolhub import main

DUMMY = "fixtures.dummy_external"


@pytest.fixture
def data_dir(tmp_path):
    dd = tmp_path / "toolhub_data"
    assert main(["--data-dir", str(dd), "config",
                 "--lm-config", "nano", "--weights", "null", "--device", "cpu"]) == 0
    return dd


def run(data_dir, *argv):
    return main(["--data-dir", str(data_dir), *argv])


def test_register_external_and_list(data_dir, capsys):
    assert run(data_dir, "register-external", DUMMY) == 0
    out = capsys.readouterr().out
    assert "Registered 'dummy_echo'" in out and "Registered 'dummy_add'" in out

    assert run(data_dir, "list") == 0
    out = capsys.readouterr().out
    assert "external" in out and "dummy_add" in out


def test_missing_package_without_yes_declines_with_command(data_dir, monkeypatch, capsys):
    from adaptrna_agentic.toolhub.external import contract

    monkeypatch.setattr(contract, "is_available", lambda package: False)

    # Non-interactive stdin -> the gate declines; the exact command is shown.
    assert run(data_dir, "register-external", DUMMY) == 1
    captured = capsys.readouterr()
    assert "pip install dummy-package" in captured.out       # "Would run: ..."
    assert "Install not approved" in captured.err
    assert "--yes" in captured.err


def test_missing_package_with_yes_runs_the_installer(data_dir, monkeypatch, capsys):
    from adaptrna_agentic.toolhub.external import contract

    state = {"installed": False, "install_calls": 0}

    def fake_is_available(package):
        return state["installed"]

    def fake_install(package):
        state["installed"] = True
        state["install_calls"] += 1
        return "9.9"

    monkeypatch.setattr(contract, "is_available", fake_is_available)
    monkeypatch.setattr(contract, "install", fake_install)

    assert run(data_dir, "register-external", DUMMY, "--yes") == 0
    out = capsys.readouterr().out
    assert "Installed dummy-package 9.9" in out
    assert state["install_calls"] == 1
    assert "Registered 'dummy_echo'" in out


def test_call_with_json_args_and_kv_form(data_dir, capsys):
    run(data_dir, "register-external", DUMMY)
    capsys.readouterr()

    assert run(data_dir, "call", "dummy_add", "--args", '{"a": 2, "b": 3}') == 0
    assert json.loads(capsys.readouterr().out)["total"] == 5.0

    assert run(data_dir, "call", "dummy_add", "a=2", "b=3") == 0
    assert json.loads(capsys.readouterr().out)["total"] == 5.0


def test_disabled_external_refuses_call(data_dir, capsys):
    run(data_dir, "register-external", DUMMY)
    run(data_dir, "deactivate", "dummy_echo")
    capsys.readouterr()

    assert run(data_dir, "call", "dummy_echo", "value=hi") == 1
    assert "activate dummy_echo" in capsys.readouterr().err


def test_predict_on_external_points_to_call(data_dir, capsys):
    run(data_dir, "register-external", DUMMY)
    capsys.readouterr()

    assert run(data_dir, "predict", "dummy_echo", "--sequences", "ACGU") == 1
    assert "toolhub call dummy_echo" in capsys.readouterr().err


def test_call_on_adapter_points_to_predict(data_dir, nano_splice_adapter, capsys):
    run(data_dir, "register", str(nano_splice_adapter))
    capsys.readouterr()

    assert run(data_dir, "call", "splice_site", "sequence=ACGU") == 1
    assert "toolhub predict splice_site" in capsys.readouterr().err


def test_golden_test_via_cli(data_dir, capsys):
    run(data_dir, "register-external", DUMMY)
    capsys.readouterr()

    assert run(data_dir, "test", "dummy_echo") == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_doctored_golden_fails_via_cli(data_dir, capsys):
    run(data_dir, "register-external", DUMMY)
    capsys.readouterr()

    from adaptrna_agentic.toolhub.registry import Registry

    registry = Registry(data_dir)
    registry.get("dummy_echo").test["golden"] = [
        {"args": {"value": "hi"}, "expect": {"value": "WRONG"}}
    ]
    registry.manifest.save()

    assert run(data_dir, "test", "dummy_echo") == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_info_on_external_shows_module(data_dir, capsys):
    run(data_dir, "register-external", DUMMY)
    capsys.readouterr()

    assert run(data_dir, "info", "dummy_echo") == 0
    out = capsys.readouterr().out
    assert DUMMY in out and '"external"' in out
