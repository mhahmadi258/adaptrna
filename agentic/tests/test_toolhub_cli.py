"""The management CLI, end to end against tmp data dirs on the nano backbone."""

import json

import pytest

from adaptrna_agentic.cli.toolhub import main

SEQ = ["GGCAUUACGGCUUAAGCUAGCUAGCUAAGGCC", "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"]


@pytest.fixture
def data_dir(tmp_path):
    dd = tmp_path / "toolhub_data"
    assert main(["--data-dir", str(dd), "config",
                 "--lm-config", "nano", "--weights", "null", "--device", "cpu"]) == 0
    return dd


def run(data_dir, *argv):
    return main(["--data-dir", str(data_dir), *argv])


def test_config_round_trip(data_dir, capsys):
    assert run(data_dir, "config") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["lm_config"] == "nano"
    assert shown["weights"] is None


def test_register_list_info(data_dir, nano_splice_adapter, capsys):
    assert run(data_dir, "register", str(nano_splice_adapter),
               "--description", "nano donor tool") == 0
    out = capsys.readouterr().out
    assert "Registered 'splice_site'" in out

    assert run(data_dir, "list") == 0
    out = capsys.readouterr().out
    assert "splice_site" in out and "active" in out

    assert run(data_dir, "info", "splice_site") == 0
    out = capsys.readouterr().out
    assert "nano donor tool" in out
    assert "format_version" in out          # engine describe_adapter section


def test_lifecycle_via_cli(data_dir, nano_splice_adapter, capsys):
    run(data_dir, "register", str(nano_splice_adapter))
    capsys.readouterr()

    assert run(data_dir, "deactivate", "splice_site") == 0
    assert "disabled" in capsys.readouterr().out

    # A disabled tool refuses predict, with the fix in the message.
    assert run(data_dir, "predict", "splice_site", "--sequences", SEQ[0]) == 1
    assert "activate splice_site" in capsys.readouterr().err

    assert run(data_dir, "activate", "splice_site") == 0
    capsys.readouterr()

    assert run(data_dir, "remove", "splice_site", "--yes") == 0
    capsys.readouterr()
    assert run(data_dir, "info", "splice_site") == 1
    assert "Known tools" in capsys.readouterr().err


def test_predict_writes_json(data_dir, nano_splice_adapter, tmp_path, capsys):
    run(data_dir, "register", str(nano_splice_adapter))
    out_file = tmp_path / "predictions.json"

    assert run(data_dir, "predict", "splice_site",
               "--sequences", *SEQ, "--output", str(out_file)) == 0

    payload = json.loads(out_file.read_text())
    assert payload["sequences"] == SEQ
    values = payload["predictions"]["splice_site"]
    assert len(values) == 2
    assert all(0.0 <= v <= 1.0 for v in values)


def test_smoke_test_via_cli(data_dir, nano_splice_adapter, capsys):
    run(data_dir, "register", str(nano_splice_adapter), "--test-sequences", *SEQ)
    capsys.readouterr()

    assert run(data_dir, "test", "splice_site") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True


def test_register_missing_file_is_actionable(data_dir, capsys):
    assert run(data_dir, "register", "no/such/adapter.pt") == 1
    assert "not found" in capsys.readouterr().err


def test_missing_weights_message_via_cli(data_dir, nano_splice_adapter, capsys):
    run(data_dir, "register", str(nano_splice_adapter))
    run(data_dir, "config", "--weights", "does/not/exist.pt")
    capsys.readouterr()

    assert run(data_dir, "predict", "splice_site", "--sequences", SEQ[0]) == 1
    assert "toolhub config --weights" in capsys.readouterr().err
