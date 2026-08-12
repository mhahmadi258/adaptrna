"""Both JSON stores are whole-file read-modify-write, so a second writer must not be able
to silently discard the first one's work.

The guard is a revision counter *inside* the file rather than an mtime: two writes of
identical content within one filesystem timestamp tick are indistinguishable by mtime,
and that is precisely the racing case."""

import json

import pytest

from adaptrna_agentic.jobs.store import JobRecord, JobStore
from adaptrna_agentic.toolhub.errors import ConcurrentModificationError
from adaptrna_agentic.toolhub.manifest import Manifest, ToolEntry


def _entry(name):
    return ToolEntry(name=name, type="external", state="active", description="d")


def _record(job_id):
    return JobRecord(id=job_id, task="t", arm="lora", command=["x"], output_dir="o")


# ---------------------------------------------------------------- manifest

def test_manifest_second_writer_is_refused(tmp_path):
    first = Manifest.load(tmp_path)
    first.save()

    second = Manifest.load(tmp_path)
    second.tools["from_second"] = _entry("from_second")
    second.save()

    first.tools["from_first"] = _entry("from_first")
    with pytest.raises(ConcurrentModificationError, match="changed on disk"):
        first.save()

    # The other writer's work survived intact.
    on_disk = Manifest.load(tmp_path)
    assert set(on_disk.tools) == {"from_second"}


def test_manifest_retry_after_reload_succeeds(tmp_path):
    Manifest.load(tmp_path).save()
    other = Manifest.load(tmp_path)
    other.tools["theirs"] = _entry("theirs")
    other.save()

    # What the error message tells the user to do.
    retried = Manifest.load(tmp_path)
    retried.tools["mine"] = _entry("mine")
    retried.save()

    assert set(Manifest.load(tmp_path).tools) == {"theirs", "mine"}


def test_identical_content_written_twice_is_still_detected(tmp_path):
    """The case an mtime+size stamp misses: same bytes, same timestamp tick."""
    first = Manifest.load(tmp_path)
    first.save()
    second = Manifest.load(tmp_path)
    second.save()                      # byte-identical to the first write

    with pytest.raises(ConcurrentModificationError):
        first.save()


def test_revision_increments_and_is_persisted(tmp_path):
    manifest = Manifest.load(tmp_path)
    manifest.save()
    manifest.save()

    payload = json.loads((tmp_path / "tools.json").read_text())
    assert payload["revision"] == 2
    assert Manifest.load(tmp_path).revision == 2


def test_manifest_without_revision_still_loads(tmp_path):
    """Files written before Phase 7 have no revision key."""
    (tmp_path / "tools.json").write_text(json.dumps({
        "format_version": 1, "backbone": {}, "tools": {},
    }))

    manifest = Manifest.load(tmp_path)
    assert manifest.revision == 0
    manifest.save()                    # and can still be written


# ---------------------------------------------------------------- job store

def test_job_store_second_writer_is_refused(tmp_path):
    first = JobStore(tmp_path)
    first.save()

    second = JobStore(tmp_path)
    second.add(_record("theirs"))

    first.jobs["mine"] = _record("mine")
    with pytest.raises(ConcurrentModificationError, match="changed on disk"):
        first.save()

    assert set(JobStore(tmp_path).jobs) == {"theirs"}


def test_job_store_revision_round_trips(tmp_path):
    store = JobStore(tmp_path)
    store.add(_record("a"))
    store.add(_record("b"))

    assert JobStore(tmp_path).revision == 2
    assert set(JobStore(tmp_path).jobs) == {"a", "b"}
