"""Resident provenance queries preserve lifecycle and portable snapshot identity.

Reads: session snapshot captured after verified load; no model bytes needed.
"""

import copy
import json
from pathlib import Path

import pytest

from beats_audioset_infer import BEATsAudioSetSession


@pytest.mark.parametrize(
    "status", ["new", "loading", "busy", "failed", "released", "closed"]
)
def test_loaded_identity_requires_ready(status, tmp_path):
    session = BEATsAudioSetSession(cache_dir=tmp_path / "absent", offline=True)
    session._status = status
    with pytest.raises(RuntimeError, match="load"):
        session.loaded_checkpoint_info()
    assert not (tmp_path / "absent").exists()


def test_loaded_identity_is_memory_only_defensive_snapshot(tmp_path, monkeypatch):
    session = BEATsAudioSetSession(cache_dir=tmp_path, offline=True)
    snapshot = session._store.provenance()
    snapshot["revision"] = "loaded-revision"
    snapshot["files"]["model.pt"]["sha256"] = "1" * 64
    snapshot["encoder_provenance"] = {"details": ["original"]}
    session._checkpoint = snapshot
    session._device = "cuda:1"
    session._status = "ready"
    original = copy.deepcopy(snapshot)
    session._store._registry["revision"] = "later-registry-change"

    def forbidden(*args, **kwargs):
        raise AssertionError("Resident identity must never consult disk/cache")

    for method in ["provenance", "resolve", "cache_info"]:
        monkeypatch.setattr(session._store, method, forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    info = session.loaded_checkpoint_info()
    assert info["model_id"] == "beats-audioset"
    assert info["checkpoint_id"] == f"{original['repo']}@loaded-revision"
    assert info["checkpoint_sha256"] == "1" * 64
    assert info["device"] == "cuda:1"
    expected = copy.deepcopy(original)
    for artifact in expected["files"].values():
        artifact.pop("path")
    assert {k: info[k] for k in expected} == expected
    assert str(tmp_path) not in json.dumps(info, allow_nan=False)
    info["files"]["model.pt"]["sha256"] = "0" * 64
    info["encoder_provenance"]["details"].append("modified")
    assert session._checkpoint == original
    assert session.loaded_checkpoint_info()["checkpoint_sha256"] == "1" * 64


def test_loaded_identity_uses_session_lock(tmp_path):
    import threading

    session = BEATsAudioSetSession(cache_dir=tmp_path, offline=True)
    started = threading.Event()
    finished = threading.Event()
    errors = []

    def query():
        started.set()
        try:
            session.loaded_checkpoint_info()
        except RuntimeError as error:
            errors.append(error)
        finally:
            finished.set()

    with session._lock:
        worker = threading.Thread(target=query)
        worker.start()
        assert started.wait(timeout=1)
        try:
            assert not finished.wait(timeout=0.05)
        finally:
            session._status = "released"
    worker.join(timeout=1)
    assert not worker.is_alive() and finished.is_set() and len(errors) == 1
