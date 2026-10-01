"""Offline input, cache, result and lifecycle contract tests. Reads: fixtures."""

import numpy as np
import pytest

from beats_audioset_infer import BEATsAudioSetSession, __version__
from beats_audioset_infer._impl.audio import normalize
from beats_audioset_infer.checkpoints import (
    CheckpointStore,
    _load_registry,
    _validate_registry,
)
from beats_audioset_infer.session import _device


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_normalization_rate(rate):
    x, info = normalize(np.zeros((2, rate), np.float32), rate, "cpu")
    assert (
        x.shape == (1, 16000)
        and info["channels"] == 2
        and info["duration_seconds"] == 1
    )


@pytest.mark.parametrize(
    "audio,rate",
    [
        (np.zeros(3000), None),
        (np.zeros(3000), True),
        (np.zeros(3000), 0),
        (np.zeros(10), 16000),
        (np.full(3000, np.nan), 16000),
        (np.zeros((1, 1, 3000)), 16000),
        (np.empty(0), 16000),
        (object(), 16000),
        (np.ones(3000, dtype=complex), 16000),
        (np.ones(3000, dtype=bool), 16000),
    ],
)
def test_invalid_input(audio, rate):
    with pytest.raises((TypeError, ValueError)):
        normalize(audio, rate, "cpu")


@pytest.mark.parametrize("choice", ["mps", "xpu", "cuda:-1", None, True, "cpu:0"])
def test_invalid_device(choice):
    with pytest.raises(ValueError):
        _device(choice)


def test_no_cold_start_and_terminal(tmp_path):
    s = BEATsAudioSetSession(cache_dir=tmp_path / "absent", offline=True)
    assert s.status == "new" and __version__ == "0.1.0"
    assert not (tmp_path / "absent").exists()
    s.cache_info()
    assert not (tmp_path / "absent").exists()
    with pytest.raises(RuntimeError):
        s.infer(np.zeros(3000), sample_rate=16000)
    s.release()
    assert s.status == "released"
    with pytest.raises(FileNotFoundError):
        s.load()
    assert s.status == "failed"
    s.close()
    s.close()
    assert s.status == "closed"
    for op in [s.load, s.release]:
        with pytest.raises(RuntimeError):
            op()


def test_cache_corruption_before_deserialization(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"incorrect")
    with pytest.raises(ValueError, match="size mismatch"):
        CheckpointStore(checkpoint_dir=tmp_path).resolve()


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "../escape"),
        ("source_revision", "bad"),
        ("updated_at", "invalid"),
        ("files", {}),
        ("audioset_licenses", {}),
    ],
)
def test_registry_validation(field, value):
    r = _load_registry()
    r[field] = value
    with pytest.raises(ValueError):
        _validate_registry(r)


def test_registry_integrity():
    r = _validate_registry(_load_registry())
    assert r["files"]["model.pt"]["size"] == 363145291
    assert r["model_license"].startswith("unspecified")


def test_custom_registry(tmp_path):
    import json

    registry = _load_registry()
    registry["revision"] = "custom-v1"
    registry["files"]["model.pt"]["url"] = "https://example.org/custom.pt"
    config = tmp_path / "registry.json"
    config.write_text(json.dumps(registry))
    s = BEATsAudioSetSession(
        cache_dir=tmp_path / "cache", checkpoint_config=config, offline=True
    )
    assert s.cache_info()["revision"] == "custom-v1"
    assert not (tmp_path / "cache").exists()
    config.write_text('{"bad": 1}')
    with pytest.raises(ValueError):
        BEATsAudioSetSession(checkpoint_config=config)
    config.write_text("{")
    with pytest.raises(ValueError):
        BEATsAudioSetSession(checkpoint_config=config)


def test_cache_atomic_download_and_hash(tmp_path, monkeypatch):
    import hashlib
    import io

    from beats_audioset_infer import checkpoints

    class Response(io.BytesIO):
        def geturl(self):
            return "https://example.org/model.pt"

    monkeypatch.setattr(checkpoints, "urlopen", lambda *a, **k: Response(b"abc"))
    store = CheckpointStore(cache_dir=tmp_path)
    path = tmp_path / "artifact.pt"
    artifact = {
        "url": "https://example.org/model.pt",
        "size": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }
    store._download(path, artifact)
    assert path.read_bytes() == b"abc" and not list(tmp_path.glob(".*.tmp"))
    artifact["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        store._download(path, artifact)
    assert path.read_bytes() == b"abc" and not list(tmp_path.glob(".*.tmp"))


def test_explicit_cuda_not_substituted(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        _device("cuda")
    with pytest.raises(RuntimeError):
        _device("cuda:0")
    assert str(_device("auto")) == "cpu"


def test_unknown_hardware_requires_reference(monkeypatch):
    from pathlib import Path

    import parity_support

    monkeypatch.setattr(
        parity_support, "fingerprint", lambda device: {"device": "different_cpu"}
    )
    with pytest.raises(AssertionError, match="CPU dispatch differs"):
        parity_support.baseline_root(Path(__file__).parent / "fixtures", "cpu")


def test_unverified_alternate_baseline_rejected(tmp_path, monkeypatch):
    import json

    import parity_support

    (tmp_path / "environment_cpu.json").write_text(
        json.dumps({"hardware": parity_support.fingerprint("cpu")})
    )
    monkeypatch.setenv("BEATS_TEST_BASELINE", str(tmp_path))
    with pytest.raises(AssertionError, match="same-host A/B provenance"):
        parity_support.baseline_root(tmp_path / "committed", "cpu")


def test_cli_rejects_colliding_artifacts_before_load(tmp_path, monkeypatch, capsys):
    import sys

    from beats_audioset_infer.cli import main

    artifact = tmp_path / "existing.npz"
    artifact.write_bytes(b"original")

    def forbidden_load(self):
        raise AssertionError("Colliding outputs must fail before load")

    monkeypatch.setattr(BEATsAudioSetSession, "load", forbidden_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "beats-audioset-infer",
            "unused.wav",
            "--output",
            str(artifact),
            "--embeddings-output",
            str(tmp_path / "." / "existing.npz"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert artifact.read_bytes() == b"original"
    assert "must be different paths" in capsys.readouterr().err
