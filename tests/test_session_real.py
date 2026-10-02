"""Real offline facade, embeddings, mapping and model lifetime regression.

Tests require the explicit pinned checkpoint path; no automatic test downloads.
Reads: original baseline fixtures and public session facade.
"""

import json
import os
import weakref
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from parity_support import baseline_root

from beats_audioset_infer import BEATsAudioSetSession


@pytest.fixture(scope="module")
def session():
    path = os.environ.get("BEATS_TEST_CHECKPOINT")
    if not path:
        pytest.skip("Set BEATS_TEST_CHECKPOINT for real public API regression")
    device = os.environ.get("BEATS_TEST_DEVICE", "cpu")
    s = BEATsAudioSetSession(
        device=device, checkpoint_dir=Path(path).parent, offline=True
    )
    s.load()
    yield s
    s.close()


@pytest.mark.parametrize("name", ["noise", "silence", "tones"])
def test_facade_parity(session, name):
    f = Path(__file__).parent / "fixtures"
    r = session.infer(f / (name + ".wav"))
    device = os.environ.get("BEATS_TEST_DEVICE", "cpu")
    gold_dir, environment = baseline_root(f, device)
    gold = np.load(gold_dir / (name + "_" + device + ".npz"))
    import torch

    compare = (
        np.testing.assert_array_equal
        if torch.__version__ == environment["torch"]
        else lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
    )
    compare(r.scores, gold["scores"][0])
    compare(r.patch_embeddings, gold["tokens"][0])
    compare(r.clip_embedding, gold["tokens"][0].mean(0))
    assert len(r.scores) == 527 and r.mids[2] == "/m/04rlf" and r.labels[2] == "Music"
    assert (
        len(set(r.mids)) == 527
        and r.embedding_info["frequency_patches_per_time_patch"] == 8
    )
    json.dumps(r.to_dict(), allow_nan=False)
    assert not r.patch_embeddings.flags.writeable


@pytest.mark.parametrize("rate", [16000, 32000, 44100])
@pytest.mark.parametrize("kind", ["array", "tensor", "file"])
def test_opt_in_padding_matches_explicit_normalized_waveform(
    session, rate, kind, tmp_path
):
    import torch

    from beats_audioset_infer._impl.resample import resample

    count = rate // 10
    audio = np.stack(
        [
            np.sin(np.arange(count, dtype=np.float32) * 0.04),
            np.cos(np.arange(count, dtype=np.float32) * 0.02),
        ]
    )
    source = audio
    input_rate = rate
    if kind == "tensor":
        source = torch.from_numpy(audio)
    elif kind == "file":
        source = tmp_path / "short.wav"
        sf.write(source, audio.T, rate, subtype="FLOAT")
        input_rate = None
    normalized = torch.from_numpy(audio).to(session.device)
    if rate != 16000:
        normalized = resample(normalized, rate, 16000)
    normalized = normalized.mean(0, keepdim=True)
    padded = torch.nn.functional.pad(normalized, (0, 2800 - normalized.shape[1]))
    explicit = session.infer(padded, sample_rate=16000)
    opt_in = session.infer(source, sample_rate=input_rate, pad_short=True)
    assert opt_in.scores == explicit.scores
    np.testing.assert_array_equal(opt_in.patch_embeddings, explicit.patch_embeddings)
    np.testing.assert_array_equal(opt_in.clip_embedding, explicit.clip_embedding)
    assert opt_in.audio["samples"] == count and opt_in.audio["channels"] == 2
    assert opt_in.audio["normalized_samples"] == normalized.shape[1]
    assert opt_in.audio["model_samples"] == 2800
    assert opt_in.audio["padding_samples"] == 2800 - normalized.shape[1]
    assert "waveform right-zero padded" in opt_in.embedding_info["padding"]
    assert opt_in.embedding_info["time_patches"] == 1
    assert session.status == "ready"


def test_inputs_reuse_release_reload(session, monkeypatch):
    f = Path(__file__).parent / "fixtures" / "tones.wav"
    data, sr = sf.read(f, dtype="float32", always_2d=True)
    first = session.infer(f)
    expected_hash = first.checkpoint["files"]["model.pt"]["sha256"]
    identity = session.loaded_checkpoint_info()
    assert identity["checkpoint_sha256"] == expected_hash
    assert identity["device"] == session.device
    assert identity["model_id"] == "beats-audioset"
    assert (
        identity["checkpoint_id"]
        == f"{first.checkpoint['repo']}@{first.checkpoint['revision']}"
    )
    assert all("path" not in artifact for artifact in identity["files"].values())
    first.checkpoint["files"]["model.pt"]["sha256"] = "0" * 64
    model = session._model
    reference = weakref.ref(model)
    session.load()
    assert session._model is model
    array = session.infer(data.T, sample_rate=sr)
    assert array.checkpoint["files"]["model.pt"]["sha256"] == expected_hash
    import torch

    tensor = session.infer(torch.from_numpy(data.T.copy()), sample_rate=sr)
    for r in [array, tensor]:
        np.testing.assert_array_equal(r.patch_embeddings, first.patch_embeddings)
        assert r.scores == first.scores
    with pytest.raises(ValueError):
        session.infer(np.zeros(10), sample_rate=sr)
    assert session.status == "ready"
    del model
    session.release()
    assert reference() is None
    with pytest.raises(RuntimeError):
        session.loaded_checkpoint_info()
    with pytest.raises(RuntimeError):
        session.infer(f)
    session.load()
    assert session.loaded_checkpoint_info() == identity
    again = session.infer(f)
    np.testing.assert_array_equal(again.patch_embeddings, first.patch_embeddings)
    assert again.scores == first.scores
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError):
        session.loaded_checkpoint_info()
    json.dumps(first.to_dict(), allow_nan=False)
