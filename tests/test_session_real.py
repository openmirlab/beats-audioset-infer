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
    gold = np.load(f / (name + "_" + device + ".npz"))
    import torch

    environment = json.loads((f / ("environment_" + device + ".json")).read_text())
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


def test_inputs_reuse_release_reload(session, monkeypatch):
    f = Path(__file__).parent / "fixtures" / "tones.wav"
    data, sr = sf.read(f, dtype="float32", always_2d=True)
    first = session.infer(f)
    model = session._model
    reference = weakref.ref(model)
    session.load()
    assert session._model is model
    array = session.infer(data.T, sample_rate=sr)
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
        session.infer(f)
    session.load()
    again = session.infer(f)
    np.testing.assert_array_equal(again.patch_embeddings, first.patch_embeddings)
    assert again.scores == first.scores
    session.close()
    assert session.status == "closed"
    json.dumps(first.to_dict(), allow_nan=False)
