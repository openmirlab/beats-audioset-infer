"""Installed-wheel public API plus CLI full artifact and lifecycle E2E.

Run python -I to exclude the checkout from import resolution. All networking is
blocked before loading; golden fixtures are explicit. Reads: installed package.
"""

import argparse
import json
import socket
import sys
import tempfile
import weakref
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import beats_audioset_infer

p = argparse.ArgumentParser()
p.add_argument("--checkpoint-dir", required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
assert "site-packages" in beats_audioset_infer.__file__, beats_audioset_infer.__file__
torch.set_num_threads(4)


def blocked(*args, **kwargs):
    raise AssertionError("Offline E2E attempted network access")


socket.create_connection = blocked
socket.socket.connect = blocked
f = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
from beats_audioset_infer import BEATsAudioSetSession

s = BEATsAudioSetSession(device=a.device, checkpoint_dir=a.checkpoint_dir, offline=True)
assert s.status == "new"
s.cache_info()
s.load()
for name in ["noise", "silence", "tones"]:
    r = s.infer(f / (name + ".wav"))
    g = np.load(f / (name + "_" + a.device + ".npz"))
    environment = json.loads((f / ("environment_" + a.device + ".json")).read_text())
    compare = (
        np.testing.assert_array_equal
        if torch.__version__ == environment["torch"]
        else lambda x, y: np.testing.assert_allclose(x, y, rtol=1e-4, atol=1e-4)
    )
    compare(r.scores, g["scores"][0])
    compare(r.patch_embeddings, g["tokens"][0])
    assert len(r.mids) == 527 and r.mids[2] == "/m/04rlf" and r.labels[2] == "Music"
    json.dumps(r.to_dict(), allow_nan=False)
    x, sr = sf.read(f / (name + ".wav"), dtype="float32", always_2d=True)
    rr = s.infer(x.T, sample_rate=sr)
    np.testing.assert_array_equal(rr.patch_embeddings, r.patch_embeddings)
    rr = s.infer(torch.from_numpy(x.T.copy()), sample_rate=sr)
    assert rr.scores == r.scores
ref = weakref.ref(s._model)
s.release()
assert ref() is None
s.load()
r = s.infer(f / "tones.wav")
s.close()
assert s.status == "closed"
json.dumps(r.to_dict(), allow_nan=False)
from beats_audioset_infer.cli import main

with tempfile.TemporaryDirectory() as td:
    output = Path(td) / "result.json"
    artifact = Path(td) / "embeddings.npz"
    sys.argv = [
        "beats-audioset-infer",
        str(f / "tones.wav"),
        "--device",
        a.device,
        "--checkpoint-dir",
        a.checkpoint_dir,
        "--offline",
        "--output",
        str(output),
        "--embeddings-output",
        str(artifact),
    ]
    assert main() == 0
    data = json.loads(output.read_text())
    assert not data["embeddings_included"] and len(data["scores"]) == 527
    with np.load(artifact, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["patch_embeddings"], r.patch_embeddings)
        np.testing.assert_array_equal(arrays["clip_embedding"], r.clip_embedding)
print(
    "Installed wheel API/file/array/tensor/full scores/embeddings/JSON/CLI NPZ/offline/release/reload/close E2E passed:",
    a.device,
)
