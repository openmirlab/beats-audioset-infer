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


class NoTorchAudio:
    """Prove the installed runtime cannot fall back to a binary audio dependency."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torchaudio" or fullname.startswith("torchaudio."):
            raise AssertionError("Installed runtime attempted a torchaudio import")


sys.meta_path.insert(0, NoTorchAudio())

import beats_audioset_infer

p = argparse.ArgumentParser()
p.add_argument("--checkpoint-dir", required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
assert "site-packages" in beats_audioset_infer.__file__, beats_audioset_infer.__file__
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_support import baseline_root

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
identity = s.loaded_checkpoint_info()
assert identity["model_id"] == "beats-audioset"
assert identity["device"] == s.device
assert all("path" not in artifact for artifact in identity["files"].values())
json.dumps(identity, allow_nan=False)
for name in ["noise", "silence", "tones"]:
    r = s.infer(f / (name + ".wav"))
    assert identity["checkpoint_sha256"] == r.checkpoint["files"]["model.pt"]["sha256"]
    gold_dir, environment = baseline_root(f, a.device)
    g = np.load(gold_dir / (name + "_" + a.device + ".npz"))
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
try:
    s.loaded_checkpoint_info()
except RuntimeError:
    pass
else:
    raise AssertionError("Released session retained loaded identity")
s.load()
assert s.loaded_checkpoint_info() == identity
r = s.infer(f / "tones.wav")
short_audio = np.sin(np.arange(1600, dtype=np.float32) * 0.04)
short_result = s.infer(short_audio, sample_rate=16000, pad_short=True)
explicit_short = s.infer(np.pad(short_audio, (0, 1200)), sample_rate=16000)
assert short_result.scores == explicit_short.scores
np.testing.assert_array_equal(
    short_result.patch_embeddings, explicit_short.patch_embeddings
)
np.testing.assert_array_equal(
    short_result.clip_embedding, explicit_short.clip_embedding
)
assert short_result.audio["padding_samples"] == 1200
assert short_result.audio["samples"] == 1600
s.close()
assert s.status == "closed"
try:
    s.loaded_checkpoint_info()
except RuntimeError:
    pass
else:
    raise AssertionError("Closed session retained loaded identity")
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
    short_file = Path(td) / "short.wav"
    sf.write(short_file, short_audio, 16000, subtype="FLOAT")
    sys.argv[1] = str(short_file)
    sys.argv.append("--pad-short")
    assert main() == 0
    short_data = json.loads(output.read_text())
    assert short_data["scores"] == list(short_result.scores)
    assert short_data["audio"]["padding_samples"] == 1200
    with np.load(artifact, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(
            arrays["patch_embeddings"], short_result.patch_embeddings
        )
        np.testing.assert_array_equal(
            arrays["clip_embedding"], short_result.clip_embedding
        )
print(
    "Installed wheel API/file/array/tensor/full scores/embeddings/JSON/CLI NPZ/offline/release/reload/close E2E passed:",
    a.device,
)
