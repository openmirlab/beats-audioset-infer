"""Capture immutable CC0 signal outputs before replacing the audio dependency.

Requires optional developer-only torchaudio 2.7.1 reference. Records raw original
resampling and Kaldi filterbanks with hardware guards. Reads: CC0 fixtures.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from parity_support import fingerprint

assert torchaudio.__version__.split("+")[0] == "2.7.1"
torch.set_num_threads(4)
fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
arrays = {}
for rate in [16000, 32000, 44100, 48000]:
    source = (
        np.random.default_rng(rate)
        .normal(0, 0.1, (2, rate // 10 + 11))
        .astype(np.float32)
    )
    arrays[f"resample_input_{rate}"] = source
    arrays[f"resample_output_{rate}"] = torchaudio.functional.resample(
        torch.from_numpy(source), rate, 16000
    ).numpy()
for name in ["silence", "noise", "tones"]:
    audio, rate = sf.read(fixtures / f"{name}.wav", dtype="float32", always_2d=True)
    waveform = torchaudio.functional.resample(
        torch.from_numpy(audio.T), rate, 16000
    ).mean(0, keepdim=True)
    arrays[f"fbank_input_{name}"] = waveform.numpy()
    arrays[f"fbank_output_{name}"] = torchaudio.compliance.kaldi.fbank(
        waveform * 2**15,
        num_mel_bins=128,
        sample_frequency=16000,
        frame_length=25,
        frame_shift=10,
    ).numpy()
path = fixtures / "audio_frontend_cpu.npz"
assert not path.exists(), "Never overwrite a captured original baseline"
np.savez_compressed(path, **arrays)
environment = {
    "torch": torch.__version__,
    "torchaudio": torchaudio.__version__,
    "numpy": np.__version__,
    "hardware": fingerprint("cpu"),
    "source_revision": "95c61b4168fc5133be8dd8c1337d929d066ae6cf",
    "source_sha256": {
        "functional.py": hashlib.sha256(
            Path(torchaudio.functional.functional.__file__).read_bytes()
        ).hexdigest(),
        "kaldi.py": hashlib.sha256(
            Path(torchaudio.compliance.kaldi.__file__).read_bytes()
        ).hexdigest(),
    },
    "license": "CC0-1.0 generated inputs; original torchaudio 2.7.1 outputs",
    "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}
(fixtures / "audio_frontend_cpu.json").write_text(
    json.dumps(environment, indent=2) + "\n"
)
print("Captured original frontend before extraction:", path.name)
