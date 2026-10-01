"""Native model regression against pre-extraction original runs. Reads: fixtures."""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from parity_support import baseline_root

from beats_audioset_infer._impl.model import BEATs, BEATsConfig
from beats_audioset_infer._impl.resample import resample


@pytest.fixture(scope="module")
def model():
    checkpoint = os.environ.get("BEATS_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set BEATS_TEST_CHECKPOINT for real model regression")
    device = os.environ.get("BEATS_TEST_DEVICE", "cpu")
    torch.set_num_threads(4)
    d = torch.load(checkpoint, weights_only=True, map_location="cpu")
    m = BEATs(BEATsConfig(d["cfg"])).eval().to(device)
    m.load_state_dict(d["model"], strict=True)
    return m, device


@pytest.mark.parametrize("name", ["noise", "silence", "tones"])
def test_native_parity(model, name):
    m, device = model
    f = Path(__file__).parent / "fixtures"
    data, sr = sf.read(f / (name + ".wav"), dtype="float32", always_2d=True)
    x = torch.from_numpy(data.T).to(device)
    if sr != 16000:
        x = resample(x, sr, 16000)
    x = x.mean(0, keepdim=True)
    gold_dir, environment = baseline_root(f, device)
    gold = np.load(gold_dir / (name + "_" + device + ".npz"))
    with torch.inference_mode():
        fb = m.preprocess(x)
        scores, mask, tokens = m.extract_features(x)
    assert mask is None
    for key, actual in [
        ("waveform", x),
        ("fbank", fb),
        ("scores", scores),
        ("tokens", tokens),
    ]:
        if torch.__version__ == environment["torch"]:
            np.testing.assert_array_equal(actual.cpu().numpy(), gold[key])
        else:
            np.testing.assert_allclose(
                actual.cpu().numpy(),
                gold[key],
                rtol=1e-4,
                atol=1e-6 if key == "waveform" else 1e-4,
            )
