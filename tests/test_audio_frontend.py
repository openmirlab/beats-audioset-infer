"""Dependency extraction matches immutable originals and optional live reference.

Recorded CPU arrays are hardware/build qualified. Live original comparisons are
optional developer checks, never a runtime torchaudio requirement. Reads: CC0 data.
"""

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from parity_support import fingerprint

from beats_audioset_infer._impl import kaldi
from beats_audioset_infer._impl.resample import resample

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def original():
    try:
        import torchaudio
    except (ImportError, OSError) as error:
        pytest.skip(f"Optional torchaudio original reference unavailable: {error}")
    if torchaudio.__version__.split("+")[0] != "2.7.1":
        pytest.skip("Live extraction reference is pinned torchaudio 2.7.1")
    return torchaudio


def test_recorded_original_frontend():
    metadata = json.loads((FIXTURES / "audio_frontend_cpu.json").read_text())
    path = FIXTURES / "audio_frontend_cpu.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["fixture_sha256"]
    if metadata["torch"] != torch.__version__ or metadata["hardware"] != fingerprint(
        "cpu"
    ):
        pytest.skip(
            "Original frontend exact arrays require their recorded Torch build and CPU dispatch; use same-host reference A/B elsewhere"
        )
    with np.load(path, allow_pickle=False) as data:
        for rate in [16000, 32000, 44100, 48000]:
            actual = resample(
                torch.from_numpy(data[f"resample_input_{rate}"]), rate, 16000
            )
            np.testing.assert_array_equal(
                actual.numpy(), data[f"resample_output_{rate}"]
            )
        for name in ["silence", "noise", "tones"]:
            actual = kaldi.fbank(
                torch.from_numpy(data[f"fbank_input_{name}"]) * 2**15,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10,
            )
            np.testing.assert_array_equal(actual.numpy(), data[f"fbank_output_{name}"])


@pytest.mark.parametrize("rate", [16000, 32000, 44100, 48000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_live_original_resample(original, rate, dtype):
    source = torch.randn(
        (2, 1307), generator=torch.Generator().manual_seed(rate), dtype=dtype
    ).to(os.environ.get("BEATS_TEST_DEVICE", "cpu"))
    assert torch.equal(
        resample(source, rate, 16000), original.functional.resample(source, rate, 16000)
    )


@pytest.mark.parametrize("name", ["silence", "noise", "tones"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_live_original_fbank(original, name, dtype):
    with np.load(FIXTURES / "audio_frontend_cpu.npz", allow_pickle=False) as data:
        source = (
            torch.from_numpy(data[f"fbank_input_{name}"]).to(
                device=os.environ.get("BEATS_TEST_DEVICE", "cpu"), dtype=dtype
            )
            * 2**15
        )
    settings = dict(
        num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10
    )
    assert torch.equal(
        kaldi.fbank(source, **settings),
        original.compliance.kaldi.fbank(source, **settings),
    )


def test_extracted_functions_are_verbatim_reference_ast(original):
    from beats_audioset_infer._impl import resample as extracted

    for target, reference, names in [
        (
            extracted,
            original.functional.functional,
            ["_get_sinc_resample_kernel", "_apply_sinc_resample_kernel", "resample"],
        ),
        (
            kaldi,
            original.compliance.kaldi,
            [
                name
                for name, value in vars(kaldi).items()
                if inspect.isfunction(value) and value.__module__ == kaldi.__name__
            ],
        ),
    ]:
        for name in names:
            assert ast.dump(
                ast.parse(inspect.getsource(getattr(target, name)))
            ) == ast.dump(ast.parse(inspect.getsource(getattr(reference, name))))
