"""Explicit short-input padding follows the existing normalization boundary.

Reads: native 16 kHz mono input contract; no model or checkpoint needed.
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from beats_audioset_infer import BEATsAudioSetSession
from beats_audioset_infer._impl.audio import normalize
from beats_audioset_infer._impl.resample import resample


@pytest.mark.parametrize("samples", [2799, 2800, 2801])
def test_padding_boundary(samples):
    audio = np.arange(samples, dtype=np.float32) / samples
    padded, info = normalize(audio, 16000, "cpu", pad_short=True)
    np.testing.assert_array_equal(padded[0, :samples], audio)
    assert info["normalized_samples"] == samples
    assert info["model_samples"] == max(samples, 2800)
    assert info["padding_samples"] == max(0, 2800 - samples)
    assert info["padding_seconds"] == max(0, 2800 - samples) / 16000
    assert info["samples"] == samples and info["duration_seconds"] == samples / 16000
    if samples < 2800:
        assert padded[0, -1] == 0
        with pytest.raises(ValueError, match="complete time patch"):
            normalize(audio, 16000, "cpu")
    else:
        default, default_info = normalize(audio, 16000, "cpu")
        torch.testing.assert_close(default, padded, rtol=0, atol=0)
        assert default_info == info


@pytest.mark.parametrize("rate", [16000, 32000, 44100])
@pytest.mark.parametrize("kind", ["array", "tensor", "file"])
def test_padding_after_resample_and_channel_average(rate, kind, tmp_path):
    samples = rate // 10
    audio = np.stack(
        [
            np.sin(np.arange(samples, dtype=np.float32) * 0.04),
            np.cos(np.arange(samples, dtype=np.float32) * 0.02),
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
    x, info = normalize(source, input_rate, "cpu", pad_short=True)
    expected = torch.from_numpy(audio)
    if rate != 16000:
        expected = resample(expected, rate, 16000)
    expected = expected.mean(0, keepdim=True)
    normalized_samples = expected.shape[1]
    expected = torch.nn.functional.pad(expected, (0, 2800 - normalized_samples))
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
    assert info["normalized_samples"] == normalized_samples
    assert info["samples"] == samples and info["channels"] == 2


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_nonbool_pad_short_rejected_before_input_io(value, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid option must fail before reading audio")

    monkeypatch.setattr(sf, "read", forbidden)
    with pytest.raises(TypeError, match="pad_short must be a bool"):
        normalize("nonexistent.wav", None, "cpu", pad_short=value)
    with pytest.raises(TypeError, match="pad_short must be a bool"):
        BEATsAudioSetSession(offline=True).infer("nonexistent.wav", pad_short=value)


@pytest.mark.parametrize("audio", [np.empty(0), np.array([np.nan]), np.array([np.inf])])
def test_padding_never_accepts_invalid_audio(audio):
    with pytest.raises(ValueError):
        normalize(audio, 16000, "cpu", pad_short=True)


def test_cli_forwards_explicit_padding(monkeypatch):
    import sys

    from beats_audioset_infer.cli import main

    seen = []

    def infer(self, audio, *, sample_rate=None, pad_short=False):
        seen.append((audio, pad_short))
        raise ValueError("stop before writing")

    monkeypatch.setattr(BEATsAudioSetSession, "load", lambda self: self)
    monkeypatch.setattr(BEATsAudioSetSession, "infer", infer)
    monkeypatch.setattr(
        sys, "argv", ["beats-audioset-infer", "short.wav", "--pad-short"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and seen == [("short.wav", True)]
