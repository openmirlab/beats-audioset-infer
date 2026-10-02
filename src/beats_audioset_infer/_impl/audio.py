"""One audio normalization owner for files and explicit-rate waveforms.

Resamples channels before mono averaging, then validates a minimum full patch
or explicitly right-zero pads short input when requested.
Reads: SoundFile metadata, caller arrays, concrete Torch device.
"""

from pathlib import Path

import numpy as np
import torch

from .resample import resample


def normalize(audio, sample_rate, device, *, pad_short=False):
    if type(pad_short) is not bool:
        raise TypeError("pad_short must be a bool")
    if isinstance(audio, (str, Path)):
        import soundfile as sf

        if sample_rate is not None:
            raise ValueError("File input uses its metadata; omit sample_rate")
        data, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
        x = torch.from_numpy(data.T)
    else:
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
        ):
            raise ValueError(
                "Waveform requires an explicit positive integer sample_rate"
            )
        if isinstance(audio, np.ndarray):
            if audio.dtype.kind not in "fiu":
                raise ValueError(
                    "Waveform must be real numeric audio, not complex or bool"
                )
            x = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
        elif isinstance(audio, torch.Tensor):
            x = audio.detach()
        else:
            raise TypeError("audio must be a path, NumPy array or Torch tensor")
        if x.ndim == 1:
            x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("Waveform shape must be [samples] or [channels, samples]")
    if x.is_complex() or x.dtype == torch.bool:
        raise ValueError("Waveform must be real numeric audio")
    x = x.to(device=device, dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise ValueError("Waveform contains non-finite values")
    info = {
        "sample_rate": sample_rate,
        "channels": x.shape[0],
        "samples": x.shape[1],
        "duration_seconds": x.shape[1] / sample_rate,
        "analysis_sample_rate": 16000,
        "analysis_channels": 1,
    }
    if sample_rate != 16000:
        x = resample(x, sample_rate, 16000)
    x = x.mean(0, keepdim=True)
    normalized_samples = x.shape[1]
    padding_samples = max(0, 2800 - normalized_samples) if pad_short else 0
    if x.shape[1] < 2800:
        if not pad_short:
            raise ValueError(
                "BEATs needs at least 0.175 seconds for one complete time patch"
            )
        x = torch.nn.functional.pad(x, (0, padding_samples))
    info.update(
        normalized_samples=normalized_samples,
        model_samples=x.shape[1],
        padding_samples=padding_samples,
        padding_seconds=padding_samples / 16000,
    )
    return x, info
