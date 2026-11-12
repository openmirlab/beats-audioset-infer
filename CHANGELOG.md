# Changelog

## Unreleased

- Make the GitHub repository public (2026-11-12) with checkpoint licensing still unresolved; no PyPI release and no org-hosted weights. Document installation from GitHub.

## 0.1.0 — unpublished

- Record installed-wheel CPU and CUDA Torch 2.13 verification, including an offline CUDA 13.0 container, full checkpoint tests/API/CLI lifecycle checks and synthetic-duration memory measurements; no quality, maximum-duration or service-budget certification is implied.
- Remove the TorchAudio runtime dependency by retaining only original pure PyTorch sinc resampling and Kaldi fbank dependency closures, verified against original frontend fixtures and live same-host outputs. Retain full BSD-2-Clause notice and declare combined `MIT AND BSD-2-Clause` code licensing; model math and tolerances are unchanged.
- Add explicit `pad_short=True` / CLI `--pad-short` for right-zero padding after native normalization, with original-duration and padding metadata; preserve the default short-input rejection and encoder arithmetic.
- Add ready-only `loaded_checkpoint_info()` for the resident verified checkpoint, returning defensive portable provenance without cache paths or filesystem access.
- Qualify recorded float goldens by hardware/CPU dispatch; require checksum-pinned original/extracted same-host strict parity on other hosts. Preserve original arrays and portable score/waveform tolerances without widening.
- Reject colliding JSON/NPZ output paths before loading or writing.

- Standalone inference-only BEATs AudioSet cpt2 scores and native768-dimensional patch embeddings in one encoder pass.
- Explicit session lifecycle, CPU/CUDA validation, checksum-verified downloads, offline local overrides and full JSON/NPZ CLI.
- Original CPU/CUDA regression fixtures recorded before extraction; code MIT separated from unresolved checkpoint license.
- Private implementation only; no public release or org-hosted weights.
