# Changelog

## 0.1.0 — unpublished

- Qualify recorded float goldens by hardware/CPU dispatch; require checksum-pinned original/extracted same-host strict parity on other hosts. Preserve original arrays and portable score/waveform tolerances without widening.
- Reject colliding JSON/NPZ output paths before loading or writing.

- Standalone inference-only BEATs AudioSet cpt2 scores and native768-dimensional patch embeddings in one encoder pass.
- Explicit session lifecycle, CPU/CUDA validation, checksum-verified downloads, offline local overrides and full JSON/NPZ CLI.
- Original CPU/CUDA regression fixtures recorded before extraction; code MIT separated from unresolved checkpoint license.
- Private implementation only; no public release or org-hosted weights.
