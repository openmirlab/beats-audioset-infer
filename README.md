# beats-audioset-infer

Bidirectional Encoder representation from Audio Transformers (BEATs) is an audio representation model, **not beat tracking**. This standalone package provides AudioSet tagging and encoder patch embeddings from the AudioSet-fine-tuned iter3+ AS2M cpt2 model.

[![Code license: MIT AND BSD-2-Clause](https://img.shields.io/badge/code-MIT%20AND%20BSD--2--Clause-blue.svg)](LICENSE)

## Why this exists

[Microsoft's original BEATs](https://github.com/microsoft/unilm/tree/master/beats) introduced audio pre-training with acoustic tokenizers. This package extracts the inference path into a namespaced, reusable PyTorch session without FAIRSEQ or training tools. The upstream model code and research are the foundation of this work.

**Model code is MIT; retained TorchAudio preprocessing is BSD-2-Clause. The source repository is public, but checkpoint rights remain unresolved; this package neither bundles nor rehosts weights and has no PyPI release.** Official OneDrive links were unavailable from our test environment; third-party mirror identity cannot be verified against official bytes.

## Acknowledgments

**ALL MODEL AND RESEARCH CREDIT BELONGS TO THE ORIGINAL BEATs AUTHORS. PLEASE SUPPORT AND CITE THEIR WORK.**

- [Microsoft](https://github.com/microsoft), and BEATs authors Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen and Furu Wei, named in the [original paper](https://arxiv.org/abs/2212.09058).
- [Original implementation: microsoft/unilm/beats](https://github.com/microsoft/unilm/tree/master/beats), pinned source `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd`.
- [FAIRSEQ](https://github.com/facebookresearch/fairseq), whose code underlies the retained transformer components; copyright and license are preserved in [NOTICE](NOTICE) and [FAIRSEQ_LICENSE](docs/FAIRSEQ_LICENSE).
- The upstream also acknowledges [VQGAN](https://github.com/CompVis/taming-transformers). Its tokenizer/quantizer is not needed or shipped here.
- [WeiChihChen's checkpoint mirror](https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2): currently required for acquisition; this attribution does not establish licensing or original-file authenticity.
- [Google AudioSet](https://research.google.com/audioset/): the 527-class label mapping.
- [TorchAudio](https://github.com/pytorch/audio/tree/95c61b4168fc5133be8dd8c1337d929d066ae6cf): its original pure PyTorch sinc-resampling and Kaldi-filterbank functions are retained under [BSD-2-Clause](docs/TORCHAUDIO_LICENSE), without depending on the TorchAudio binary runtime.

## Citation

Copied from the upstream README:

```bibtex
@article{Chen2022beats,
  title = {BEATs: Audio Pre-Training with Acoustic Tokenizers},
  author = {Sanyuan Chen and Yu Wu and Chengyi Wang and Shujie Liu and Daniel Tompkins and Zhuo Chen and Furu Wei},
  eprint = {2212.09058},
  archivePrefix = {arXiv},
  year = {2022}
}
```

## Features

- One encoder pass returns all 527 native AudioSet scores and 768-dimensional contextual patch-token embeddings.
- NumPy CPU arrays remain usable after model release; additional derived mean-pooled clip embedding.
- Files, NumPy arrays and Torch tensors share one input normalizer.
- CPU and CUDA devices, explicit reusable lifecycle, pinned downloads, SHA256 verification and offline cache.
- Complete JSON results or NPZ embedding artifacts through the `beats-audioset-infer` CLI.

## Scope

Only iter3+ AS2M AudioSet-fine-tuned cpt2 is supported. This is audio representation/tagging, not musical beat, onset or tempo tracking. Native sigmoid scores are not calibrated confidence. There is no model quality improvement, silence-specific behavior, hidden segmentation or duration clipping.

Training, evaluation frameworks, dataset loaders, pre-training tokenizers, VQGAN, translation, summarization and fine-tuning remain out of scope. Single clips are unpadded by default; short-input padding is explicit and variable-length batching is not exposed.

## Install

Python 3.10+; dependencies have version floors. Verified with Python 3.11/Torch 2.7.1/NumPy 2.2.6 on CPU and CUDA, Python 3.13/Torch 2.13 on CPU, and an Ubuntu 24.04 CUDA 13.0 container with Python 3.13.15/Torch 2.13.0+cu130/NumPy 2.5.3 on an RTX 4090. Install the appropriate Torch build for your CPU/CUDA platform first. TorchAudio is not a runtime dependency: only the used pure PyTorch resampling and Kaldi-filterbank dependency closures are included, with original arithmetic and licensing preserved.

This public source package is not on PyPI. Install it from GitHub or from a checkout:

```bash
python -m pip install "beats-audioset-infer @ git+https://github.com/openmirlab/beats-audioset-infer"
python -m pip install .
```

Or build and install its wheel:

```bash
python -m pip install build
python -m build
python -m pip install dist/beats_audioset_infer-0.1.0-py3-none-any.whl
```

## Quick Start

```python
from beats_audioset_infer import BEATsAudioSetSession

with BEATsAudioSetSession(device="auto") as session:
    session.load()
    result = session.infer("audio.wav")
    print(result.ranked_tags[:5])
    print(result.patch_embeddings.shape)  # [time_patches * 8, 768]
    print(result.clip_embedding.shape)    # [768], additional arithmetic mean
    session.release()
    session.load()                       # reload supported
```

Construction/context entry performs no download or cold start. `infer()` requires `ready`. `load()` is idempotent; repeated `infer()` reuses the resident model. `release()` drops the session-owned model and permits reload. `close()` is terminal/idempotent. `status` reports `new`, `loading`, `ready`, `busy`, `failed`, `released` or `closed`. Operations are serialized per session; separate sessions own independent models. The package does not promise driver context or all allocator workspace returns to zero; service process teardown is separate.

## Inputs and devices

Paths use SoundFile metadata; omit `sample_rate` for files. Bare waveforms require a positive integer sample rate and shape `[samples]` or `[channels, samples]`, with real finite audio amplitudes conventionally in `[-1, 1]`. A stereo NumPy array from SoundFile must be transposed from `[samples, channels]`.

```python
import numpy as np
from beats_audioset_infer import BEATsAudioSetSession
with BEATsAudioSetSession(device="cpu") as session:
    session.load()
    result = session.infer(np.zeros(16000, dtype=np.float32), sample_rate=16000)
```

Channels are resampled to 16 kHz using the retained original TorchAudio 2.7.1 pure PyTorch sinc-resampling function, then averaged to mono. This is the input convention validated against our upstream baseline; upstream `extract_features` itself expects prepared mono 16 kHz input. By default, inputs with fewer than 2,800 normalized samples (0.175 seconds) cannot form one full patch and raise before encoder execution. Longer audio is processed directly; no deployment duration ceiling or quality/resource certification is claimed.

For short, nonempty audio, opt in with `session.infer(audio, sample_rate=16000, pad_short=True)` or CLI `--pad-short`. After the existing resampling and channel averaging, the normalized waveform is right-zero padded to 2,800 samples; empty or non-finite input still fails. The option must be an actual Python `bool`. This changes the model input and is not a claim of short-clip recognition quality. Scores and embeddings match explicitly providing that same padded normalized waveform. `result.audio` retains original samples and duration, and records `normalized_samples` before padding, `model_samples`, `padding_samples` and `padding_seconds`. Native encoder patch tokens represent the padded waveform, including its added silence; `embedding_info.padding` makes that explicit. The derived clip embedding still averages all native tokens without normalization or masking.

Devices: `auto` chooses CUDA 0 when available, otherwise CPU; `cpu`, `cuda` and `cuda:N` are honored. An unavailable explicit CUDA device raises. MPS/MLX are unsupported. Global precision flags are left to the caller; there is no implicit mixed precision or TF32 modification.

## Scores and embedding semantics

`result.scores`, `labels` and `mids` preserve **native classifier index order**, which differs from Google AudioSet CSV order. For example native index 2 is `/m/04rlf`, Music. `ranked_tags` sorts descending by score, resolving exact ties by native index. All 527 values are retained. Upstream prediction averages per-token logits, then applies sigmoid; it does not average probabilities.

`patch_embeddings` is the final encoder output immediately before the predictor, shape `[time_patches * 8, 768]`, float32. Order is time-major and frequency-minor. The model uses 128 mel bins with 16×16 patches: 8 frequency patches per time patch. Kaldi fbank uses 25 ms windows and 10 ms hop; each time patch spans 175 ms of input support and advances 160 ms. Incomplete final patches are dropped. These tokens are contextual after attention and should not be read as independently localized events or frame-level class predictions.

`clip_embedding` is **additional arithmetic mean pooling across all native patch tokens**, not a separately trained native clip head and not L2-normalized. `embedding_info` includes axes, shapes, dtype and preprocessing geometry. A single unpadded clip has no padded tokens; all returned tokens are valid.

`audio` reports original metadata and actual analysis rate/channels. `checkpoint` includes revision, source revision, hashes, URLs and licensing. `result.to_dict()` serializes full arrays to JSON-compatible lists. `to_dict(include_embeddings=False)` explicitly marks that arrays were omitted; arrays remain accessible in the result.

## CLI and complete artifacts

```bash
beats-audioset-infer audio.wav --device cpu --output result.json
beats-audioset-infer audio.wav --device cuda:0 --output result.json --embeddings-output embeddings.npz
```

Default JSON contains **all scores and embeddings**, without truncation. JSON and NPZ output paths must differ; identical resolved paths are rejected before model loading or writing. `--embeddings-output` writes full `patch_embeddings` and `clip_embedding` arrays to NPZ; JSON includes their metadata and the artifact path, with omission explicitly marked. NPZ is not a pickle:

```python
import numpy as np
with np.load("embeddings.npz", allow_pickle=False) as data:
    patches = data["patch_embeddings"]
    clip = data["clip_embedding"]
```

`--cache-dir`, `--checkpoint-dir`, `--checkpoint-config`, `--offline` match the session arguments. Results are informational model outputs, not calibrated decisions.

## Checkpoints and offline use

A generic `checkpoint_config="/path/to/registry.json"` (CLI `--checkpoint-config`) overrides the registry, including artifact URLs and expected hashes, after schema validation. It must retain the selected compatible model architecture and native label contract; overrides do not bypass verification.

The default registry is package-owned [config/checkpoints.json](src/beats_audioset_infer/config/checkpoints.json). First explicit `load()` downloads missing managed artifacts atomically and verifies size plus SHA256; existing corruption raises instead of silent replacement. Concurrent acquisition uses a file lock. `cache_info()` is read-only, uses the same path resolver, never creates directories/downloads, and reports integrity as `not_checked` until actual loading verifies bytes.

After successful `load()`, `session.loaded_checkpoint_info()` returns a defensive, JSON-compatible snapshot of the verified resident checkpoint. It includes `model_id="beats-audioset"`, `checkpoint_id="<repo>@<revision>"`, the primary weight's `checkpoint_sha256`, the resolved `device`, and original provenance and artifact identities with local paths removed. This memory-only query does not download, hash files or inspect the cache; it describes the loaded model even if registry metadata subsequently changes. It raises `RuntimeError` unless the session is ready, including after `release()` or `close()`. Loading again after release establishes a new snapshot. This verifies artifact bytes against the registry, not the mirror's original identity or its authorization.

Default root: `~/.cache/beats-audioset-infer/`; files are under revision `172e4071a2912392d9151c70750b9a12f55fc54e`. Override through `cache_dir=` or `BEATS_AUDIOSET_INFER_CACHE`. A read-only manual folder uses `checkpoint_dir=`; missing explicit model bytes are not substituted automatically.

Manual download URLs:

- [Pinned third-party cpt2](https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2/resolve/172e4071a2912392d9151c70750b9a12f55fc54e/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt), save as `model.pt`, 363,145,291 bytes; SHA256 `e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c`.
- [Google AudioSet labels](https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv), save as `class_labels_indices.csv`, 14,675 bytes; SHA256 `cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429`.

```python
from beats_audioset_infer import BEATsAudioSetSession
with BEATsAudioSetSession(device="cpu", checkpoint_dir="/path/to/files", offline=True) as session:
    session.load()
    result = session.infer("audio.wav")
```

`torch.load(..., weights_only=True)` accepts the selected checkpoint's ordinary config dictionary, tensors and index-to-MID label dictionary; arbitrary pickle fallback is never enabled. Strict model state loading and label bijection validation are mandatory.

[Official-source verification](docs/provenance.md) records actual cpt1/cpt2 download attempts. The official source could not be retrieved in this environment; selected bytes come from the third-party mirror above, and **their original provenance cannot be authenticated**. Checkpoint authorization is unresolved; code MIT is a separate fact. No permission is inferred from absent metadata.

## What this project will NEVER bundle

Model checkpoints, private audio, datasets, training/evaluation frameworks, virtual environments, credentials or local benchmark reports. Deterministic CC0-generated fixtures and compact original-output regressions are committed; full checkpoints stay outside Git.

## Development

```bash
python -m pip install '.[dev]'
pytest -q
BEATS_TEST_CHECKPOINT=/path/to/model.pt BEATS_TEST_DEVICE=cpu pytest -q
BEATS_TEST_CHECKPOINT=/path/to/model.pt BEATS_TEST_DEVICE=cuda pytest -q
python -m build
```

Place `class_labels_indices.csv` beside the model for offline real-session tests. Default tests do not download weights; real model tests explicitly skip unless a checkpoint is supplied. Three synthetic fixtures cover silence, changing tones/stereo 44.1 kHz and noise/22.05 kHz; CPU and CUDA baselines were captured from unmodified upstream before extraction, with repeated runs proving reproducibility. Full waveform, fbank, scores and encoder tokens are compared. Bit-exact comparisons require the recorded Torch build and hardware/CPU dispatch. On different hardware, run the checksum-verified original/extracted same-host comparison first and select its local baseline with `BEATS_TEST_BASELINE`; portable waveform and scores keep their existing tolerances. No tolerance is raised to hide nonportable frontend/token differences. See [hardware parity](docs/hardware-parity.md) for the demonstrated window arithmetic/log-fbank cause and exact commands. [Verification record](docs/verification.md) separates actual local checks from CI coverage.

Developer-only reference tools `capture_baseline.py`, `capture_audio_frontend.py` and `verify_upstream_parity.py` require the original TorchAudio 2.7.1 reference in a separate matching Torch 2.7.1 environment. They are not required for package use or ordinary tests. Optional live-reference tests skip explicitly when that reference is absent; committed original frontend arrays retain their hardware/build guards. `capture_audio_frontend.py` refuses to overwrite the recorded original arrays.

The installed wheel also passed its full checkpoint-enabled GPU tests and API/CLI E2E in the CUDA 13.0 container with TorchAudio imports blocked, including full scores/embeddings, short-input padding, loaded provenance and release/reload. Synthetic Gaussian inputs of 1, 10, 30, 60 and 120 seconds executed with finite scores. These checks establish execution and lifecycle behavior, not recognition quality, a supported maximum duration or a service resource budget. The [verification record](docs/verification.md) includes the exact environment and measured Torch memory counters.

## License

Model/package integration code: MIT, with Microsoft and openmirlab copyright. FAIRSEQ notice retained. Original TorchAudio preprocessing: BSD-2-Clause, Copyright (c) 2017 Facebook Inc. (Soumith Chintala); full notice shipped with the package. Combined code license expression: `MIT AND BSD-2-Clause`. **Weights: no explicit license located; unresolved.** AudioSet dataset metadata is CC BY 4.0; ontology CC BY-SA 4.0. Numeric golden traces derive from the selected model and do not establish additional rights to its checkpoint. Public source availability does not establish permission to redistribute the checkpoint.

## Support

Use this repository's [issue tracker](https://github.com/openmirlab/beats-audioset-infer/issues) for package bugs. Original model research belongs to [Microsoft BEATs](https://github.com/microsoft/unilm/tree/master/beats). No upstream licensing issue has been opened automatically.
