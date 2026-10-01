# beats-audioset-infer maintainer guide

Standalone BEATs AudioSet tagging and audio representation, never beat tracking. Research and original code belong to Microsoft and Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen and Furu Wei. Code MIT; checkpoint grant absent/unclear. Third-party cpt2 provenance cannot be verified against official bytes, because actual official attempts returned403. Repository stays private, no public releases/org HF rehosting/upstream issue without explicit authorization.

Inference-only boundary: model preprocessing, encoder/predictor, input normalization, result conversion, download/cache and session lifecycle. No FAIRSEQ dependency, VQGAN tokenizer, training, backward helpers, stochastic layer-drop, quantization-noise hooks, datasets/evaluation or privateaudio. Keep native classifier index order and checkpoint key names. Predictor averages logits then sigmoid. Embeddings are contextual pre-predictor patch tokens, not independent time frames; clip embedding is explicitly derived arithmetic mean.

Every load-bearing file has a title/rationale/Reads docstring. Private core hides checkpoint math; checkpoint config owns artifact identity; resolver owns cache/download and validated generic JSON registry overrides; session owns model/device/lifecycle; audio owns all input normalization; result owns portable serialization. No shared runtime dependency across sibling packages.

Verification commands from checkout:

```bash
python -m pip install '.[dev]'
pytest -q
BEATS_TEST_CHECKPOINT=/path/to/model.pt BEATS_TEST_DEVICE=cpu pytest -q
BEATS_TEST_CHECKPOINT=/path/to/model.pt BEATS_TEST_DEVICE=cuda pytest -q
python -m ruff check src tests tools
python -m build
python -m pip install --force-reinstall --no-deps dist/beats_audioset_infer-0.1.0-py3-none-any.whl
python -I tools/verify_installed_e2e.py --checkpoint-dir /path/to/files --device cpu
```

The explicit checkpoint directory must also contain verified `class_labels_indices.csv`. Tests never implicitly download. Regenerate baseline with tools/capture_baseline.py against an untouched upstream reference, supplying --upstream, --checkpoint, --fixtures tests/fixtures and --device cpu/cuda. Baselinecommit precedes core extraction; CPU/CUDA original repeatability and extracted outputs proven exact on recorded environment. See docs/verification.md for measured status and structural audit.

Public openmirlab-skills capability map inspected: private/unreleased repositories are explicitly excluded. No public usage entry added while this package is private. Parent owns corresponding engineering radar/status record.

Never add weights, privateinputs, userpaths, rawhost reports, credentialfiles, environment/pipfreezelocalURLs or caches to Git. Inspect tracked files plus wheel/sdist contents before push. .gitignore cannot remove existinghistory. Only CC0generated audio and necessary compactnumeric baseline fixtures are eligible here.

Local completion evidence (2026-10-01): 39tests passed on Python3.11 CPU, Python3.11 CUDA, and Python3.10 CPU; default no-checkpoint run32passed/7explicitmodelskips. Installed-wheel public API+CLI NPZ E2E passed on allthree environments, with sockets blocked and completeembeddings exported. Fresh package-managed download verified bothartifact hashes. Wheel19members and sdist41members inspected before finaldoc rebuild: no weights/privateaudio/env/cache. Exact numerical claims restricted to recorded Torch2.7.1+cu126 baseline; Torch2.7.1+cpu checks passed documented1e-4tolerance. Retained torch.nn.utils.weight_norm emits an upstream deprecation warning; replacing it withoutcheckpointkey/parityproof is outofscope.


Hardware regression correction: hosted tonal fbank/token gold comparison exposed CPU-dependent upstream window arithmetic. NO AVX reproduction showed bit-exact original/extracted model math; replaying AVX2 pre-FFT window into the original NO AVX path closes the frontend discrepancy to zero. Original arrays preserved, recording hardware metadata added; no tolerance increase. Unknown hosts require checksum-pinned original same-host proof before BEATS_TEST_BASELINE-selected real tests/E2E. CI enforces this. See docs/hardware-parity.md; verification source/tools stay outside wheel, generated references/results stay under ignored local/.

```bash
python tools/verify_upstream_parity.py --checkpoint /path/to/model.pt --output local/same-host-baseline
BEATS_TEST_BASELINE=$PWD/local/same-host-baseline BEATS_TEST_CHECKPOINT=/path/to/model.pt pytest -q
BEATS_TEST_BASELINE=$PWD/local/same-host-baseline python -I tools/verify_installed_e2e.py --checkpoint-dir /path/to/files --device cpu
```

CLI rejects colliding resolved JSON/NPZ paths before model loading or touching files, preserving existing output bytes.
