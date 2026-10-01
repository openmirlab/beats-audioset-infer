# Hardware-qualified numerical regression

The first hosted real-model CI run passed waveform/527scores and all offline contracts, but failed the tonal signal's recorded fbank (`0.00158429` maximum absolute difference) and encoder tokens (`0.00032401`). Those fixtures were recorded on a different CPU. These failures were investigated rather than accepted by increasing tolerances.

## Reproduction and closure proof

Changing only Torch CPU dispatch on one machine from AVX2 to NO AVX (`ATEN_CPU_CAPABILITY=default`) reproduced the same failure class in the unmodified original BEATs implementation, Torch2.7.1+cpu:

| Stage, tonal signal | Maximum absolute difference |
| --- | ---: |
| Resampled mono waveform | 0 (bit-exact) |
| Kaldi window tensor before FFT | 0.00012207031 |
| Complex FFT | 0.0078125 |
| Raw log-fbank | 0.0043940544 |
| BEATs-normalized fbank | 0.0003350973 |
| Native original encoder tokens | 0.0001433194 |
| Original 527scores | 0.0000089854 |

At the largest log-fbank difference, mel energies were `0.08089802` versus `0.08054333`. Narrow-band tones have low-energy bins where small pre-FFT window arithmetic differences are amplified by the logarithm. Reusing the saved identical mono waveform did not remove the difference, ruling out resampling in this reproduction. **Replaying the AVX2 pre-FFT window into the NO AVX original Kaldi path reduced the complete raw log-fbank difference to exactly zero.** Replaying the reference FFT or mel energies also gave zero. This isolates the difference to CPU-dispatched preprocessing before FFT; it does not identify or claim a specific ATen internal instruction or operation.

The extracted core matched the unmodified original bit-for-bit on both dispatch settings, across every waveform, fbank, 527score and 768-token array. This proves the tested difference is environment-dependent upstream arithmetic, not introduced model math. The larger hosted-run value still needs the hosted same-host comparison; local reproduction is not a claim to know its exact CPU instruction path.

## Verification contract

No tolerance was increased. Original recorded numeric fixture arrays remain unchanged. Their metadata now records the actual recording CPU/model/dispatch or GPU/precision flags. On matching hardware, existing exact/specified-build tolerance checks continue.

On a different host, `tools/verify_upstream_parity.py` must run first. It downloads only three original inference source files to an ignored reference directory, pins the original Git revision, checks each source SHA256, checks checkpoint SHA256 and compares original/extracted **bit-exactly at every stage**. It reports all drift against permanent recorded fixtures and separately replays their saved waveform through the original frontend. Portable waveform and 527score bounds remain rtol1e-4 and atol1e-6/1e-4 respectively.

Only after that strict comparison passes are new host-specific original fixtures saved in the explicitly local output directory. `BEATS_TEST_BASELINE` selects those verified fixtures for public facade and installed-wheel E2E. A different hardware fingerprint or alternate baseline without verified reference provenance fails loudly; no unexplained skip or widened tolerance can hide changed model output. The local baseline is not committed or distributed.

```bash
python tools/verify_upstream_parity.py --checkpoint /path/to/model.pt --output local/same-host-baseline
BEATS_TEST_BASELINE=$PWD/local/same-host-baseline BEATS_TEST_CHECKPOINT=/path/to/model.pt pytest -q
BEATS_TEST_BASELINE=$PWD/local/same-host-baseline python -I tools/verify_installed_e2e.py --checkpoint-dir /path/to/files --device cpu
```

Hosted real-model CI runs this same-host proof before regression and installed-wheel CLI/API checks. No training/runtime dependency on upstream code was added; reference tooling never ships in the wheel. Model math, weights, defaults, label order and full embeddings are unchanged.
