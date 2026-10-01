"""Strict original/extracted same-host A/B with pinned, checksum-verified source.

Reference source and generated outputs stay outside shipped package data. This
also reports cross-host golden drift and a saved-waveform frontend counterfactual.
Reads: immutable upstream files, explicit checkpoint and committed CC0 signals.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import torch
import torchaudio
from parity_support import SOURCE_HASHES, SOURCE_REVISION, fingerprint

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--device", default="cpu")
p.add_argument("--upstream", default="local/upstream-reference")
p.add_argument("--output", default="local/same-host-baseline")
a = p.parse_args()
root = Path(a.upstream)
root.mkdir(parents=True, exist_ok=True)
for name, checksum in SOURCE_HASHES.items():
    path = root / name
    if not path.exists():
        with urlopen(
            f"https://raw.githubusercontent.com/microsoft/unilm/{SOURCE_REVISION}/beats/{name}",
            timeout=60,
        ) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError("Unexpected original source hash: " + name)
        path.write_bytes(data)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError("Reference source modified: " + name)
# No source vendoring or source edits: import only the immutable external reference.
sys.path.insert(0, str(root.resolve()))
from BEATs import BEATs as OriginalBEATs
from BEATs import BEATsConfig as OriginalConfig

sys.path.pop(0)
from beats_audioset_infer._impl.model import BEATs, BEATsConfig

torch.set_num_threads(4)
data = torch.load(a.checkpoint, weights_only=True, map_location="cpu")
assert (
    hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    == "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c"
)
original = OriginalBEATs(OriginalConfig(data["cfg"])).eval().to(a.device)
original.load_state_dict(data["model"], strict=True)
extracted = BEATs(BEATsConfig(data["cfg"])).eval().to(a.device)
extracted.load_state_dict(data["model"], strict=True)
fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
report = {
    "hardware": fingerprint(a.device),
    "torch": torch.__version__,
    "source_revision": SOURCE_REVISION,
    "source_hashes": SOURCE_HASHES,
    "fixtures": {},
}
traces = {}
original.encoder.register_forward_hook(
    lambda mod, inputs, result: traces.update(
        tokens=result[0].detach().cpu().numpy().copy()
    )
)
for path in sorted(fixtures.glob("*.wav")):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(wav.T).to(a.device)
    if sr != 16000:
        x = torchaudio.functional.resample(x, sr, 16000)
    x = x.mean(0, keepdim=True)
    gold = np.load(fixtures / (path.stem + "_" + a.device + ".npz"))
    with torch.inference_mode():
        fbank = original.preprocess(x)
        scores, _ = original.extract_features(x)
        tokens = traces["tokens"].copy()
        new_fbank = extracted.preprocess(x)
        new_scores, mask, new_tokens = extracted.extract_features(x)
        original_arrays = {
            "waveform": x.cpu().numpy(),
            "fbank": fbank.cpu().numpy(),
            "scores": scores.cpu().numpy(),
            "tokens": tokens,
        }
        extracted_arrays = {
            "waveform": x.cpu().numpy(),
            "fbank": new_fbank.cpu().numpy(),
            "scores": new_scores.cpu().numpy(),
            "tokens": new_tokens.cpu().numpy(),
        }
        for stage in original_arrays:
            np.testing.assert_array_equal(
                extracted_arrays[stage],
                original_arrays[stage],
                err_msg=path.stem + ": original/extracted " + stage,
            )
        # Portable waveform/scores bounds remain unchanged; fbank/token drift is reported, not hidden.
        np.testing.assert_allclose(
            original_arrays["waveform"], gold["waveform"], rtol=1e-4, atol=1e-6
        )
        np.testing.assert_allclose(
            original_arrays["scores"], gold["scores"], rtol=1e-4, atol=1e-4
        )
        replay_fb = (
            original.preprocess(torch.from_numpy(gold["waveform"].copy()).to(a.device))
            .cpu()
            .numpy()
        )
    details = {
        "same_host_original_extracted": "bit_exact_all_stages",
        "recorded_golden_max_abs": {
            stage: float(np.max(np.abs(arr - gold[stage])))
            for stage, arr in original_arrays.items()
        },
        "original_fbank_on_saved_golden_waveform_max_abs": float(
            np.max(np.abs(replay_fb - gold["fbank"]))
        ),
    }
    report["fixtures"][path.stem] = details
    print(path.stem, json.dumps(details))
    np.savez_compressed(out / (path.stem + "_" + a.device + ".npz"), **original_arrays)
environment = {
    "hardware": fingerprint(a.device),
    "torch": torch.__version__,
    "torchaudio": torchaudio.__version__,
    "numpy": np.__version__,
    "device": a.device,
    "source_revision": SOURCE_REVISION,
    "source_hashes": SOURCE_HASHES,
    "same_host_reference_verified": True,
}
(out / ("environment_" + a.device + ".json")).write_text(
    json.dumps(environment, indent=2) + "\n"
)
(out / "reference-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    "Strict checksum-verified original/extracted same-host parity passed; generated baseline remains local."
)
