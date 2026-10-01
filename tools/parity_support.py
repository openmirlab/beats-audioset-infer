"""Verification-only hardware guard and immutable upstream source identities.

This module never ships in the wheel. Numeric goldens remain tied to the CPU
window dispatch or GPU that produced them; unknown hosts require same-host A/B.
Reads: Torch verification environment and reference-fixture metadata.
"""

import json
import os
import platform
from pathlib import Path

import torch

SOURCE_REVISION = "31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd"
SOURCE_HASHES = {
    "BEATs.py": "27f289db7c56ce26f2ceb50d3719854b91b2dec1c2830d8b1dd8de1bbee19eeb",
    "backbone.py": "31c0378379a7e0f1d1069f9da444fb86890fe1ea078959a2dcd39640cdcadbaa",
    "modules.py": "edeb6b6cd6a784da749f932c3e0783c0bce556fc768d0d23a4d53d4b819eb424",
}


def fingerprint(device):
    if device.startswith("cuda"):
        return {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(torch.device(device)),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }
    cpu = "unknown"
    path = Path("/proc/cpuinfo")
    if path.is_file():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in path.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "device": "cpu",
        "architecture": platform.machine(),
        "cpu": cpu,
        "dispatch": torch.backends.cpu.get_cpu_capability(),
    }


def baseline_root(fixtures, device):
    root = Path(os.environ.get("BEATS_TEST_BASELINE", fixtures))
    environment = json.loads((root / ("environment_" + device + ".json")).read_text())
    if environment.get("hardware") != fingerprint(device):
        raise AssertionError(
            "Recorded numeric golden hardware/CPU dispatch differs. Run tools/verify_upstream_parity.py on this host and set BEATS_TEST_BASELINE to its verified same-host outputs; nonportable arrays must not pass by widening tolerances."
        )
    if root != Path(fixtures) and (
        not environment.get("same_host_reference_verified")
        or environment.get("source_hashes") != SOURCE_HASHES
    ):
        raise AssertionError(
            "Alternate baseline requires checksum-verified original same-host A/B provenance"
        )
    return root, environment
