"""Public file-to-full-JSON CLI, optionally writing embeddings to NPZ.

No preview truncation: JSON contains all scores and embeddings unless their
full arrays are written to an explicitly named artifact. Reads: session API.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="BEATs AudioSet tagging and audio representation; not beat tracking"
    )
    from .__about__ import __version__

    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("audio")
    p.add_argument("--device", default="auto")
    p.add_argument("--cache-dir")
    p.add_argument("--checkpoint-dir")
    p.add_argument(
        "--checkpoint-config", help="Alternate validated checkpoint JSON registry"
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument("--output", help="Complete JSON output; defaults to stdout")
    p.add_argument(
        "--embeddings-output",
        help="Write complete patch/clip embeddings as NPZ instead of inline JSON",
    )
    a = p.parse_args()
    if (
        a.output
        and a.embeddings_output
        and Path(a.output).resolve() == Path(a.embeddings_output).resolve()
    ):
        p.error("--output and --embeddings-output must be different paths")
    from .session import BEATsAudioSetSession

    try:
        with BEATsAudioSetSession(
            device=a.device,
            cache_dir=a.cache_dir,
            checkpoint_dir=a.checkpoint_dir,
            offline=a.offline,
            checkpoint_config=a.checkpoint_config,
        ) as s:
            s.load()
            r = s.infer(a.audio)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        p.error(str(exc))
    data = r.to_dict(include_embeddings=a.embeddings_output is None)
    if a.embeddings_output:
        import numpy as np

        path = Path(a.embeddings_output)
        with path.open("wb") as f:
            np.savez_compressed(
                f, patch_embeddings=r.patch_embeddings, clip_embedding=r.clip_embedding
            )
        data["embedding_artifact"] = {
            "path": str(path),
            "format": "npz",
            "arrays": ["patch_embeddings", "clip_embedding"],
        }
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if a.output:
        Path(a.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0
