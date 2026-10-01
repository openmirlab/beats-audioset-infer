"""Portable full AudioSet scores and native encoder patch embeddings.

Arrays own CPU storage, so release of the session cannot invalidate results.
Reads: NumPy and model-output metadata; no model or device state.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BEATsAudioSetResult:
    scores: tuple[float, ...]
    labels: tuple[str, ...]
    mids: tuple[str, ...]
    patch_embeddings: np.ndarray
    clip_embedding: np.ndarray
    embedding_info: dict
    audio: dict
    checkpoint: dict

    @property
    def ranked_tags(self):
        return tuple(
            {
                "index": i,
                "mid": self.mids[i],
                "label": self.labels[i],
                "score": self.scores[i],
            }
            for i in sorted(range(len(self.scores)), key=lambda i: (-self.scores[i], i))
        )

    def to_dict(self, *, include_embeddings=True):
        result = {
            "scores": list(self.scores),
            "labels": list(self.labels),
            "mids": list(self.mids),
            "ranked_tags": list(self.ranked_tags),
            "embedding_info": self.embedding_info,
            "audio": self.audio,
            "checkpoint": self.checkpoint,
        }
        if include_embeddings:
            result.update(
                patch_embeddings=self.patch_embeddings.tolist(),
                clip_embedding=self.clip_embedding.tolist(),
            )
        else:
            result["embeddings_included"] = False
        return result
