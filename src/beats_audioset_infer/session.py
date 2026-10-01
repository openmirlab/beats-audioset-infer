"""Independent BEATs resident model and explicit reusable session lifecycle.

Owns model/device/lifecycle; the checkpoint resolver and audio normalizer own
artifact and input policy. Reads: checkpoint store, native core, result contract.
"""

import copy
import csv
import gc
import threading

from .checkpoints import CheckpointStore


def _device(request):
    import torch

    if request == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if request == "cpu":
        return torch.device("cpu")
    if isinstance(request, str) and (
        request == "cuda" or (request.startswith("cuda:") and request[5:].isdigit())
    ):
        index = 0 if request == "cuda" else int(request[5:])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device unavailable: {request}")
        return torch.device(f"cuda:{index}")
    raise ValueError("device must be auto, cpu, cuda or cuda:N; mps is unsupported")


class BEATsAudioSetSession:
    """Explicit load, ready-only infer, reloadable release, terminal close."""

    def __init__(
        self,
        *,
        device="auto",
        cache_dir=None,
        checkpoint_dir=None,
        offline=False,
        checkpoint_config=None,
    ):
        self._request = device
        self._store = CheckpointStore(
            cache_dir, checkpoint_dir, offline, checkpoint_config
        )
        self._lock = threading.RLock()
        self._status = "new"
        self._model = None
        self._device = None
        self._labels = None
        self._mids = None
        self._checkpoint = None

    @property
    def status(self):
        return self._status

    @property
    def device(self):
        return str(self._device) if self._device is not None else None

    def cache_info(self):
        return self._store.cache_info()

    def _open(self):
        if self._status == "closed":
            raise RuntimeError("Session is closed")

    def load(self):
        with self._lock:
            self._open()
            if self._status == "ready":
                return self
            self._status = "loading"
            try:
                import torch

                from ._impl.model import BEATs, BEATsConfig

                device = _device(self._request)
                files = self._store.resolve()
                data = torch.load(
                    files["model.pt"], weights_only=True, map_location="cpu"
                )
                if not isinstance(data, dict) or set(data) != {
                    "cfg",
                    "model",
                    "label_dict",
                }:
                    raise ValueError("Unexpected BEATs checkpoint structure")
                cfg = data["cfg"]
                mapping = data["label_dict"]
                if (
                    not isinstance(cfg, dict)
                    or not cfg.get("finetuned_model")
                    or cfg.get("predictor_class") != 527
                    or cfg.get("input_patch_size") != 16
                    or cfg.get("encoder_embed_dim") != 768
                ):
                    raise ValueError(
                        "Expected selected AudioSet fine-tuned BEATs configuration"
                    )
                if (
                    not isinstance(mapping, dict)
                    or set(mapping) != set(range(527))
                    or any(not isinstance(v, str) for v in mapping.values())
                    or len(set(mapping.values())) != 527
                ):
                    raise ValueError(
                        "label_dict must map native indices 0..526 to 527 unique AudioSet MIDs"
                    )
                with files["class_labels_indices.csv"].open() as f:
                    rows = list(csv.DictReader(f, skipinitialspace=True))
                labels = {r["mid"]: r["display_name"] for r in rows}
                if len(labels) != 527 or set(labels) != set(mapping.values()):
                    raise ValueError(
                        "AudioSet label mapping does not cover checkpoint MIDs"
                    )
                model = BEATs(BEATsConfig(cfg))
                model.load_state_dict(data["model"], strict=True)
                model.eval().to(device)
                self._model = model
                self._device = device
                self._mids = tuple(mapping[i] for i in range(527))
                self._labels = tuple(labels[mid] for mid in self._mids)
                self._checkpoint = self._store.provenance()
                self._status = "ready"
                return self
            except BaseException:
                self._model = None
                self._device = None
                self._status = "failed"
                raise

    def infer(self, audio, *, sample_rate=None):
        with self._lock:
            self._open()
            if self._status != "ready":
                raise RuntimeError("Call load() before infer()")
            import torch

            from ._impl.audio import normalize
            from .results import BEATsAudioSetResult

            x, info = normalize(audio, sample_rate, self._device)
            self._status = "busy"
            try:
                with torch.inference_mode():
                    scores, mask, tokens = self._model.extract_features(x)
                patches = tokens[0].detach().cpu().numpy().copy()
                clip = patches.mean(axis=0)
                patches.setflags(write=False)
                clip.setflags(write=False)
                details = {
                    "type": "native_final_encoder_patch_tokens",
                    "shape": list(patches.shape),
                    "dtype": str(patches.dtype),
                    "axes": ["time_patch_then_frequency_patch", "encoder_dimension"],
                    "encoder_dimension": 768,
                    "frequency_patches_per_time_patch": 8,
                    "time_patches": patches.shape[0] // 8,
                    "patch_size_fbank_frames": 16,
                    "patch_size_mel_bins": 16,
                    "fbank_mel_bins": 128,
                    "fbank_window_seconds": 0.025,
                    "fbank_hop_seconds": 0.01,
                    "time_patch_stride_seconds": 0.16,
                    "time_patch_support_seconds": 0.175,
                    "partial_final_patch": "dropped",
                    "clip_embedding": "additional arithmetic mean across all native patch tokens; not normalized",
                    "padding": "none; one unpadded clip per call",
                }
                return BEATsAudioSetResult(
                    tuple(float(v) for v in scores[0].cpu()),
                    self._labels,
                    self._mids,
                    patches,
                    clip,
                    details,
                    info,
                    copy.deepcopy(self._checkpoint),
                )
            finally:
                self._status = "ready"

    def release(self):
        with self._lock:
            self._open()
            self._model = None
            self._device = None
            self._labels = None
            self._mids = None
            self._checkpoint = None
            self._status = "released"
            gc.collect()

    def close(self):
        with self._lock:
            if self._status == "closed":
                return
            self.release()
            self._status = "closed"

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *args):
        self.close()
