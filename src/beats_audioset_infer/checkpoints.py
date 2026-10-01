"""Resolve verified, revision-pinned model artifacts independently of Hugging Face.

Local overrides are read-only. Cache inspection never starts a download or
creates a directory; only resolve() may populate the managed cache.
Reads: package-owned config/checkpoints.json, checkpoint paths and cache files.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from datetime import date
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from filelock import FileLock

LABELS_FILE = "class_labels_indices.csv"


def _load_registry(checkpoint_config=None) -> dict:
    try:
        if checkpoint_config is not None:
            return json.loads(Path(checkpoint_config).read_text(encoding="utf-8"))
        return json.loads(
            resources.files("beats_audioset_infer")
            .joinpath("config/checkpoints.json")
            .read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read the package checkpoint registry.") from exc


def _validate_registry(registry) -> dict:
    """Reject malformed identities before constructing cache paths."""
    required = {
        "repo",
        "revision",
        "source_revision",
        "updated_at",
        "model_license",
        "model_license_source",
        "audioset_licenses",
        "files",
    }
    if not isinstance(registry, dict) or not required.issubset(registry):
        raise ValueError(
            "Invalid checkpoint registry: missing identity or artifact fields."
        )
    for key in required - {"files", "audioset_licenses"}:
        if not isinstance(registry[key], str) or not registry[key]:
            raise ValueError(f"Invalid checkpoint registry field: {key}.")
    revision = registry["revision"]
    if (
        revision in (".", "..")
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", revision) is None
    ):
        raise ValueError("Invalid checkpoint registry revision: unsafe cache path.")
    if re.fullmatch(r"[0-9a-f]{40}", registry["source_revision"]) is None:
        raise ValueError(
            "Invalid checkpoint registry source_revision: expected Git SHA."
        )
    try:
        date.fromisoformat(registry["updated_at"])
    except ValueError as exc:
        raise ValueError(
            "Invalid checkpoint registry updated_at: expected ISO date."
        ) from exc
    licenses = registry["audioset_licenses"]
    if not isinstance(licenses, dict) or any(
        not isinstance(licenses.get(key), str) or not licenses[key]
        for key in ("dataset", "ontology", "source")
    ):
        raise ValueError("Invalid checkpoint registry AudioSet license provenance.")
    files = registry["files"]
    if not isinstance(files, dict) or set(files) != {
        "model.pt",
        LABELS_FILE,
    }:
        raise ValueError(
            "Invalid checkpoint registry: expected weights and labels artifacts."
        )
    for filename, artifact in files.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"Invalid checkpoint registry artifact: {filename}.")
        checksum, size, url = (
            artifact.get("sha256"),
            artifact.get("size"),
            artifact.get("url"),
        )
        if (
            not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
        ):
            raise ValueError(f"Invalid checkpoint registry SHA256: {filename}.")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"Invalid checkpoint registry size: {filename}.")
        try:
            parsed = urlparse(url) if isinstance(url, str) else None
            if parsed is None or parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("Expected an absolute HTTPS URL.")
        except ValueError as exc:
            raise ValueError(
                f"Invalid checkpoint registry HTTPS URL: {filename}."
            ) from exc
    return copy.deepcopy(registry)


class CheckpointStore:
    """Manage one immutable checkpoint revision and its AudioSet label asset."""

    def __init__(
        self, cache_dir=None, checkpoint_dir=None, offline=False, checkpoint_config=None
    ):
        self._registry = _validate_registry(_load_registry(checkpoint_config))
        root = (
            cache_dir
            if cache_dir is not None
            else os.environ.get(
                "BEATS_AUDIOSET_INFER_CACHE", "~/.cache/beats-audioset-infer"
            )
        )
        self.cache_dir = Path(root).expanduser().absolute() / self._registry["revision"]
        self.checkpoint_dir = (
            Path(checkpoint_dir).expanduser().absolute()
            if checkpoint_dir is not None
            else None
        )
        self.offline = bool(offline)

    def _path(self, filename: str) -> Path:
        if self.checkpoint_dir is not None:
            explicit = self.checkpoint_dir / filename
            if filename != LABELS_FILE or explicit.exists():
                return explicit
        return self.cache_dir / filename

    def _verify(self, path: Path, artifact: dict) -> None:
        actual_size = path.stat().st_size
        if actual_size != artifact["size"]:
            raise ValueError(
                f"Checkpoint size mismatch for {path}: "
                f"expected {artifact['size']}, found {actual_size}."
            )
        checksum = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != artifact["sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch for {path}.")

    def _download(self, path: Path, artifact: dict) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as target:
                temporary = Path(target.name)
                size = 0
                with urlopen(artifact["url"], timeout=60) as response:
                    if urlparse(response.geturl()).scheme != "https":
                        raise ValueError(
                            "Checkpoint download redirected away from HTTPS."
                        )
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > artifact["size"]:
                            raise ValueError(
                                f"Checkpoint download exceeds expected size: {path.name}"
                            )
                        target.write(block)
                target.flush()
                os.fsync(target.fileno())
            self._verify(temporary, artifact)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def resolve(self) -> dict[str, Path]:
        """Verify all artifacts, downloading missing managed files when online.

        An existing corrupt file always raises; it is never silently replaced.
        Missing explicit model files raise even in online mode.
        """
        resolved = {}
        for filename, artifact in self._registry["files"].items():
            path = self._path(filename)
            if path.exists():
                self._verify(path, artifact)
            elif self.checkpoint_dir is not None and path.parent == self.checkpoint_dir:
                raise FileNotFoundError(f"Missing explicit checkpoint artifact: {path}")
            elif self.offline:
                raise FileNotFoundError(
                    f"Checkpoint artifact unavailable offline: {path}"
                )
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with FileLock(str(path) + ".lock"):
                    if path.exists():
                        self._verify(path, artifact)
                    else:
                        self._download(path, artifact)
            resolved[filename] = path
        return resolved

    def cache_info(self) -> dict:
        """Inspect paths without hashing, writing, downloading or claiming integrity."""
        files = {}
        for filename, artifact in self._registry["files"].items():
            path = self._path(filename)
            exists = path.is_file()
            files[filename] = {
                "path": str(path),
                "exists": exists,
                "expected_sha256": artifact["sha256"],
                "expected_size": artifact["size"],
                "size": path.stat().st_size if exists else None,
                "integrity": "not_checked",
                "source": "explicit"
                if self.checkpoint_dir is not None
                and path.parent == self.checkpoint_dir
                else "cache",
            }
        return {
            "repo": self._registry["repo"],
            "revision": self._registry["revision"],
            "cache_dir": str(self.cache_dir),
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "offline": self.offline,
            "files": files,
        }

    def provenance(self) -> dict:
        """Return expected artifact identities and licenses, without asserting verification."""
        return {
            "repo": self._registry["repo"],
            "revision": self._registry["revision"],
            "source_revision": self._registry["source_revision"],
            "updated_at": self._registry["updated_at"],
            "model_license": self._registry["model_license"],
            "model_license_source": self._registry["model_license_source"],
            "encoder_provenance": copy.deepcopy(
                self._registry.get("encoder_provenance", {})
            ),
            "audioset_licenses": dict(self._registry["audioset_licenses"]),
            "files": {
                filename: {
                    "path": str(self._path(filename)),
                    "sha256": artifact["sha256"],
                    "size": artifact["size"],
                    "url": artifact["url"],
                }
                for filename, artifact in self._registry["files"].items()
            },
        }
