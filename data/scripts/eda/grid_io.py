"""Read-only discovery and deterministic sampling for generated HALO grids."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO / "data" / "datasets"


@dataclass(frozen=True)
class GridRef:
    """Metadata and paths for one persisted dataset/stream grid."""

    dataset: str
    stream: str
    alignment: str
    rate_hz: float
    channels: tuple[str, ...]
    mask: tuple[bool, ...]
    labels: tuple[str, ...]
    subjects: tuple[str, ...]
    shape: tuple[int, int, int]
    grid_dir: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.stream}"

    @property
    def n_windows(self) -> int:
        return self.shape[0]

    @property
    def duration_seconds(self) -> float:
        return self.shape[0] * self.shape[1] / self.rate_hz

    def load_data(self) -> np.ndarray:
        """Memory-map the grid so callers read only selected windows."""
        return np.load(self.grid_dir / "data.npy", mmap_mode="r")


def discover_grids(
    alignment: str = "harmonised",
    datasets_dir: Path = DATASETS_DIR,
) -> list[GridRef]:
    """Discover and validate all persisted grids for one alignment."""
    refs: list[GridRef] = []
    pattern = f"*/grids/{alignment}/*/meta.json"
    for meta_path in sorted(datasets_dir.glob(pattern)):
        grid_dir = meta_path.parent
        data_path = grid_dir / "data.npy"
        mask_path = grid_dir / "mask.npy"
        if not data_path.exists() or not mask_path.exists():
            continue

        meta = json.loads(meta_path.read_text())
        data = np.load(data_path, mmap_mode="r")
        mask = np.load(mask_path).astype(bool)
        channels = tuple(map(str, meta["channels"]))
        labels = tuple(map(str, meta["labels"]))
        subjects = tuple(map(str, meta["subjects"]))

        if data.ndim != 3:
            raise ValueError(f"{grid_dir}: expected (N,T,C), got {data.shape}")
        if len(labels) != data.shape[0] or len(subjects) != data.shape[0]:
            raise ValueError(f"{grid_dir}: labels/subjects do not match window count")
        if len(channels) != data.shape[2] or mask.shape != (data.shape[2],):
            raise ValueError(f"{grid_dir}: channels/mask do not match channel dimension")

        refs.append(GridRef(
            dataset=str(meta["dataset"]),
            stream=str(meta["stream_id"]),
            alignment=str(meta["alignment"]),
            rate_hz=float(meta["rate_hz"]),
            channels=channels,
            mask=tuple(map(bool, mask)),
            labels=labels,
            subjects=subjects,
            shape=tuple(map(int, data.shape)),
            grid_dir=grid_dir,
        ))
    return refs


def triad_indices(ref: GridRef, modality: str) -> tuple[int, int, int] | None:
    """Return valid xyz indices for ``acc`` or ``gyro``, else ``None``."""
    names = tuple(f"{modality}_{axis}" for axis in "xyz")
    if not all(name in ref.channels for name in names):
        return None
    indices = tuple(ref.channels.index(name) for name in names)
    if not all(ref.mask[index] for index in indices):
        return None
    return indices


def matching_refs(
    refs: Iterable[GridRef],
    label: str,
    selectors: Sequence[str] | None = None,
) -> list[GridRef]:
    """Select grids containing ``label`` by dataset or dataset/stream selector."""
    eligible = [ref for ref in refs if label in ref.labels]
    if not selectors:
        return eligible

    selected: list[GridRef] = []
    for selector in selectors:
        for ref in eligible:
            if ref in selected:
                continue
            if selector == ref.dataset or selector == ref.key:
                selected.append(ref)
    return selected


def sample_indices(ref: GridRef, label: str, count: int, seed: int) -> np.ndarray:
    """Choose reproducible windows, preferring distinct subjects when possible."""
    candidates = np.fromiter(
        (index for index, value in enumerate(ref.labels) if value == label),
        dtype=np.int64,
    )
    if len(candidates) < count:
        raise ValueError(f"{ref.key}: requested {count} {label!r} windows, found {len(candidates)}")
    digest = hashlib.blake2b(
        f"{seed}:{ref.key}:{label}".encode("utf-8"), digest_size=8
    ).digest()
    local_seed = int.from_bytes(digest, "little")
    rng = np.random.default_rng(local_seed)

    by_subject: dict[str, list[int]] = {}
    for index in candidates:
        by_subject.setdefault(ref.subjects[int(index)], []).append(int(index))
    subject_order = np.asarray(sorted(by_subject), dtype=object)
    rng.shuffle(subject_order)

    chosen = [int(rng.choice(by_subject[str(subject)])) for subject in subject_order[:count]]
    if len(chosen) < count:
        remaining = np.asarray([index for index in candidates if int(index) not in chosen])
        chosen.extend(map(int, rng.choice(remaining, size=count - len(chosen), replace=False)))
    return np.sort(np.asarray(chosen, dtype=np.int64))


def output_dir(path: Path | None = None) -> Path:
    result = path or Path(__file__).resolve().parent / "outputs"
    result.mkdir(parents=True, exist_ok=True)
    return result


def output_subdir(root: Path, *parts: str) -> Path:
    """Create and return an analysis-specific directory below an output root."""
    result = root.joinpath(*parts)
    result.mkdir(parents=True, exist_ok=True)
    return result
