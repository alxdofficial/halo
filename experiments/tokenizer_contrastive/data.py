"""Subject-disjoint grid indexing and domain-balanced contrastive batches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

from data.scripts.curate.deployment_policy import STREAM_SPECS
from data.scripts.eda.grid_io import GridRef, discover_grids

from .config import ExperimentConfig
from .views import PairedViewMaker, ViewConfig


GRAVITY_STATE = {(spec.dataset, spec.stream_id): spec.gravity_state for spec in STREAM_SPECS}


@dataclass(frozen=True)
class WindowRecord:
    ref_index: int
    window_index: int
    label: str
    label_id: int
    domain: str
    subject: str
    split: str


def _rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.blake2b(f"{seed}:{key}".encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def subject_splits(
    refs: Sequence[GridRef], validation_fraction: float, test_fraction: float, seed: int
) -> dict[tuple[str, str], str]:
    """Assign each dataset subject to exactly one split across all device streams."""
    by_dataset: dict[str, set[str]] = {}
    for ref in refs:
        by_dataset.setdefault(ref.dataset, set()).update(ref.subjects)

    assignments: dict[tuple[str, str], str] = {}
    for dataset, subject_set in sorted(by_dataset.items()):
        subjects = np.asarray(sorted(subject_set), dtype=object)
        _rng(seed, dataset).shuffle(subjects)
        count = len(subjects)
        if count < 3:
            n_validation = 0
            n_test = 0
        else:
            n_validation = max(1, int(round(count * validation_fraction)))
            n_test = max(1, int(round(count * test_fraction)))
            while n_validation + n_test >= count:
                if n_test > 1:
                    n_test -= 1
                elif n_validation > 1:
                    n_validation -= 1
                else:
                    break
        for index, subject in enumerate(subjects):
            split = "train"
            if index < n_validation:
                split = "validation"
            elif index < n_validation + n_test:
                split = "test"
            assignments[(dataset, str(subject))] = split
    return assignments


def build_index(config: ExperimentConfig) -> tuple[list[GridRef], dict[str, list[WindowRecord]], dict]:
    config.validate()
    refs = [
        ref for ref in discover_grids(config.alignment)
        if ref.dataset in config.datasets
    ]
    if not refs:
        raise FileNotFoundError("No selected generated grids were found")
    if any(ref.shape[2] != 6 for ref in refs):
        raise ValueError("The first contrastive experiment requires the harmonised six-channel view")

    assignments = subject_splits(
        refs, config.validation_fraction, config.test_fraction, config.seed
    )
    selected_labels = config.labels or tuple(sorted({
        label
        for ref in refs
        for label, subject in zip(ref.labels, ref.subjects)
        if assignments[(ref.dataset, subject)] == "train"
    }))
    if not selected_labels:
        raise ValueError("No training labels were selected")
    label_to_id = {label: index for index, label in enumerate(selected_labels)}
    grouped: dict[tuple[str, str, str], list[WindowRecord]] = {}
    for ref_index, ref in enumerate(refs):
        for window_index, (label, subject) in enumerate(zip(ref.labels, ref.subjects)):
            if label not in label_to_id:
                continue
            split = assignments[(ref.dataset, subject)]
            record = WindowRecord(
                ref_index=ref_index,
                window_index=window_index,
                label=label,
                label_id=label_to_id[label],
                domain=ref.key,
                subject=subject,
                split=split,
            )
            grouped.setdefault((split, ref.key, label), []).append(record)

    records = {split: [] for split in ("train", "validation", "test")}
    cap = config.max_windows_per_domain_label
    for (split, domain, label), values in sorted(grouped.items()):
        if cap > 0 and len(values) > cap:
            chosen = _rng(config.seed, f"{split}:{domain}:{label}").choice(
                len(values), size=cap, replace=False
            )
            values = [values[int(index)] for index in sorted(chosen)]
        records[split].extend(values)

    for split, values in records.items():
        if split != "test" and not values:
            raise ValueError(f"No records available for required split {split!r}")

    summary = {
        "labels": label_to_id,
        "streams": [ref.key for ref in refs],
        "counts": {
            split: {
                "total": len(values),
                "by_label": {
                    label: sum(record.label == label for record in values)
                    for label in selected_labels
                },
                "by_domain": {
                    domain: sum(record.domain == domain for record in values)
                    for domain in sorted({record.domain for record in values})
                },
            }
            for split, values in records.items()
        },
    }
    return refs, records, summary


class ContrastiveWindowDataset(Dataset):
    def __init__(
        self,
        refs: Sequence[GridRef],
        records: Sequence[WindowRecord],
        config: ExperimentConfig,
    ):
        self.refs = tuple(refs)
        self.records = tuple(records)
        self.views = PairedViewMaker(ViewConfig(
            rotation_probability=config.rotation_probability,
            jitter_probability=config.jitter_probability,
            time_shift_probability=config.time_shift_probability,
        ))
        self._arrays: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _array(self, ref: GridRef) -> np.ndarray:
        if ref.grid_dir not in self._arrays:
            self._arrays[ref.grid_dir] = ref.load_data()
        return self._arrays[ref.grid_dir]

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        ref = self.refs[record.ref_index]
        data = torch.from_numpy(
            np.asarray(self._array(ref)[record.window_index], dtype=np.float32).copy()
        )
        view_a, view_b = self.views(
            data=data,
            channels=ref.channels,
            gravity_state=GRAVITY_STATE.get((ref.dataset, ref.stream), "unknown"),
            rate_hz=ref.rate_hz,
            label=record.label,
            dataset=ref.dataset,
        )
        return {
            "view_a": view_a,
            "view_b": view_b,
            "channel_mask": torch.tensor(ref.mask, dtype=torch.bool),
            "rate_hz": ref.rate_hz,
            "label_id": record.label_id,
            "label": record.label,
            "domain": record.domain,
            "subject": record.subject,
            "window_index": record.window_index,
        }


def collate_windows(samples: Sequence[dict]) -> dict:
    return {
        "view_a": pad_sequence(
            [sample["view_a"] for sample in samples], batch_first=True
        ),
        "view_b": pad_sequence(
            [sample["view_b"] for sample in samples], batch_first=True
        ),
        "window_length": torch.tensor(
            [sample["view_a"].shape[0] for sample in samples], dtype=torch.long
        ),
        "channel_mask": torch.stack([sample["channel_mask"] for sample in samples]),
        "rate_hz": torch.tensor([sample["rate_hz"] for sample in samples], dtype=torch.float32),
        "label_id": torch.tensor([sample["label_id"] for sample in samples], dtype=torch.long),
        "label": [sample["label"] for sample in samples],
        "domain": [sample["domain"] for sample in samples],
        "subject": [sample["subject"] for sample in samples],
        "window_index": torch.tensor(
            [sample["window_index"] for sample in samples], dtype=torch.long
        ),
    }


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Choose P labels and K windows per label, spreading K across domains."""

    def __init__(
        self,
        records: Sequence[WindowRecord],
        classes_per_batch: int,
        samples_per_class: int,
        steps: int,
        seed: int,
    ):
        self.records = tuple(records)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.steps = steps
        self.seed = seed
        self.epoch = 0
        self.by_label_domain: dict[int, dict[str, list[int]]] = {}
        for index, record in enumerate(records):
            self.by_label_domain.setdefault(record.label_id, {}).setdefault(
                record.domain, []
            ).append(index)
        self.labels = tuple(sorted(self.by_label_domain))
        if len(self.labels) < classes_per_batch:
            raise ValueError(
                f"Need {classes_per_batch} represented labels, found {len(self.labels)}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = _rng(self.seed + self.epoch, "domain-balanced-batches")
        for _ in range(self.steps):
            labels = rng.choice(self.labels, size=self.classes_per_batch, replace=False)
            batch = []
            for label in labels:
                by_domain = self.by_label_domain[int(label)]
                domains = np.asarray(sorted(by_domain), dtype=object)
                chosen_domains = rng.choice(
                    domains,
                    size=self.samples_per_class,
                    replace=len(domains) < self.samples_per_class,
                )
                for domain in chosen_domains:
                    batch.append(int(rng.choice(by_domain[str(domain)])))
            rng.shuffle(batch)
            yield batch


class CoverageBalancedBatchSampler(DomainBalancedBatchSampler):
    """Cover every record once per epoch, with class/domain-balanced filler slots."""

    def __init__(
        self,
        records: Sequence[WindowRecord],
        classes_per_batch: int,
        samples_per_class: int,
        seed: int,
    ):
        super().__init__(
            records,
            classes_per_batch=classes_per_batch,
            samples_per_class=samples_per_class,
            steps=1,
            seed=seed,
        )
        chunks = {
            label: int(np.ceil(sum(len(values) for values in by_domain.values()) / samples_per_class))
            for label, by_domain in self.by_label_domain.items()
        }
        self.steps = max(
            max(chunks.values()),
            int(np.ceil(sum(chunks.values()) / classes_per_batch)),
        )

    def __iter__(self) -> Iterator[list[int]]:
        rng = _rng(self.seed + self.epoch, "coverage-balanced-batches")
        queues: dict[int, dict[str, list[int]]] = {}
        all_indices: dict[int, dict[str, tuple[int, ...]]] = {}
        for label, by_domain in self.by_label_domain.items():
            queues[label] = {}
            all_indices[label] = {}
            for domain, indices in by_domain.items():
                shuffled = list(indices)
                rng.shuffle(shuffled)
                queues[label][domain] = shuffled
                all_indices[label][domain] = tuple(indices)

        def remaining(label: int) -> int:
            return sum(len(values) for values in queues[label].values())

        for _ in range(self.steps):
            active = [label for label in self.labels if remaining(label) > 0]
            rng.shuffle(active)
            active.sort(key=remaining, reverse=True)
            selected = active[:self.classes_per_batch]
            if len(selected) < self.classes_per_batch:
                fillers = [label for label in self.labels if label not in selected]
                rng.shuffle(fillers)
                selected.extend(fillers[:self.classes_per_batch - len(selected)])

            batch = []
            for label in selected:
                used_domains: set[str] = set()
                for _sample in range(self.samples_per_class):
                    available = [
                        domain for domain, values in queues[label].items()
                        if values and domain not in used_domains
                    ]
                    if not available:
                        available = [
                            domain for domain, values in queues[label].items() if values
                        ]
                    if available:
                        domain = str(rng.choice(np.asarray(sorted(available), dtype=object)))
                        index = queues[label][domain].pop()
                    else:
                        domains = np.asarray(sorted(all_indices[label]), dtype=object)
                        domain = str(rng.choice(domains))
                        index = int(rng.choice(all_indices[label][domain]))
                    used_domains.add(domain)
                    batch.append(index)
            rng.shuffle(batch)
            yield batch

        if any(remaining(label) for label in self.labels):
            raise RuntimeError("Coverage sampler ended before consuming every record")
