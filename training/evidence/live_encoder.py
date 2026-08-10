"""Recover source windows and re-encode selected Phase-B patches with a live tokenizer."""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace

import numpy as np
import torch

from data.scripts.augmentations import AugmentationConfig, IMUAugmenter, IMUSample
from data.scripts.eda.grid_io import discover_grids, grid_corpus_fingerprint
from training.evidence.patch_episodes import EvidenceBatch, QueryPatches
from training.evidence.subject_style import SubjectStyle, apply_subject_style
from training.tokenizer.eval_transfer import encode_dataset_detailed
from training.tokenizer.pretrain_data import (
    CHANNELS,
    _stream_gravity_state,
    stream_channel_descriptions,
    stream_sensor_texts,
)


@dataclass(frozen=True)
class PatchViewSpec:
    """Physical view applied consistently to every patch of one source execution."""

    subject_style: SubjectStyle | None = None
    generic_seed: int | None = None


@contextmanager
def _seeded_augmentation(seed: int):
    """Make the legacy global-RNG augmenter deterministic without perturbing trainer RNGs."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    # The augmenter draws from the CPU default generator. Seeding it directly avoids mutating
    # every CUDA generator, which would otherwise perturb trainer dropout and sampling streams.
    torch.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


class SourcePatchEncoder:
    """Resolve schema-v3 bank patch rows back to canonical native-grid windows."""

    def __init__(self, bank: dict, device, *, verify_fingerprint: bool = True):
        if int(bank.get("schema_version", 0)) < 3:
            raise ValueError("live tokenizer encoding requires a schema-v3 source-aware bank")
        self.bank = bank
        self.device = torch.device(device)
        alignment = str(bank.get("source_alignment", "native"))
        if alignment != "native":
            raise ValueError(f"unsupported Phase-B source alignment {alignment!r}")
        self.cfg_names = {int(key): str(value) for key, value in bank["cfg_names"].items()}
        represented = set(self.cfg_names.values())
        refs = [ref for ref in discover_grids(alignment) if ref.key in represented]
        self.refs = {ref.key: ref for ref in refs}
        missing = sorted(set(self.cfg_names.values()) - set(self.refs))
        if missing:
            raise RuntimeError(f"source grids needed by the memory bank are missing: {missing}")
        # Keep one read-only mmap per represented stream. Fine-tuning repeatedly revisits the active
        # index; reopening the same .npy file for every episode adds avoidable filesystem overhead.
        self.data = {key: self.refs[key].load_data() for key in set(self.cfg_names.values())}
        if verify_fingerprint:
            expected = (bank.get("corpus") or {}).get("source_grid_fp")
            current = grid_corpus_fingerprint(alignment, refs)
            if not expected or current != expected:
                raise RuntimeError(
                    "source grids no longer match the rows recorded in the memory bank; "
                    "rebuild the bank before tokenizer fine-tuning"
                )
        self.phase_b_augmenter = IMUAugmenter(AugmentationConfig.phase_b_generic())

    def _view_window(self, cfg_id: int, parent_window: int, spec: PatchViewSpec) -> np.ndarray:
        key = self.cfg_names[int(cfg_id)]
        ref = self.refs[key]
        source_row = int(torch.as_tensor(self.bank["source_row"])[parent_window])
        raw = np.asarray(self.data[key][source_row], dtype=np.float32).copy()
        if spec.subject_style is not None:
            raw = apply_subject_style(raw, float(ref.rate_hz), ref.mask, spec.subject_style)
        if spec.generic_seed is None:
            return raw
        role_text, sensor_text, sensor_id = stream_sensor_texts(
            ref.dataset,
            ref.stream,
            has_accel=bool(any(ref.mask[:3])),
            has_gyro=bool(any(ref.mask[3:])),
        )
        sample = IMUSample(
            data=torch.from_numpy(raw),
            channel_names=list(CHANNELS),
            sampling_rate=float(ref.rate_hz),
            channel_descriptions=stream_channel_descriptions(ref.dataset, ref.stream),
            channel_mask=[bool(value) for value in ref.mask],
            role_descriptions=role_text,
            sensor_descriptions=sensor_text,
            sensor_id=sensor_id,
            gravity_state=_stream_gravity_state(ref.dataset, ref.stream),
        )
        with _seeded_augmentation(int(spec.generic_seed)):
            sample = self.phase_b_augmenter(sample)
        if sample.channel_names != list(CHANNELS) or sample.data.shape != raw.shape:
            raise RuntimeError("Phase-B generic augmentation changed source patch layout")
        sample.data[:, ~torch.as_tensor(ref.mask, dtype=torch.bool)] = 0.0
        return sample.data.detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_patch_rows_with_views(
        self,
        patch_rows: torch.Tensor,
        view_specs: list[PatchViewSpec],
        encoder,
        *,
        requires_grad: bool,
    ) -> torch.Tensor:
        """Encode occurrence-specific physical views while retaining stored patch ordinals."""
        requested = patch_rows.detach().cpu().long().reshape(-1)
        if len(requested) != len(view_specs):
            raise ValueError("one PatchViewSpec is required for every requested patch row")
        if not len(requested):
            return torch.empty(0, int(self.bank["d_model"]), device=self.device)
        patch = self.bank["patch"]
        parent = torch.as_tensor(patch["window"])[requested].long()
        ordinal = torch.as_tensor(patch["ordinal"])[requested].long()

        # Encode each (source window, physical view) once even though all of its patch rows occur.
        keys: list[tuple[int, PatchViewSpec]] = []
        key_to_local: dict[tuple[int, PatchViewSpec], int] = {}
        occurrence_key = []
        for parent_id, spec in zip(parent.tolist(), view_specs):
            key = (int(parent_id), spec)
            if key not in key_to_local:
                key_to_local[key] = len(keys)
                keys.append(key)
            occurrence_key.append(key_to_local[key])
        key_cfg = torch.tensor([
            int(torch.as_tensor(self.bank["cfg"])[parent_id]) for parent_id, _ in keys
        ])
        vectors: list[torch.Tensor | None] = [None] * len(requested)

        for cfg_id in sorted(key_cfg.unique().tolist()):
            local_keys = torch.nonzero(key_cfg.eq(cfg_id), as_tuple=True)[0].tolist()
            cfg_name = self.cfg_names[int(cfg_id)]
            ref = self.refs[cfg_name]
            windows = np.stack([
                self._view_window(cfg_id, keys[key_i][0], keys[key_i][1])
                for key_i in local_keys
            ])
            encoded = encode_dataset_detailed(
                encoder,
                windows,
                stream_channel_descriptions(ref.dataset, ref.stream),
                self.device,
                float(ref.rate_hz),
                _stream_gravity_state(ref.dataset, ref.stream),
                channel_mask=ref.mask,
                dataset=ref.dataset,
                stream=ref.stream,
                requires_grad=requires_grad,
                batch_size=min(256, len(windows)),
            )
            encoded_parent = encoded["patch_window"].detach().cpu().long()
            order = torch.argsort(encoded_parent, stable=True)
            counts = torch.bincount(encoded_parent, minlength=len(local_keys))
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
            global_to_cfg_local = {key_i: position for position, key_i in enumerate(local_keys)}
            # One gather, not one per patch. The row-at-a-time form issued a host->device copy and a
            # separate index op for every requested occurrence, so a single episode paid hundreds of
            # tiny transfers; the selected rows and their order are unchanged.
            wanted = torch.as_tensor(
                [index for index, key_i in enumerate(occurrence_key)
                 if key_i in global_to_cfg_local],
                dtype=torch.long,
            )
            if not len(wanted):
                continue
            local_window = torch.as_tensor(
                [global_to_cfg_local[occurrence_key[int(index)]] for index in wanted],
                dtype=torch.long,
            )
            patch_ordinal = ordinal[wanted]
            if bool((patch_ordinal >= counts[local_window]).any()):
                raise RuntimeError(f"{cfg_name}: augmented patch layout differs from the bank")
            encoded_rows = order[offsets[local_window] + patch_ordinal]
            gathered = encoded["patch_Z"][encoded_rows.to(encoded["patch_Z"].device)]
            for position, request_i in enumerate(wanted.tolist()):
                vectors[request_i] = gathered[position]
        if any(value is None for value in vectors):
            raise RuntimeError("failed to encode every augmented Phase-B patch occurrence")
        return torch.stack(vectors)  # type: ignore[arg-type]

    def encode_patch_rows(
        self,
        patch_rows: torch.Tensor,
        encoder,
        *,
        requires_grad: bool,
    ) -> torch.Tensor:
        """Return live vectors in exactly ``patch_rows`` order, encoding each source window once."""
        requested = patch_rows.detach().cpu().long().reshape(-1)
        if not len(requested):
            return torch.empty(0, int(self.bank["d_model"]), device=self.device)
        if int(requested.min()) < 0 or int(requested.max()) >= len(self.bank["patch"]["Z"]):
            raise IndexError("requested patch row is outside the source-aware bank")
        unique_rows, inverse = torch.unique(requested, sorted=True, return_inverse=True)
        patch = self.bank["patch"]
        parent = torch.as_tensor(patch["window"])[unique_rows].long()
        config = torch.as_tensor(patch["cfg"])[unique_rows].long()
        ordinal = torch.as_tensor(patch["ordinal"])[unique_rows].long()
        source_row = torch.as_tensor(self.bank["source_row"]).long()
        vectors: list[torch.Tensor | None] = [None] * len(unique_rows)

        for cfg_id in sorted(config.unique().tolist()):
            cfg_positions = torch.nonzero(config.eq(cfg_id), as_tuple=True)[0]
            cfg_windows = parent[cfg_positions]
            unique_windows, local_window = torch.unique(
                cfg_windows, sorted=True, return_inverse=True
            )
            key = self.cfg_names[int(cfg_id)]
            ref = self.refs[key]
            rows = source_row[unique_windows].numpy()
            if len(rows) and int(rows.max()) >= ref.n_windows:
                raise RuntimeError(f"{key}: bank source_row exceeds the current grid")
            encoded = encode_dataset_detailed(
                encoder,
                np.asarray(self.data[key][rows]),
                stream_channel_descriptions(ref.dataset, ref.stream),
                self.device,
                float(ref.rate_hz),
                _stream_gravity_state(ref.dataset, ref.stream),
                channel_mask=ref.mask,
                dataset=ref.dataset,
                stream=ref.stream,
                requires_grad=requires_grad,
                batch_size=min(256, max(1, len(rows))),
            )
            encoded_parent = encoded["patch_window"].detach().cpu().long()
            order = torch.argsort(encoded_parent, stable=True)
            counts = torch.bincount(encoded_parent, minlength=len(unique_windows))
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
            requested_ordinal = ordinal[cfg_positions]
            available = counts[local_window]
            if bool((requested_ordinal >= available).any()):
                raise RuntimeError(f"{key}: live patch layout differs from the bank")
            encoded_row = order[offsets[local_window] + requested_ordinal]

            # Patch structure must remain identical even though train-mode dropout changes vectors.
            for field, bank_field in (
                ("patch_time", "time"),
                ("patch_duration", "duration"),
                ("patch_resolution", "resolution"),
            ):
                live = encoded[field][encoded_row.to(encoded[field].device)].detach().cpu()
                stored = torch.as_tensor(patch[bank_field])[unique_rows[cfg_positions]].cpu()
                if field == "patch_resolution":
                    matches = torch.equal(live.long(), stored.long())
                else:
                    matches = torch.allclose(live.float(), stored.float(), atol=1e-6, rtol=0)
                if not matches:
                    raise RuntimeError(f"{key}: live {field} differs from source-bank metadata")
            selected = encoded["patch_Z"][encoded_row.to(encoded["patch_Z"].device)]
            for position, value in zip(cfg_positions.tolist(), selected.unbind(0)):
                vectors[position] = value

        if any(value is None for value in vectors):
            raise RuntimeError("failed to recover every requested source patch")
        unique_vectors = torch.stack(vectors)  # type: ignore[arg-type]
        # Live vectors belong on the encoder/model device regardless of where CPU-resident row ids
        # came from. Returning them to the row-id device caused GPU->CPU->GPU copies on index refresh.
        return unique_vectors[inverse.to(unique_vectors.device)]

    def reencode_query_views(
        self,
        query: QueryPatches,
        encoder,
        view_specs: list[PatchViewSpec],
        *,
        requires_grad: bool,
    ) -> QueryPatches:
        if len(view_specs) != query.Z.shape[0]:
            raise ValueError("one query view is required for each episode batch row")
        if bool((query.row[query.mask] < 0).any()):
            raise ValueError("external queries cannot be physically augmented from this bank")
        occurrence_specs = []
        for batch_row, spec in enumerate(view_specs):
            active_windows = query.window[batch_row, query.mask[batch_row]].detach().cpu().tolist()
            for source_window in active_windows:
                seed = None if spec.generic_seed is None else (
                    int(spec.generic_seed) + 1_000_003 * int(source_window)
                ) % (2**31 - 1)
                occurrence_specs.append(PatchViewSpec(spec.subject_style, seed))
        live = self.encode_patch_rows_with_views(
            query.row[query.mask], occurrence_specs, encoder, requires_grad=requires_grad
        )
        z = query.Z.new_zeros(query.Z.shape)
        z = z.masked_scatter(query.mask.unsqueeze(-1).expand_as(z), live.reshape(-1))
        return replace(query, Z=z)

    def reencode_evidence(
        self,
        evidence: EvidenceBatch,
        encoder,
        *,
        requires_grad: bool,
    ) -> torch.Tensor:
        rows = evidence.index[evidence.mask]
        live = self.encode_patch_rows(rows, encoder, requires_grad=requires_grad)
        d = live.shape[-1]
        z = live.new_zeros((*evidence.index.shape, d))
        z = z.masked_scatter(evidence.mask.unsqueeze(-1).expand_as(z), live.reshape(-1))
        return z
