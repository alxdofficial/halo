"""Patch-bank indexing, leakage masks, and evidence assembly for Phase-B episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from model.evidence.patch_retrieval import PatchRetrieval


@dataclass
class QueryPatches:
    Z: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor
    duration: torch.Tensor
    resolution: torch.Tensor
    window: torch.Tensor
    event: torch.Tensor
    subj: torch.Tensor
    cfg: torch.Tensor
    sensor: torch.Tensor


@dataclass
class EvidenceBatch:
    index: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor
    head: torch.Tensor
    query_patch: torch.Tensor


class PatchTable:
    """CPU-resident patch metadata with efficient parent-window and label row lookup."""

    def __init__(self, bank: dict):
        self.patch = bank["patch"]
        self.n_windows = len(bank["Z"])
        self.window_label = torch.as_tensor(bank["y"], dtype=torch.long).cpu()
        self.window_event = torch.as_tensor(bank["event"], dtype=torch.long).cpu()
        self.window_event_verified = torch.as_tensor(
            bank["event_verified"], dtype=torch.bool
        ).cpu()
        window = torch.as_tensor(self.patch["window"], dtype=torch.long).cpu()
        order = torch.argsort(window, stable=True)
        self.order = order
        sorted_window = window[order]
        counts = torch.bincount(sorted_window, minlength=self.n_windows)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        self.label_rows: dict[int, torch.Tensor] = {}
        labels = torch.as_tensor(self.patch["y"], dtype=torch.long).cpu()
        for label in torch.unique(labels).tolist():
            self.label_rows[int(label)] = torch.nonzero(labels == label, as_tuple=True)[0]
        self.event_windows: dict[int, torch.Tensor] = {}
        for event in torch.unique(self.window_event[self.window_event_verified]).tolist():
            rows = torch.nonzero(
                self.window_event_verified & self.window_event.eq(event), as_tuple=True
            )[0]
            self.event_windows[int(event)] = rows

    def rows_for_windows(self, windows: torch.Tensor) -> list[torch.Tensor]:
        rows = []
        for window in windows.detach().cpu().tolist():
            lo, hi = int(self.offsets[window]), int(self.offsets[window + 1])
            rows.append(self.order[lo:hi])
        return rows

    def gather_queries(
        self,
        windows: torch.Tensor,
        device,
        *,
        expand_verified_events: bool = True,
        allowed_window_mask: torch.Tensor | None = None,
    ) -> QueryPatches:
        window_groups = []
        allowed_cpu = (
            None if allowed_window_mask is None
            else allowed_window_mask.detach().cpu().bool()
        )
        for window in windows.detach().cpu().tolist():
            group = torch.tensor([window], dtype=torch.long)
            if expand_verified_events and bool(self.window_event_verified[window]):
                siblings = self.event_windows.get(int(self.window_event[window]), group)
                # An event id is only a positive identity when labels agree. This catches converter
                # clock/segmentation mistakes instead of silently making a multi-label query set.
                siblings = siblings[self.window_label[siblings].eq(self.window_label[window])]
                if allowed_cpu is not None:
                    siblings = siblings[allowed_cpu[siblings]]
                if len(siblings):
                    group = siblings
            window_groups.append(group)
        rows = []
        for group in window_groups:
            parts = self.rows_for_windows(group)
            rows.append(torch.cat(parts) if parts else torch.empty(0, dtype=torch.long))
        if not rows or min(map(len, rows)) < 1:
            raise ValueError("every query window must have at least one valid patch")
        B, Q = len(rows), max(map(len, rows))
        d = int(self.patch["Z"].shape[1])
        Z = torch.zeros(B, Q, d, dtype=torch.float32, device=device)
        mask = torch.zeros(B, Q, dtype=torch.bool, device=device)

        def allocate(dtype):
            return torch.zeros(B, Q, dtype=dtype, device=device)

        time = allocate(torch.float32)
        duration = allocate(torch.float32)
        resolution = torch.full((B, Q), -1, dtype=torch.long, device=device)
        window = allocate(torch.long)
        event = allocate(torch.long)
        subj = allocate(torch.long)
        cfg = allocate(torch.long)
        sensor = allocate(torch.long)
        for b, row in enumerate(rows):
            n = len(row)
            mask[b, :n] = True
            Z[b, :n] = torch.as_tensor(self.patch["Z"])[row].float().to(device)
            time[b, :n] = torch.as_tensor(self.patch["time"])[row].float().to(device)
            duration[b, :n] = torch.as_tensor(self.patch["duration"])[row].float().to(device)
            resolution[b, :n] = torch.as_tensor(self.patch["resolution"])[row].long().to(device)
            window[b, :n] = torch.as_tensor(self.patch["window"])[row].long().to(device)
            event[b, :n] = torch.as_tensor(self.patch["event"])[row].long().to(device)
            subj[b, :n] = torch.as_tensor(self.patch["subj"])[row].long().to(device)
            cfg[b, :n] = torch.as_tensor(self.patch["cfg"])[row].long().to(device)
            sensor[b, :n] = torch.as_tensor(self.patch["sensor"])[row].long().to(device)
        return QueryPatches(
            Z=Z, mask=mask, time=time, duration=duration, resolution=resolution,
            window=window, event=event, subj=subj, cfg=cfg, sensor=sensor,
        )

    def sample_index_rows(
        self,
        memory_window_mask: torch.Tensor,
        per_label: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Label-balanced, config/resolution-stratified patch roster for an EMA index rebuild."""
        memory_window_mask = memory_window_mask.detach().cpu().bool()
        selected = []
        patch_window = torch.as_tensor(self.patch["window"], dtype=torch.long).cpu()
        patch_cfg = torch.as_tensor(self.patch["cfg"], dtype=torch.long).cpu()
        patch_resolution = torch.as_tensor(self.patch["resolution"], dtype=torch.long).cpu()
        for label in sorted(self.label_rows):
            rows = self.label_rows[label]
            rows = rows[memory_window_mask[patch_window[rows]]]
            if per_label > 0 and len(rows) > per_label:
                groups: dict[tuple[int, int], torch.Tensor] = {}
                for config, resolution in zip(
                    patch_cfg[rows].tolist(), patch_resolution[rows].tolist()
                ):
                    groups.setdefault((config, resolution), [])
                for group in groups:
                    match = patch_cfg[rows].eq(group[0]) & patch_resolution[rows].eq(group[1])
                    groups[group] = rows[match]
                quota = max(1, per_label // len(groups))
                picks = []
                for group in sorted(groups):
                    candidates = groups[group]
                    take = min(quota, len(candidates))
                    chosen = rng.choice(len(candidates), size=take, replace=False)
                    picks.append(candidates[torch.from_numpy(np.sort(chosen))])
                picked = torch.cat(picks)
                if len(picked) > per_label:
                    keep = rng.choice(len(picked), size=per_label, replace=False)
                    picked = picked[torch.from_numpy(np.sort(keep))]
                elif len(picked) < per_label:
                    remaining = rows[~torch.isin(rows, picked)]
                    take = min(per_label - len(picked), len(remaining))
                    if take:
                        fill = rng.choice(len(remaining), size=take, replace=False)
                        picked = torch.cat([
                            picked, remaining[torch.from_numpy(np.sort(fill))]
                        ])
                rows = picked.sort().values
            selected.append(rows)
        if not selected or not any(len(rows) for rows in selected):
            raise ValueError("no patch rows are eligible for the training memory index")
        return torch.cat(selected)


def queries_from_encoded(
    encoded: dict[str, torch.Tensor],
    windows: torch.Tensor,
    device,
    *,
    sensor_id: int = -1,
) -> QueryPatches:
    """Pack selected windows from ``encode_dataset_detailed`` into a padded query patch set."""
    selected_rows = []
    patch_window = encoded["patch_window"].long()
    for window in windows.detach().cpu().tolist():
        selected_rows.append(torch.nonzero(patch_window.eq(window), as_tuple=True)[0])
    if not selected_rows or min(map(len, selected_rows)) < 1:
        raise ValueError("every external query window must contain at least one valid patch")
    B, Q = len(selected_rows), max(map(len, selected_rows))
    d = int(encoded["patch_Z"].shape[1])
    Z = torch.zeros(B, Q, d, dtype=torch.float32, device=device)
    mask = torch.zeros(B, Q, dtype=torch.bool, device=device)
    time = torch.zeros(B, Q, dtype=torch.float32, device=device)
    duration = torch.zeros(B, Q, dtype=torch.float32, device=device)
    resolution = torch.full((B, Q), -1, dtype=torch.long, device=device)
    window_id = torch.zeros(B, Q, dtype=torch.long, device=device)
    # Negative ids cannot collide with nonnegative memory ids. Each external window is its own
    # event/window; all its patches share a local sensor-membership id.
    event = torch.zeros(B, Q, dtype=torch.long, device=device)
    subj = torch.zeros(B, Q, dtype=torch.long, device=device)
    cfg = torch.full((B, Q), int(sensor_id), dtype=torch.long, device=device)
    for b, rows in enumerate(selected_rows):
        n = len(rows)
        mask[b, :n] = True
        Z[b, :n] = encoded["patch_Z"][rows].float().to(device)
        time[b, :n] = encoded["patch_time"][rows].float().to(device)
        duration[b, :n] = encoded["patch_duration"][rows].float().to(device)
        resolution[b, :n] = encoded["patch_resolution"][rows].long().to(device)
        external_id = -(b + 1)
        window_id[b, :n] = external_id
        event[b, :n] = external_id
        subj[b, :n] = external_id
    return QueryPatches(
        Z=Z, mask=mask, time=time, duration=duration, resolution=resolution,
        window=window_id, event=event, subj=subj, cfg=cfg, sensor=cfg.clone(),
    )


def build_allowed_mask(
    patch: dict,
    index_rows: torch.Tensor,
    query: QueryPatches,
    query_label: torch.Tensor,
    candidates: torch.Tensor,
    *,
    truth_present: bool,
    true_support: int | None,
    other_support: int | None,
    config_mode: str,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Apply leakage, config, and independently controlled candidate-support constraints."""
    device = query.Z.device
    rows = index_rows.detach().cpu()
    y = torch.as_tensor(patch["y"])[rows].long().to(device)
    subj = torch.as_tensor(patch["subj"])[rows].long().to(device)
    event = torch.as_tensor(patch["event"])[rows].long().to(device)
    window = torch.as_tensor(patch["window"])[rows].long().to(device)
    cfg = torch.as_tensor(patch["cfg"])[rows].long().to(device)
    B, Q = query.mask.shape
    allowed = (
        subj.view(1, 1, -1).ne(query.subj.unsqueeze(-1))
        & event.view(1, 1, -1).ne(query.event.unsqueeze(-1))
        & window.view(1, 1, -1).ne(query.window.unsqueeze(-1))
        & query.mask.unsqueeze(-1)
    )
    if config_mode == "same":
        allowed &= cfg.view(1, 1, -1).eq(query.cfg.unsqueeze(-1))
    elif config_mode in {"cross", "query_absent"}:
        allowed &= cfg.view(1, 1, -1).ne(query.cfg.unsqueeze(-1))
    elif config_mode != "any":
        raise ValueError(f"unknown config_mode {config_mode!r}")

    # Support is sampled once per query window and shared across its patches. This prevents a
    # nominal budget of one from turning into Q examples merely because the query has Q patches.
    candidate_set = set(candidates.detach().cpu().tolist())
    for b in range(B):
        row_allowed = allowed[b].any(0)
        if not truth_present:
            row_allowed &= y.ne(query_label[b])
        for label in candidate_set:
            cap = true_support if truth_present and label == int(query_label[b]) else other_support
            if cap is None:
                continue
            label_rows = torch.nonzero(row_allowed & y.eq(label), as_tuple=True)[0]
            label_windows = torch.unique(window[label_rows])
            if len(label_windows) <= cap:
                continue
            keep_np = rng.choice(label_windows.detach().cpu().numpy(), size=cap, replace=False) \
                if cap > 0 else np.empty(0, dtype=np.int64)
            keep_window = torch.from_numpy(np.asarray(keep_np)).to(device)
            keep = torch.isin(window, keep_window) if len(keep_np) else torch.zeros_like(row_allowed)
            remove = y.eq(label) & ~keep
            allowed[b, :, remove] = False
    return allowed


def assemble_evidence(
    retrieval: PatchRetrieval,
    online_scores: torch.Tensor,
    index_rows: torch.Tensor,
    patch: dict,
    *,
    max_evidence: int,
    max_per_window: int,
    max_per_label: int,
    tau: float,
) -> EvidenceBatch:
    """Deduplicate/cap retrieved rows and normalize contribution by subspace × resolution."""
    B, Q, H, K = retrieval.index.shape
    device = online_scores.device
    selected_global = index_rows.to(device)[retrieval.index]
    patch_window = torch.as_tensor(patch["window"], device=device, dtype=torch.long)
    patch_y = torch.as_tensor(patch["y"], device=device, dtype=torch.long)
    patch_resolution = torch.as_tensor(patch["resolution"], device=device, dtype=torch.long)
    patch_duration = torch.as_tensor(patch["duration"], device=device, dtype=torch.float32)

    chosen: list[list[tuple[int, float, int, int]]] = []
    for b in range(B):
        candidates = []
        for q in range(Q):
            for h in range(H):
                for j in range(K):
                    if bool(retrieval.valid[b, q, h, j]):
                        candidates.append((
                            int(selected_global[b, q, h, j]),
                            float(online_scores[b, q, h, j].detach()),
                            h, q, j,
                        ))
        candidates.sort(key=lambda value: value[1], reverse=True)
        window_count: dict[int, int] = {}
        label_count: dict[int, int] = {}
        accepted = []
        seen_head_patch: set[tuple[int, int]] = set()
        for global_row, score, head, q_slot, j in candidates:
            identity = (head, global_row)
            if identity in seen_head_patch:
                continue
            source_window = int(patch_window[global_row])
            label = int(patch_y[global_row])
            if window_count.get(source_window, 0) >= max_per_window:
                continue
            if label_count.get(label, 0) >= max_per_label:
                continue
            seen_head_patch.add(identity)
            window_count[source_window] = window_count.get(source_window, 0) + 1
            label_count[label] = label_count.get(label, 0) + 1
            accepted.append((global_row, score, head, q_slot, j))
            if len(accepted) >= max_evidence:
                break
        if not accepted:
            raise ValueError("retrieval produced no evidence after leakage and contribution caps")
        chosen.append(accepted)

    E = max(map(len, chosen))
    index = torch.zeros(B, E, dtype=torch.long, device=device)
    scores = torch.zeros(B, E, dtype=torch.float32, device=device)
    mask = torch.zeros(B, E, dtype=torch.bool, device=device)
    head = torch.zeros(B, E, dtype=torch.long, device=device)
    query_patch = torch.zeros(B, E, dtype=torch.long, device=device)
    for b, accepted in enumerate(chosen):
        n = len(accepted)
        index[b, :n] = torch.tensor([v[0] for v in accepted], device=device)
        scores[b, :n] = torch.stack(
            [online_scores[b, v[3], v[2], v[4]] for v in accepted]
        )
        head[b, :n] = torch.tensor([v[2] for v in accepted], device=device)
        query_patch[b, :n] = torch.tensor([v[3] for v in accepted], device=device)
        mask[b, :n] = True

    # Equal total prior mass per active (subspace,resolution) group. Duration weighting within each
    # group prevents a denser short grid from winning simply by producing more retrievable patches.
    weights = torch.zeros_like(scores)
    resolution = patch_resolution[index]
    duration = patch_duration[index]
    for b in range(B):
        group_ids = head[b] * 3 + (resolution[b].clamp(-1, 1) + 1)
        active_groups = torch.unique(group_ids[mask[b]])
        for group in active_groups:
            select = mask[b] & group_ids.eq(group)
            raw = torch.softmax(scores[b, select] / tau, dim=0) * duration[b, select]
            raw = raw / raw.sum().clamp_min(1e-8)
            weights[b, select] = raw / len(active_groups)
    return EvidenceBatch(
        index=index, weights=weights, scores=scores, mask=mask,
        head=head, query_patch=query_patch,
    )
