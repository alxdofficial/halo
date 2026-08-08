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
    row: torch.Tensor


@dataclass
class EvidenceBatch:
    index: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor
    mask: torch.Tensor
    head: torch.Tensor
    query_patch: torch.Tensor
    local_index: torch.Tensor | None = None


@dataclass
class EpisodeMemoryView:
    """One batch-level overlay on the immutable active memory index."""

    allowed: torch.Tensor
    support_mask: torch.Tensor
    support_candidate: torch.Tensor
    candidate_ids: torch.Tensor
    query_label: torch.Tensor
    support_units_per_candidate: torch.Tensor
    episode_type: str
    label_mode: str

    @property
    def support_rows(self) -> torch.Tensor:
        return torch.nonzero(self.support_mask, as_tuple=True)[0]


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
        self.window_cfg = torch.as_tensor(bank["cfg"], dtype=torch.long).cpu()
        self.window_subj = torch.as_tensor(bank["subj"], dtype=torch.long).cpu()
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
        verified_rows = torch.nonzero(self.window_event_verified, as_tuple=True)[0]
        if len(verified_rows):
            event_order = torch.argsort(self.window_event[verified_rows], stable=True)
            sorted_rows = verified_rows[event_order]
            sorted_events = self.window_event[sorted_rows]
            events, counts = torch.unique_consecutive(sorted_events, return_counts=True)
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
            for position, event in enumerate(events.tolist()):
                self.event_windows[int(event)] = sorted_rows[
                    int(offsets[position]):int(offsets[position + 1])
                ]

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
        row_id = torch.full((B, Q), -1, dtype=torch.long, device=device)
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
            row_id[b, :n] = row.to(device)
        return QueryPatches(
            Z=Z, mask=mask, time=time, duration=duration, resolution=resolution,
            window=window, event=event, subj=subj, cfg=cfg, sensor=sensor, row=row_id,
        )

    def sample_index_rows(
        self,
        memory_window_mask: torch.Tensor,
        windows_per_label: int,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Label/config/subject-balanced source windows, retaining all of their patch grids."""
        memory_window_mask = memory_window_mask.detach().cpu().bool()
        if windows_per_label < 1:
            raise ValueError("windows_per_label must be positive")
        selected_windows = []
        for label in sorted(self.label_rows):
            candidates = torch.nonzero(
                memory_window_mask & self.window_label.eq(label), as_tuple=True
            )[0]
            if len(candidates) <= windows_per_label:
                selected_windows.append(candidates)
                continue
            configs = sorted(self.window_cfg[candidates].unique().tolist())
            # When a label occurs in more configurations than the active-window budget, assigning
            # the remainder to sorted config ids would permanently exclude later datasets/streams.
            # Randomize only the quota order; the supplied generator keeps the roster reproducible.
            configs = [int(value) for value in rng.permutation(configs)]
            base, extra = divmod(windows_per_label, len(configs))
            picked = []
            for position, config in enumerate(configs):
                quota = base + int(position < extra)
                if quota == 0:
                    continue
                cfg_windows = candidates[self.window_cfg[candidates].eq(config)]
                # Temper large subjects by drawing subject round-robin within a configuration.
                subject_buckets = {}
                for subject in self.window_subj[cfg_windows].unique().tolist():
                    values = cfg_windows[self.window_subj[cfg_windows].eq(subject)].numpy()
                    subject_buckets[int(subject)] = list(rng.permutation(values))
                cfg_pick = []
                while len(cfg_pick) < quota and any(subject_buckets.values()):
                    for subject in sorted(subject_buckets):
                        if subject_buckets[subject] and len(cfg_pick) < quota:
                            cfg_pick.append(subject_buckets[subject].pop())
                picked.extend(cfg_pick)
            if len(picked) < windows_per_label:
                remaining = candidates[~torch.isin(candidates, torch.tensor(picked))]
                take = min(windows_per_label - len(picked), len(remaining))
                picked.extend(rng.choice(remaining.numpy(), size=take, replace=False).tolist())
            selected_windows.append(torch.tensor(sorted(picked), dtype=torch.long))
        selected = [
            row
            for windows in selected_windows
            for row in self.rows_for_windows(windows)
        ]
        if not selected or not any(len(rows) for rows in selected):
            raise ValueError("no patch rows are eligible for the training memory index")
        return torch.cat(selected)


def _support_units(patch: dict, rows: torch.Tensor, device) -> torch.Tensor:
    """Return one support identity per patch: verified event, otherwise source window."""
    window = torch.as_tensor(patch["window"])[rows].long().to(device)
    event = torch.as_tensor(patch["event"])[rows].long().to(device)
    verified = torch.as_tensor(patch["event_verified"])[rows].bool().to(device)
    # Offset verified events so their ids cannot collide with ordinary source-window ids.
    offset = int(torch.as_tensor(patch["window"]).max()) + 1
    return torch.where(verified, event + offset, window)


def build_episode_memory_view(
    patch: dict,
    index_rows: torch.Tensor,
    query: QueryPatches,
    query_label: torch.Tensor,
    candidates: torch.Tensor,
    *,
    support_count: int,
    episode_type: str,
    label_mode: str,
    rng: np.random.Generator,
    truth_present: bool = True,
) -> EpisodeMemoryView:
    """Construct support/background roles without mutating labels stored in the archive.

    Candidate concepts are removed from ordinary background memory. Exactly ``support_count``
    event/window identities per candidate are then restored as provided support. All patches from
    one selected identity are retained, including verified synchronous placements.
    """
    if episode_type not in {
        "semantic_zero_support", "ordinary_few_support",
        "cross_subject_few_support", "same_subject_enrollment",
    }:
        raise ValueError(f"unknown episode type {episode_type!r}")
    if support_count < 0:
        raise ValueError("support_count must be nonnegative")
    if episode_type == "semantic_zero_support" and support_count != 0:
        raise ValueError("semantic_zero_support requires support_count=0")
    if label_mode == "random_alias" and support_count == 0:
        raise ValueError("random aliases are unanswerable without provided support")

    device = query.Z.device
    rows_cpu = index_rows.detach().cpu().long()
    y = torch.as_tensor(patch["y"])[rows_cpu].long().to(device)
    subj = torch.as_tensor(patch["subj"])[rows_cpu].long().to(device)
    event = torch.as_tensor(patch["event"])[rows_cpu].long().to(device)
    window = torch.as_tensor(patch["window"])[rows_cpu].long().to(device)
    support_unit = _support_units(patch, rows_cpu, device)
    candidates = candidates.long().to(device)
    query_label = query_label.long().to(device)
    query_truth_is_candidate = torch.isin(query_label, candidates)
    if truth_present and bool((~query_truth_is_candidate).any()):
        raise ValueError("every query label must be in an answerable episode candidate set")
    if not truth_present and bool(query_truth_is_candidate.any()):
        raise ValueError("truth-absent episodes cannot include a query label as a candidate")

    # A support identity can never be the query execution itself. For verified events this also
    # excludes every synchronous placement, rather than leaking a sibling sensor as enrollment.
    query_windows = torch.unique(query.window[query.mask])
    query_events = torch.unique(query.event[query.mask])
    eligible_support = ~torch.isin(window, query_windows) & ~torch.isin(event, query_events)
    support_mask = torch.zeros(len(rows_cpu), dtype=torch.bool, device=device)
    support_candidate = torch.full(
        (len(rows_cpu),), -1, dtype=torch.long, device=device
    )
    realized = []
    for candidate_position, label in enumerate(candidates.tolist()):
        label_rows = y.eq(label) & eligible_support
        if episode_type == "cross_subject_few_support":
            # Person-disjointness only. Acquisition disjointness is deliberately NOT required: 35 of
            # the 93 corpus labels exist in exactly one stream, so demanding it would silently drop
            # them from this regime entirely. Support recorded on another stream is allowed and
            # happens naturally wherever the label supports it; how often it happened is measured by
            # describe_episode_composition rather than enforced here.
            query_rows = query_label.eq(label)
            related_query_mask = (
                query_rows.unsqueeze(1) & query.mask if truth_present else query.mask
            )
            query_subjects = torch.unique(query.subj[related_query_mask])
            label_rows &= ~torch.isin(subj, query_subjects)
        units = torch.unique(support_unit[label_rows])
        if len(units) < support_count:
            raise ValueError(
                f"candidate label {label} has {len(units)} eligible support units, "
                f"needs {support_count}"
            )
        if support_count:
            chosen = rng.choice(
                units.detach().cpu().numpy(), size=support_count, replace=False
            )
            chosen_t = torch.as_tensor(chosen, dtype=torch.long, device=device)
            selected = y.eq(label) & torch.isin(support_unit, chosen_t)
            support_mask |= selected
            support_candidate[selected] = candidate_position
        realized.append(support_count)

    B, Q = query.mask.shape
    allowed = query.mask.unsqueeze(-1).expand(B, Q, len(rows_cpu)).clone()
    allowed &= window.view(1, 1, -1).ne(query.window.unsqueeze(-1))
    allowed &= event.view(1, 1, -1).ne(query.event.unsqueeze(-1))
    # Candidate classes enter memory only through explicitly provided support. Background labels
    # remain available as distractors and as semantic bridges in coherent-label episodes.
    candidate_rows = torch.isin(y, candidates)
    allowed &= (~candidate_rows | support_mask).view(1, 1, -1)
    if not truth_present:
        # An unanswerable example must not leak its true concept through ordinary background rows.
        allowed &= y.view(1, 1, -1).ne(query_label.view(B, 1, 1))
    if not bool((allowed.any(-1) | ~query.mask).all()):
        raise ValueError("at least one query patch has no eligible episodic memory rows")
    return EpisodeMemoryView(
        allowed=allowed,
        support_mask=support_mask,
        support_candidate=support_candidate,
        candidate_ids=candidates,
        query_label=query_label,
        support_units_per_candidate=torch.tensor(realized, device=device),
        episode_type=episode_type,
        label_mode=label_mode,
    )


def balanced_memory_log_prior(patch: dict, index_rows: torch.Tensor, device) -> torch.Tensor:
    """A count-neutral prior over label -> window -> resolution -> represented duration."""
    rows = index_rows.detach().cpu().long()
    y = torch.as_tensor(patch["y"])[rows].long().cpu()
    window = torch.as_tensor(patch["window"])[rows].long().cpu()
    resolution = torch.as_tensor(patch["resolution"])[rows].long().cpu()
    duration = torch.as_tensor(patch["duration"])[rows].float().cpu().clamp_min(1e-6)
    weight = torch.zeros_like(duration)
    labels = torch.unique(y)
    for label in labels.tolist():
        label_rows = torch.nonzero(y.eq(label), as_tuple=True)[0]
        windows = torch.unique(window[label_rows])
        for source_window in windows.tolist():
            window_rows = label_rows[window[label_rows].eq(source_window)]
            resolutions = torch.unique(resolution[window_rows])
            for grid in resolutions.tolist():
                group = window_rows[resolution[window_rows].eq(grid)]
                weight[group] = (
                    duration[group] / duration[group].sum().clamp_min(1e-8)
                    / len(resolutions) / len(windows) / len(labels)
                )
    if not bool((weight > 0).all()):
        raise ValueError("failed to assign a positive prior to every active memory row")
    return weight.log().to(device)


def simultaneous_stream_pairs(
    window_cfg: torch.Tensor,
    window_event: torch.Tensor,
) -> set[frozenset]:
    """Stream pairs captured at the same instant, i.e. sharing verified event ids.

    xrf_v2's six placements and nfi_fared's back/wrist are recorded synchronously and therefore
    share an event; wisdm's phone and watch cover the same activities in separate sessions and
    share none. The distinction is invisible to the config id alone, so precompute it once and let
    :func:`describe_episode_composition` report which kind an episode actually drew.
    """
    cfg = window_cfg.detach().cpu().long()
    event = window_event.detach().cpu().long()
    order = torch.argsort(event, stable=True)
    events, counts = torch.unique_consecutive(event[order], return_counts=True)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    pairs: set[frozenset] = set()
    for position in torch.nonzero(counts > 1, as_tuple=True)[0].tolist():
        members = sorted({
            int(value) for value in
            cfg[order[int(offsets[position]):int(offsets[position + 1])]].tolist()
        })
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pairs.add(frozenset((a, b)))
    return pairs


def describe_episode_composition(
    patch: dict,
    index_rows: torch.Tensor,
    query: QueryPatches,
    view: EpisodeMemoryView,
    simultaneous_pairs: set[frozenset] | None = None,
) -> dict:
    """Plain-language account of what an episode's enrolled support actually consists of.

    Reports counts in whole recorded executions (and the patch rows they contribute) rather than
    enforcing anything, so an episode whose label only exists on one stream is described honestly
    instead of being excluded.
    """
    rows = index_rows.detach().cpu().long()
    support_local = view.support_rows.detach().cpu()
    subj = torch.as_tensor(patch["subj"])[rows].long()
    cfg = torch.as_tensor(patch["cfg"])[rows].long()
    unit = _support_units(patch, rows, torch.device("cpu"))
    candidate_of = view.support_candidate.detach().cpu()
    candidates = view.candidate_ids.detach().cpu().tolist()
    query_label = view.query_label.detach().cpu()
    q_mask = query.mask.detach().cpu()
    q_subj = query.subj.detach().cpu()
    q_cfg = query.cfg.detach().cpu()
    pairs = simultaneous_pairs or set()

    # NOTE on the stream categories: provided support can never be the query's own execution, so
    # these describe the RELATIONSHIP between the two streams, not whether these two particular
    # recordings happened at once. "worn simultaneously" means the corpus captures those streams
    # together as one rig (xrf_v2's placements, nfi_fared back+wrist); "independently recorded"
    # means the same activity was covered by both streams in unrelated sessions (wisdm's phone and
    # watch, or two different studies).
    counts = {
        "enrolled_examples": 0, "enrolled_patches": int(len(support_local)),
        "enrolled_examples_without_matching_query": 0,
        "performed_by_a_different_person": 0, "performed_by_the_same_person": 0,
        "from_the_query_s_own_stream": 0,
        "from_a_second_stream_worn_simultaneously": 0,
        "from_an_independently_recorded_stream": 0,
    }
    for position, label in enumerate(candidates):
        picked = support_local[candidate_of[support_local].eq(position)]
        if not len(picked):
            continue
        rows_for_label = q_mask & query_label.eq(int(label)).unsqueeze(1)
        has_matching_query = bool(rows_for_label.any())
        query_subjects = set(q_subj[rows_for_label].tolist()) if has_matching_query else set()
        query_configs = set(q_cfg[rows_for_label].tolist()) if has_matching_query else set()
        for execution in torch.unique(unit[picked]).tolist():
            member = picked[unit[picked].eq(execution)]
            counts["enrolled_examples"] += 1
            if not has_matching_query:
                # Distractor support has no corresponding query execution. Comparing it with an
                # unrelated query would overstate same-person and same-stream enrollment.
                counts["enrolled_examples_without_matching_query"] += 1
                continue
            key = ("performed_by_the_same_person"
                   if set(subj[member].tolist()) & query_subjects
                   else "performed_by_a_different_person")
            counts[key] += 1
            streams = set(cfg[member].tolist())
            if streams & query_configs:
                counts["from_the_query_s_own_stream"] += 1
            outside_streams = streams - query_configs
            paired_within_support = any(
                frozenset((left, right)) in pairs
                for left in streams for right in streams if left < right
            )
            paired_outside = {
                stream for stream in outside_streams
                if any(frozenset((stream, query_stream)) in pairs
                       for query_stream in query_configs)
            }
            if paired_within_support or paired_outside:
                counts["from_a_second_stream_worn_simultaneously"] += 1
            if any(stream not in paired_outside for stream in outside_streams):
                counts["from_an_independently_recorded_stream"] += 1
    counts["synthetic_persona"] = {
        "same_subject_enrollment": "one persona shared by the query and its support",
        "cross_subject_few_support": "a different persona for the query and its support",
    }.get(view.episode_type, "none applied")
    return counts


def support_capacity_by_label(
    patch: dict,
    index_rows: torch.Tensor,
    n_labels: int,
) -> torch.Tensor:
    """Count independent verified-event/window support identities per canonical label."""
    rows = index_rows.detach().cpu().long()
    labels = torch.as_tensor(patch["y"])[rows].long()
    units = _support_units(patch, rows, torch.device("cpu"))
    capacity = torch.zeros(n_labels, dtype=torch.long)
    for label in torch.unique(labels).tolist():
        capacity[int(label)] = torch.unique(units[labels.eq(label)]).numel()
    return capacity


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
        row=torch.full((B, Q), -1, dtype=torch.long, device=device),
    )


def build_allowed_mask(
    patch: dict,
    index_rows: torch.Tensor,
    query: QueryPatches,
    query_label: torch.Tensor,
    *,
    truth_present: bool,
    true_support: int | None,
    config_mode: str,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Apply leakage, configuration, memory-roster, and per-label support constraints.

    Candidate labels are intentionally absent from this function: they define what the model may
    answer, while the active index defines what memory actually contains.
    """
    device = query.Z.device
    rows = index_rows.detach().cpu()
    y = torch.as_tensor(patch["y"])[rows].long().to(device)
    subj = torch.as_tensor(patch["subj"])[rows].long().to(device)
    event = torch.as_tensor(patch["event"])[rows].long().to(device)
    window = torch.as_tensor(patch["window"])[rows].long().to(device)
    support_unit = _support_units(patch, rows, device)
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
    elif config_mode == "cross":
        allowed &= cfg.view(1, 1, -1).ne(query.cfg.unsqueeze(-1))
    elif config_mode == "query_absent":
        for batch_index in range(B):
            query_configs = torch.unique(query.cfg[batch_index, query.mask[batch_index]])
            allowed[batch_index] &= ~torch.isin(cfg, query_configs).view(1, -1)
    elif config_mode != "any":
        raise ValueError(f"unknown config_mode {config_mode!r}")

    # Support is sampled once per source example and shared across its patches. Verified
    # synchronous placements share an event-level support identity; unpaired data uses a window.
    for b in range(B):
        row_allowed = allowed[b].any(0)
        if not truth_present:
            row_allowed &= y.ne(query_label[b])
            allowed[b, :, y.eq(query_label[b])] = False
        active_labels = torch.unique(y[row_allowed]).tolist()
        for label in active_labels:
            # Only familiarity with the truth is an episode condition. Non-truth labels follow the
            # same balanced active-index policy and are not governed by a second support knob.
            cap = true_support if truth_present and label == int(query_label[b]) else None
            if cap is None:
                continue
            label_rows = torch.nonzero(row_allowed & y.eq(label), as_tuple=True)[0]
            label_units = torch.unique(support_unit[label_rows])
            if len(label_units) <= cap:
                continue
            keep_np = rng.choice(label_units.detach().cpu().numpy(), size=cap, replace=False) \
                if cap > 0 else np.empty(0, dtype=np.int64)
            keep_unit = torch.from_numpy(np.asarray(keep_np)).to(device)
            keep = torch.isin(support_unit, keep_unit) \
                if len(keep_np) else torch.zeros_like(row_allowed)
            remove = y.eq(label) & ~keep
            allowed[b, :, remove] = False
    return allowed


def realized_support_examples(
    patch: dict,
    index_rows: torch.Tensor,
    allowed: torch.Tensor,
    query_label: torch.Tensor,
) -> torch.Tensor:
    """Count distinct eligible true-label events/windows for each query episode row."""
    device = allowed.device
    rows = index_rows.detach().cpu()
    y = torch.as_tensor(patch["y"])[rows].long().to(device)
    support_unit = _support_units(patch, rows, device)
    counts = []
    for b in range(allowed.shape[0]):
        eligible = allowed[b].any(0) & y.eq(query_label[b])
        counts.append(torch.unique(support_unit[eligible]).numel())
    return torch.tensor(counts, dtype=torch.long, device=device)


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
    # Roster selection is deliberately non-differentiable. Copy its bounded top-k state once;
    # scalar reads from CUDA inside this loop would otherwise synchronize once per candidate.
    retrieval_index_cpu = retrieval.index.detach().cpu()
    retrieval_valid_cpu = retrieval.valid.detach().cpu()
    selected_global_cpu = index_rows.detach().cpu()[retrieval_index_cpu]
    detached_scores_cpu = online_scores.detach().float().cpu()
    patch_window = torch.as_tensor(patch["window"], dtype=torch.long).cpu()
    patch_y = torch.as_tensor(patch["y"], dtype=torch.long).cpu()
    chosen: list[list[tuple[int, float, int, int]]] = []
    for b in range(B):
        candidates = []
        for q in range(Q):
            for h in range(H):
                for j in range(K):
                    if bool(retrieval_valid_cpu[b, q, h, j]):
                        candidates.append((
                            int(selected_global_cpu[b, q, h, j]),
                            float(detached_scores_cpu[b, q, h, j]),
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

    evidence = EvidenceBatch(
        index=index, weights=torch.zeros_like(scores), scores=scores, mask=mask,
        head=head, query_patch=query_patch,
        local_index=torch.zeros(B, E, dtype=torch.long, device=device),
    )
    for b, accepted in enumerate(chosen):
        n = len(accepted)
        evidence.local_index[b, :n] = torch.tensor(
            [int(retrieval_index_cpu[b, value[3], value[2], value[4]]) for value in accepted],
            device=device,
        )
    return reweight_evidence(evidence, scores, patch, tau=tau)


def reweight_evidence(
    evidence: EvidenceBatch,
    scores: torch.Tensor,
    patch: dict,
    *,
    tau: float,
) -> EvidenceBatch:
    """Normalize live pair scores without changing the detached selected evidence roster."""
    if scores.shape != evidence.index.shape:
        raise ValueError("evidence scores must align with assembled evidence")
    device = scores.device
    patch_resolution = torch.as_tensor(patch["resolution"], device=device, dtype=torch.long)
    patch_duration = torch.as_tensor(patch["duration"], device=device, dtype=torch.float32)
    weights = torch.zeros_like(scores)
    resolution = patch_resolution[evidence.index]
    duration = patch_duration[evidence.index]
    B = scores.shape[0]
    for b in range(B):
        group_ids = evidence.head[b] * 3 + (resolution[b].clamp(-1, 1) + 1)
        active_groups = torch.unique(group_ids[evidence.mask[b]])
        for group in active_groups:
            select = evidence.mask[b] & group_ids.eq(group)
            raw = torch.softmax(scores[b, select] / tau, dim=0) * duration[b, select]
            raw = raw / raw.sum().clamp_min(1e-8)
            weights[b, select] = raw / len(active_groups)
    return EvidenceBatch(
        index=evidence.index, weights=weights, scores=scores, mask=evidence.mask,
        head=evidence.head, query_patch=evidence.query_patch,
        local_index=evidence.local_index,
    )
