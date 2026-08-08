"""Construct an inference-time enrollment overlay without changing model weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from training.evidence.patch_episodes import EpisodeMemoryView, QueryPatches


@dataclass
class EnrollmentMemory:
    bank: dict
    index_rows: torch.Tensor
    selector_z: torch.Tensor
    support_mask: torch.Tensor
    support_candidate: torch.Tensor
    candidate_ids: torch.Tensor
    canonical_text: torch.Tensor
    candidate_text: torch.Tensor
    support_units_per_candidate: torch.Tensor

    def episode_view(
        self,
        query: QueryPatches,
        query_candidate_position: torch.Tensor,
        *,
        label_mode: str,
    ) -> EpisodeMemoryView:
        query_candidate_position = query_candidate_position.long().to(query.Z.device)
        query_label = self.candidate_ids.to(query.Z.device)[query_candidate_position]
        allowed = query.mask.unsqueeze(-1).expand(
            query.mask.shape[0], query.mask.shape[1], len(self.index_rows)
        ).clone()
        return EpisodeMemoryView(
            allowed=allowed,
            support_mask=self.support_mask.to(query.Z.device),
            support_candidate=self.support_candidate.to(query.Z.device),
            candidate_ids=self.candidate_ids.to(query.Z.device),
            query_label=query_label,
            support_units_per_candidate=self.support_units_per_candidate.to(query.Z.device),
            episode_type="runtime_enrollment",
            label_mode=label_mode,
        )


def build_enrollment_memory(
    base_bank: dict,
    base_index_rows: torch.Tensor,
    base_selector_z: torch.Tensor,
    encoded: dict[str, torch.Tensor],
    support_windows: torch.Tensor,
    support_candidate_position: torch.Tensor,
    canonical_text: torch.Tensor,
    candidate_text: torch.Tensor,
) -> EnrollmentMemory:
    """Append labeled external support patches to a detached active corpus view."""
    device = base_selector_z.device
    support_windows = support_windows.detach().cpu().long()
    support_candidate_position = support_candidate_position.detach().cpu().long()
    if len(support_windows) != len(support_candidate_position):
        raise ValueError("support windows and candidate positions must align")
    if torch.unique(support_windows).numel() != len(support_windows):
        raise ValueError("an enrollment source window may be assigned only once")
    n_candidates = candidate_text.shape[0]
    if not len(support_windows) or bool((support_candidate_position < 0).any()) \
            or bool((support_candidate_position >= n_candidates).any()):
        raise ValueError("runtime enrollment needs valid support for at least one candidate")
    counts = torch.bincount(support_candidate_position, minlength=n_candidates)
    if bool((counts != counts[0]).any()):
        raise ValueError("every runtime candidate must receive the same support count")

    patch_window = encoded["patch_window"].detach().cpu().long()
    selected = torch.isin(patch_window, support_windows)
    if not bool(selected.any()):
        raise ValueError("selected enrollment windows exported no valid patches")
    selected_rows = torch.nonzero(selected, as_tuple=True)[0]
    selected_parent = patch_window[selected_rows]
    position_by_window = {
        int(window): int(position)
        for window, position in zip(
            support_windows.tolist(), support_candidate_position.tolist()
        )
    }
    patch_candidate = torch.tensor([
        position_by_window[int(window)] for window in selected_parent.tolist()
    ], dtype=torch.long)

    base_rows = base_index_rows.detach().cpu().long()
    base_patch = base_bank["patch"]
    n_base = len(base_rows)
    n_support = len(selected_rows)
    vocab_size = canonical_text.shape[0]
    candidate_ids = torch.arange(
        vocab_size, vocab_size + n_candidates, dtype=torch.long, device=device
    )
    max_window = int(torch.as_tensor(base_patch["window"])[base_rows].max()) + 1
    support_window_id = torch.tensor([
        max_window + support_windows.tolist().index(int(window))
        for window in selected_parent.tolist()
    ], dtype=torch.long)
    ordinal = torch.zeros(n_support, dtype=torch.long)
    seen: dict[int, int] = {}
    for i, window in enumerate(selected_parent.tolist()):
        ordinal[i] = seen.get(int(window), 0)
        seen[int(window)] = int(ordinal[i]) + 1

    def base(name, dtype=None):
        value = torch.as_tensor(base_patch[name])[base_rows]
        return value.to(dtype=dtype) if dtype is not None else value

    support_y = candidate_ids.detach().cpu()[patch_candidate]
    support_cfg_id = int(base("cfg", torch.long).max()) + 1
    temporary_patch = {
        "Z": torch.cat([
            base("Z").cpu(), encoded["patch_Z"][selected_rows].detach().cpu().half()
        ]),
        "y": torch.cat([base("y", torch.long).cpu(), support_y]),
        "subj": torch.cat([
            base("subj", torch.long).cpu(),
            torch.full((n_support,), int(base("subj", torch.long).max()) + 1),
        ]),
        "cfg": torch.cat([
            base("cfg", torch.long).cpu(),
            torch.full((n_support,), support_cfg_id, dtype=torch.long),
        ]),
        "sensor": torch.cat([
            base("sensor", torch.long).cpu(),
            torch.full((n_support,), support_cfg_id, dtype=torch.long),
        ]),
        "window": torch.cat([base("window", torch.long).cpu(), support_window_id]),
        "event": torch.cat([base("event", torch.long).cpu(), support_window_id]),
        "event_verified": torch.cat([
            base("event_verified", torch.bool).cpu(),
            torch.zeros(n_support, dtype=torch.bool),
        ]),
        "time": torch.cat([
            base("time", torch.float32).cpu(),
            encoded["patch_time"][selected_rows].detach().cpu().float(),
        ]),
        "duration": torch.cat([
            base("duration", torch.float32).cpu(),
            encoded["patch_duration"][selected_rows].detach().cpu().float(),
        ]),
        "resolution": torch.cat([
            base("resolution", torch.long).cpu(),
            encoded["patch_resolution"][selected_rows].detach().cpu().long(),
        ]),
        "ordinal": torch.cat([base("ordinal", torch.long).cpu(), ordinal]),
    }
    selector_z = F.normalize(torch.cat([
        base_selector_z,
        encoded["patch_Z"][selected_rows.to(encoded["patch_Z"].device)].detach().to(device).float(),
    ]), dim=-1)
    support_mask = torch.zeros(n_base + n_support, dtype=torch.bool, device=device)
    support_mask[n_base:] = True
    local_candidate = torch.full(
        (n_base + n_support,), -1, dtype=torch.long, device=device
    )
    local_candidate[n_base:] = patch_candidate.to(device)
    return EnrollmentMemory(
        bank={"patch": temporary_patch},
        index_rows=torch.arange(n_base + n_support, dtype=torch.long),
        selector_z=selector_z,
        support_mask=support_mask,
        support_candidate=local_candidate,
        candidate_ids=candidate_ids,
        canonical_text=torch.cat([canonical_text, candidate_text], dim=0),
        candidate_text=candidate_text,
        support_units_per_candidate=counts.to(device),
    )
