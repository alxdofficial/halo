"""Leakage-safe grouping and validation for application task manifests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from applications.motion_monitoring.data.contracts import RawRecording


RecordingKey = tuple[str, str]


def recording_key(recording: RawRecording) -> RecordingKey:
    return recording.dataset, recording.recording_id


def subject_leakage_group(recording: RawRecording) -> str:
    """Return the indivisible unit for subject-disjoint application splits.

    A source adapter may provide a conservative linkage group when released session
    identifiers are known to repeat people but the directional identity mapping is
    unresolved. Otherwise the adapter's canonical subject ID is the grouping unit.
    """

    linkage = recording.metadata.get("identity_linkage_group")
    if linkage is not None and str(linkage).strip():
        return f"{recording.dataset}:linkage:{str(linkage).strip()}"
    return f"{recording.dataset}:subject:{recording.subject_id}"


def validate_subject_disjoint_assignments(
    recordings: Iterable[RawRecording],
    assignments: Mapping[RecordingKey, str],
) -> dict[str, tuple[RecordingKey, ...]]:
    """Reject missing assignments or any leakage group spanning multiple splits."""

    groups: dict[str, list[RecordingKey]] = defaultdict(list)
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    seen: set[RecordingKey] = set()
    for recording in recordings:
        key = recording_key(recording)
        if key in seen:
            raise ValueError(f"duplicate recording identity in manifest: {key!r}")
        seen.add(key)
        if key not in assignments:
            raise ValueError(f"manifest has no split assignment for {key!r}")
        split = str(assignments[key]).strip()
        if not split:
            raise ValueError(f"manifest has an empty split assignment for {key!r}")
        group = subject_leakage_group(recording)
        groups[group].append(key)
        splits_by_group[group].add(split)

    if not seen:
        raise ValueError("subject-disjoint validation requires recordings")
    extra = set(assignments) - seen
    if extra:
        raise ValueError(f"manifest contains unknown recording assignments: {sorted(extra)!r}")
    leaking = {
        group: sorted(splits)
        for group, splits in splits_by_group.items()
        if len(splits) > 1
    }
    if leaking:
        details = "; ".join(
            f"{group} -> {splits}" for group, splits in sorted(leaking.items())
        )
        raise ValueError(f"subject leakage groups cross splits: {details}")
    return {group: tuple(keys) for group, keys in sorted(groups.items())}
