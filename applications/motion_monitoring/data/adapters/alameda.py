"""Streaming adapter for the quality-controlled ALAMEDA PD subset."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd

from applications.motion_monitoring.data.contracts import RawRecording, SensorStream


_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sources" / "alameda"
_ARCHIVE_NAME = "pd_geneactiv_dataset.zip"
_CLINICAL_WORKBOOK = "PD GeneActiv Dataset/Clinical Annotations/ALAMEDA-CRF-PD-Pilot.xlsx"
_PREFIX = "PD GeneActiv Dataset/GeneActiv Recordings/"
_RATE_HZ = 100.0
_CHANNELS = ("acc_x", "acc_y", "acc_z")

# These are the four campaigns that survived calibration, known-placement,
# duplicate-identity, and nearby-clinical-visit screening documented in
# docs/data/APPLICATION_DATASETS.md. Do not silently broaden this set.
ELIGIBLE_CAMPAIGNS = frozenset(
    {
        "11_tAteNDLe_right_wrist_066530_2023-10-03_2023-10-10",
        "11_tAteNDLe_right_wrist_066544_2023-02-02_2023-02-09",
        "4_UMustRop_left_wrist_M3_2023-02-02",
        "4_UMustRop_left_wrist_M6_2023-06-14",
    }
)


def _archive_path(root: Path | None) -> Path:
    source = _DEFAULT_ROOT if root is None else Path(root)
    candidates = (source / "downloads" / _ARCHIVE_NAME, source / _ARCHIVE_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"ALAMEDA archive not found beneath {source}")


def _campaign(member: str) -> str:
    parts = PurePosixPath(member).parts
    if len(parts) < 4 or not member.startswith(_PREFIX):
        raise ValueError(f"unexpected ALAMEDA member path: {member}")
    return parts[2]


def _campaign_fields(campaign: str) -> tuple[str, str, str]:
    parts = campaign.split("_")
    if len(parts) < 5:
        raise ValueError(f"invalid ALAMEDA campaign name: {campaign}")
    participant = parts[0]
    placement_index = next(
        (
            index
            for index in range(2, len(parts) - 1)
            if parts[index : index + 2] in (["left", "wrist"], ["right", "wrist"])
        ),
        None,
    )
    if placement_index is None:
        raise ValueError(f"ALAMEDA campaign has no known wrist placement: {campaign}")
    placement = "_".join(parts[placement_index : placement_index + 2])
    pseudonym = parts[1]
    return participant, pseudonym, placement


def _relative_timestamps(index: pd.Index) -> tuple[np.ndarray, str]:
    """Preserve the released datetime clock while exposing stable relative seconds."""

    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
        raise ValueError("ALAMEDA Parquet index must be a timezone-naive DatetimeIndex")
    nanoseconds = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if len(nanoseconds) < 2 or np.any(np.diff(nanoseconds) <= 0):
        raise ValueError(
            "ALAMEDA source clock must be non-empty and strictly increasing"
        )
    timestamps = (nanoseconds - nanoseconds[0]).astype(np.float64) / 1e9
    return timestamps, index[0].isoformat()


@lru_cache(maxsize=2)
def _clinical_visits(archive_name: str) -> pd.DataFrame:
    """Load only documented visit-level motor totals from the release archive."""

    with ZipFile(archive_name) as archive:
        visits = pd.read_excel(BytesIO(archive.read(_CLINICAL_WORKBOOK)), sheet_name="Visits")
        motor = pd.read_excel(BytesIO(archive.read(_CLINICAL_WORKBOOK)), sheet_name="UPDRS_3")
    merged = visits[["VisitID", "PID", "VisitDate", "Phase"]].merge(
        motor[["VisitID", "UPDRS_3_total"]], on="VisitID", how="left", validate="one_to_one"
    )
    merged["VisitDate"] = pd.to_datetime(merged["VisitDate"], errors="raise").dt.normalize()
    return merged


def _nearest_clinical_visit(
    visits: pd.DataFrame, *, participant: str, source_date: str, max_days: int = 14
) -> dict[str, object]:
    """Return a source-documented clinical association, never an inferred one."""

    observation_date = pd.Timestamp(source_date).normalize()
    candidates = visits.loc[visits["PID"] == int(participant)].copy()
    if candidates.empty:
        return {"clinical_linkage_status": "no_participant_visit"}
    candidates["days_from_observation"] = (
        candidates["VisitDate"] - observation_date
    ).abs().dt.days
    nearest = candidates.sort_values(["days_from_observation", "VisitID"]).iloc[0]
    distance = int(nearest["days_from_observation"])
    if distance > max_days:
        return {
            "clinical_linkage_status": "no_visit_within_14_days",
            "nearest_clinical_visit_days": distance,
        }
    score = pd.to_numeric(pd.Series([nearest["UPDRS_3_total"]]), errors="coerce").iloc[0]
    return {
        "clinical_linkage_status": "linked_within_14_days",
        "clinical_visit_id": int(nearest["VisitID"]),
        "clinical_visit_date": pd.Timestamp(nearest["VisitDate"]).date().isoformat(),
        "clinical_visit_phase": str(nearest["Phase"]),
        "clinical_visit_days": distance,
        "updrs_3_total": None if pd.isna(score) else float(score),
    }


def iter_recordings(
    root: Path | None = None,
    limit: int | None = None,
) -> Iterator[RawRecording]:
    """Yield selected daily observations directly from the compressed release."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative or None")
    archive_path = _archive_path(root)
    yielded = 0
    with ZipFile(archive_path) as archive:
        clinical_visits = _clinical_visits(str(archive_path))
        members = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".parquet") and _campaign(name) in ELIGIBLE_CAMPAIGNS
        )
        metadata = {
            _campaign(name): json.loads(archive.read(name))
            for name in archive.namelist()
            if name.endswith("_meta.json") and _campaign(name) in ELIGIBLE_CAMPAIGNS
        }
        if set(metadata) != set(ELIGIBLE_CAMPAIGNS):
            missing = sorted(set(ELIGIBLE_CAMPAIGNS) - set(metadata))
            raise ValueError(f"ALAMEDA eligible campaign metadata missing: {missing}")
        for member in members:
            if limit is not None and yielded >= limit:
                break
            campaign = _campaign(member)
            participant, pseudonym, placement = _campaign_fields(campaign)
            frame = pd.read_parquet(
                BytesIO(archive.read(member)), columns=["x", "y", "z"]
            )
            values = frame[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
            valid = np.isfinite(values)
            timestamps, source_start = _relative_timestamps(frame.index)
            source_meta = metadata[campaign]
            date = PurePosixPath(member).stem.rsplit("_data_", 1)[-1]
            clinical_metadata = _nearest_clinical_visit(
                clinical_visits, participant=participant, source_date=date
            )
            yield RawRecording(
                dataset="alameda",
                recording_id=f"alameda:{campaign}:{date}",
                subject_id=f"participant_{participant}",
                session_id=campaign,
                streams=(
                    SensorStream(
                        stream_id=f"geneactiv_{placement}",
                        placement=placement,
                        device="GENEActiv wrist accelerometer",
                        timestamps_sec=timestamps,
                        values=np.nan_to_num(values),
                        channels=_CHANNELS,
                        valid=valid,
                        gravity_state="present",
                        nominal_rate_hz=_RATE_HZ,
                        metadata={
                            "source_member": member,
                            "source_acceleration_unit": "g",
                            "calibration_ok": bool(source_meta.get("CalibOK", 0)),
                            "device_id": str(source_meta.get("DeviceID", "")),
                            "source_start_datetime": source_start,
                        },
                    ),
                ),
                events=(),
                split="evaluation",
                metadata={
                    "pseudonym": pseudonym,
                    "campaign": campaign,
                    "source_date": date,
                    "observation_kind": "free_living_day",
                    "bounded_execution_annotations": False,
                    **clinical_metadata,
                },
            )
            yielded += 1
