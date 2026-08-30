import json
from pathlib import Path

from applications.motion_monitoring.data.annotation_inventory import build_inventory


INVENTORY = Path("applications/motion_monitoring/data/ANNOTATION_INVENTORY.json")


def test_inventory_distinguishes_instance_boundaries_from_coarse_annotations():
    datasets = build_inventory()["datasets"]
    assert datasets["c_mhad"]["exact_instance_intervals"] is True
    assert datasets["openpack"]["exact_instance_intervals"] is True
    assert datasets["recofit"]["exact_instance_intervals"] is False
    assert datasets["aidlab_har"]["exact_instance_intervals"] is False
    assert datasets["xrf_v2"]["adapter_gap"]


def test_measured_cache_inventory_contains_real_event_statistics():
    datasets = build_inventory()["datasets"]
    assert datasets["c_mhad"]["application_cache"]["event_count"] == 2039
    assert (
        datasets["openpack"]["application_cache"]["event_kinds"]["fine_action"] > 50_000
    )
    assert (
        datasets["oca"]["application_cache"]["recordings_with_repeated_label_instances"]
        > 0
    )


def test_tracked_annotation_inventory_is_current():
    assert json.loads(INVENTORY.read_text()) == build_inventory()
