"""Channel-name utilities shared across the data pipeline.

Pure helpers over standardized channel names (`acc_x`, `hand_gyro_z`, ...). Kept here (not in the
loader) so both the assembly/augmentation code and the loader depend on one implementation.
"""

from __future__ import annotations

import re
from typing import Dict, List

_AXIS = re.compile(r"_([xyz]|[1-4])$")


def group_channels_by_sensor(channel_names: List[str]) -> Dict[str, List[str]]:
    """Group channels by sensor prefix, dropping the axis suffix.

    `acc_x/acc_y/acc_z -> "acc"`; `hand_gyro_x/... -> "hand_gyro"`. Channels within a group are sorted
    for deterministic ordering (x before y before z / 1..4).
    """
    groups: Dict[str, List[str]] = {}
    for channel in channel_names:
        m = _AXIS.search(channel)
        group_name = channel[: m.start()] if m else channel
        groups.setdefault(group_name, []).append(channel)
    for group_name in groups:
        groups[group_name] = sorted(groups[group_name])
    return groups
