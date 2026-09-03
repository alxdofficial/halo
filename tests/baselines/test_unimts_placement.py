from types import SimpleNamespace

from baselines.unimts.adapter import _joint_for


def test_kneepad_leg_streams_preserve_side_and_segment() -> None:
    expected = {
        "left_rectus_femoris": 1,
        "left_hamstrings": 1,
        "left_tibialis_anterior": 2,
        "left_gastrocnemius": 2,
        "right_rectus_femoris": 5,
        "right_hamstrings": 5,
        "right_tibialis_anterior": 6,
        "right_gastrocnemius": 6,
    }
    for stream, joint in expected.items():
        assert _joint_for(SimpleNamespace(dataset="kneepad", stream=stream)) == joint
