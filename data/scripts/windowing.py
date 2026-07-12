"""
Variable-length windowing for HAR datasets.

Provides activity-aware window sizing with random overlap for training diversity.
Window durations are tuned based on activity complexity:
- Simple activities (walking, sitting): 2-20s windows
- Complex activities (cooking, cleaning): 10-90s windows

The max session duration is capped at 90 seconds to fit within the model's
context window (~45 patches at ~2 seconds each).
"""

import random
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

# ============================================================================
# Activity-Based Window Duration Ranges (in seconds)
# ============================================================================

ACTIVITY_WINDOW_RANGES = {
    # Simple/repetitive activities - short windows sufficient
    "walking": (2, 20),
    "standing": (2, 15),
    "sitting": (2, 15),
    "lying": (2, 15),
    "laying": (2, 15),  # UCI-HAR uses "laying"
    "running": (2, 15),
    "jogging": (2, 15),
    "cycling": (3, 20),

    # Stair activities - medium duration
    "walking_upstairs": (3, 20),
    "walking_downstairs": (3, 20),
    "ascending_stairs": (3, 20),
    "descending_stairs": (3, 20),
    "climbing_stairs": (3, 20),
    "stairs": (3, 20),
    "going_up_stairs": (3, 20),
    "going_down_stairs": (3, 20),

    # Exercise activities - medium duration
    "nordic_walking": (5, 30),
    "rope_jumping": (5, 30),
    "jump_front_back": (3, 15),
    "jumping": (3, 15),
    "frontal_elevation_arms": (3, 15),
    "knees_bending": (3, 15),
    "waist_bends_forward": (3, 15),

    # Hand activities - medium duration
    "brushing_teeth": (5, 30),
    "typing": (5, 30),
    "writing": (5, 30),
    "clapping": (2, 10),
    "folding_clothes": (10, 45),
    "dribbling": (3, 20),
    "kicking": (2, 15),
    "playing_catch": (3, 20),

    # Eating activities - medium-long duration
    "eating": (10, 60),
    "eating_pasta": (10, 60),
    "eating_chips": (10, 45),
    "eating_sandwich": (10, 60),
    "eating_soup": (10, 60),
    "drinking": (5, 30),

    # Complex/compound activities - longer windows needed
    "ironing": (10, 60),
    "vacuum_cleaning": (10, 60),
    "cooking": (15, 90),
    "cleaning": (15, 90),

    # Transitions - short duration
    "standing_up_from_sitting": (2, 10),
    "standing_up_from_laying": (2, 10),
    "sitting_down_from_standing": (2, 10),
    "lying_down_from_standing": (2, 10),
    # HAPT postural transitions
    "stand_to_sit": (2, 8),
    "sit_to_stand": (2, 8),
    "sit_to_lie": (2, 8),
    "lie_to_sit": (2, 8),
    "stand_to_lie": (2, 8),
    "lie_to_stand": (2, 8),

    # KU-HAR activities
    "picking_up": (2, 10),
    "push_up": (3, 20),
    "sit_up": (3, 20),
    "table_tennis": (5, 30),
    "walking_backwards": (5, 30),
    "talking_sitting": (10, 60),
    "talking_standing": (10, 60),

    # Gym exercises (RecGym) - medium duration
    "squat": (5, 30),
    "bench_press": (5, 30),
    "leg_press": (5, 30),
    "leg_curl": (5, 30),
    "arm_curl": (5, 30),
    "adductor_machine": (5, 30),
    "stairclimber": (5, 45),
    "rope_skipping": (5, 30),

    # Falls (MobiAct) - very short
    "fall_forward": (2, 8),
    "fall_backward_knees": (2, 8),
    "fall_backward_sitting": (2, 8),
    "fall_sideways": (2, 8),

    # UniMiB SHAR falls (very short events)
    "falling_forward": (2, 8),
    "falling_backward": (2, 8),
    "falling_left": (2, 8),
    "falling_right": (2, 8),
    "falling_hitting_obstacle": (2, 8),
    "falling_with_protection": (2, 8),
    "falling_backward_sitting": (2, 8),
    "syncope": (2, 8),
    "sitting_down": (2, 10),

    # Vehicle entry (MobiAct) - short
    "car_step_in": (2, 10),
    "car_step_out": (2, 10),
    "sitting_chair": (2, 15),

    # Construction work (VTT-ConIoT) - medium duration
    "carrying": (5, 30),
    "lifting": (3, 20),
    "pushing_cart": (5, 30),
    "climbing_ladder": (5, 30),
    "kneeling_work": (5, 45),
    "standing_work": (5, 45),
    "roll_painting": (10, 60),
    "spraying_paint": (10, 60),
    "leveling_paint": (10, 60),
    "raising_hands": (3, 20),
    "walking_straight": (2, 20),
    "walking_winding": (2, 20),
    "laying_back": (2, 15),

    # DSADS treadmill/gym activities
    "lying_back": (2, 15),
    "lying_right_side": (2, 15),
    "stairs_up": (3, 20),
    "stairs_down": (3, 20),
    "walking_parking": (2, 20),
    "walking_treadmill_flat": (2, 20),
    "walking_treadmill_incline": (2, 20),
    "running_treadmill": (2, 15),
    "exercising_stepper": (5, 30),
    "exercising_cross_trainer": (5, 30),
    "cycling_horizontal": (3, 20),
    "cycling_vertical": (3, 20),
    "playing_basketball": (5, 30),
    "moving_elevator": (5, 30),
    "standing_elevator": (5, 30),

    # REALDISP fitness exercises (A1-A33)
    "jump_up": (2, 10),
    # jump_front_back already defined above (3, 15) for MHEALTH — reused by REALDISP
    "jump_sideways": (2, 10),
    "jump_legs_arms": (2, 10),
    "jump_rope": (3, 20),
    "trunk_twist_arms_out": (3, 15),
    "trunk_twist_elbows_bent": (3, 15),
    # waist_bends_forward already defined above (3, 15) for MHEALTH — reused by REALDISP
    "waist_rotation": (3, 15),
    "waist_bend_cross": (3, 15),
    "reach_heels_backwards": (3, 15),
    "lateral_bend": (3, 15),
    "lateral_bend_arm_up": (3, 15),
    "forward_stretching": (3, 20),
    "upper_lower_twist": (3, 15),
    "lateral_arm_elevation": (3, 15),
    "frontal_arm_elevation": (3, 15),
    "frontal_hand_claps": (3, 15),
    "frontal_crossing_arms": (3, 15),
    "shoulders_high_rotation": (3, 15),
    "shoulders_low_rotation": (3, 15),
    "arms_inner_rotation": (3, 15),
    "knees_to_breast": (3, 15),
    "heels_to_backside": (3, 15),
    "knees_bending_crouching": (3, 15),
    "knees_alternating_forward": (3, 15),
    "rotation_on_knees": (3, 15),
    "rowing": (5, 30),
    "elliptical_bike": (5, 30),

    # Daphnet FoG - gait/freeze episodes
    "freezing_gait": (2, 15),

    # USC-HAD activities
    "walking_forward": (2, 20),
    "walking_left": (2, 20),
    "walking_right": (2, 20),
    "running_forward": (2, 15),
    "jumping_up": (2, 10),
    "sleeping": (5, 30),
    "elevator_up": (5, 30),
    "elevator_down": (5, 30),

    # HARTH activities
    "shuffling": (2, 15),
    "cycling_sit": (3, 20),
    "cycling_stand": (3, 20),
    "transport_sit": (5, 30),
    "transport_stand": (5, 30),
}

# Maximum session duration to fit in context window (45 patches × ~2s)
MAX_SESSION_DURATION = 90  # seconds

# Minimum window size (below this, data is kept as single session)
MIN_WINDOW_SIZE = 2  # seconds


def get_window_range(activity: str) -> Tuple[float, float]:
    """
    Get (min_sec, max_sec) duration range for an activity.

    Args:
        activity: Activity label (e.g., "walking", "eating_pasta")

    Returns:
        Tuple of (min_duration, max_duration) in seconds
    """
    # Normalize activity name
    activity_normalized = activity.lower().strip()

    if activity_normalized in ACTIVITY_WINDOW_RANGES:
        return ACTIVITY_WINDOW_RANGES[activity_normalized]

    # Default range for unknown activities
    return (2, 30)


def create_variable_windows(
    df: pd.DataFrame,
    session_prefix: str,
    activity: str,
    sample_rate: float,
    min_overlap: float = 0.5,
    max_overlap: float = 0.8,
    seed: Optional[int] = None,
) -> List[Tuple[str, pd.DataFrame, str]]:
    """
    Split a recording into variable-length windows based on activity type.

    Window durations are randomly sampled from activity-appropriate ranges.
    Overlap between windows is also randomized for variety.

    Args:
        df: DataFrame with sensor data (assumes standard column names)
        session_prefix: Base session ID (e.g., "user01_walking")
        activity: Activity label for window duration lookup
        sample_rate: Sampling rate in Hz
        min_overlap: Minimum overlap ratio between windows (0.5 = 50%)
        max_overlap: Maximum overlap ratio between windows (0.8 = 80%)
        seed: Optional random seed for reproducibility

    Returns:
        List of tuples: (session_id, window_df, activity)

    Example:
        >>> windows = create_variable_windows(
        ...     df=sensor_df,
        ...     session_prefix="user01_walking",
        ...     activity="walking",
        ...     sample_rate=50.0
        ... )
        >>> # Returns: [("user01_walking_0000", df0, "walking"), ...]
    """
    if seed is not None:
        random.seed(seed)

    total_samples = len(df)
    total_duration = total_samples / sample_rate

    # Get duration range for this activity
    min_dur, max_dur = get_window_range(activity)
    max_dur = min(max_dur, MAX_SESSION_DURATION)  # Cap at context window limit

    # If recording shorter than min_dur, keep as single session
    if total_duration <= min_dur:
        return [(session_prefix, df.copy(), activity)]

    windows = []
    position = 0
    win_idx = 0

    while position < total_samples:
        # Random duration for this window
        duration = random.uniform(min_dur, max_dur)
        window_samples = int(duration * sample_rate)

        end = min(position + window_samples, total_samples)

        # Skip if remaining data too short
        remaining_samples = end - position
        if remaining_samples < MIN_WINDOW_SIZE * sample_rate:
            break

        window_df = df.iloc[position:end].copy()
        session_id = f"{session_prefix}_{win_idx:04d}"
        windows.append((session_id, window_df, activity))

        # Stride: move by (1 - overlap) of window size
        # Random overlap for variety
        overlap = random.uniform(min_overlap, max_overlap)
        stride = int(window_samples * (1 - overlap))
        stride = max(stride, int(MIN_WINDOW_SIZE * sample_rate))  # Ensure minimum stride

        position += stride
        win_idx += 1

    return windows


def save_windows(
    windows: List[Tuple[str, pd.DataFrame, str]],
    output_dir: Path,
    labels_dict: dict,
) -> int:
    """
    Save windowed sessions to output directory.

    Args:
        windows: List of (session_id, df, activity) tuples
        output_dir: Path to sessions output directory
        labels_dict: Dictionary to update with session labels

    Returns:
        Number of windows saved
    """
    sessions_dir = output_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    for session_id, window_df, activity in windows:
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        data_path = session_dir / "data.parquet"
        window_df.to_parquet(data_path, index=False)

        # Update labels
        labels_dict[session_id] = activity

    return len(windows)


def estimate_window_count(
    total_duration_sec: float,
    activity: str,
    avg_overlap: float = 0.65,
) -> int:
    """
    Estimate number of windows that will be created from a recording.

    Args:
        total_duration_sec: Total recording duration in seconds
        activity: Activity label for window duration lookup
        avg_overlap: Average overlap ratio (default 0.65 = midpoint of 0.5-0.8)

    Returns:
        Estimated number of windows
    """
    min_dur, max_dur = get_window_range(activity)
    max_dur = min(max_dur, MAX_SESSION_DURATION)
    avg_dur = (min_dur + max_dur) / 2

    if total_duration_sec <= min_dur:
        return 1

    stride = avg_dur * (1 - avg_overlap)
    return max(1, int((total_duration_sec - avg_dur) / stride) + 1)


if __name__ == "__main__":
    # Test the windowing functions
    import numpy as np

    print("=" * 70)
    print("Variable-Length Windowing Test")
    print("=" * 70)

    # Test window range lookup
    print("\nWindow ranges for various activities:")
    test_activities = ["walking", "sitting", "eating_pasta", "vacuum_cleaning", "cooking", "unknown_activity"]
    for activity in test_activities:
        min_dur, max_dur = get_window_range(activity)
        print(f"  {activity}: {min_dur}-{max_dur} seconds")

    # Test windowing on synthetic data
    print("\nTest windowing on synthetic 3-minute recording (50Hz):")
    sample_rate = 50.0
    duration_sec = 180  # 3 minutes
    n_samples = int(duration_sec * sample_rate)

    # Create dummy sensor data
    test_df = pd.DataFrame({
        "acc_x": np.random.randn(n_samples),
        "acc_y": np.random.randn(n_samples),
        "acc_z": np.random.randn(n_samples),
    })

    test_cases = ["walking", "eating_pasta", "vacuum_cleaning"]
    for activity in test_cases:
        windows = create_variable_windows(
            df=test_df,
            session_prefix=f"test_{activity}",
            activity=activity,
            sample_rate=sample_rate,
            seed=42,
        )

        avg_duration = np.mean([len(w[1]) / sample_rate for w in windows])
        print(f"\n  {activity}:")
        print(f"    Windows created: {len(windows)}")
        print(f"    Avg window duration: {avg_duration:.1f}s")
        print(f"    Window durations: {[f'{len(w[1])/sample_rate:.1f}s' for w in windows[:5]]}...")
