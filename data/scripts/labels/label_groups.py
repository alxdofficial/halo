"""
Label groups for synonym handling in semantic alignment.

Shared between training (for class balancing) and evaluation (for group-aware metrics).
"""

from typing import Dict, List


# =============================================================================
# Label Groups - Synonyms that should be treated as equivalent
# =============================================================================

# LABEL_GROUPS: Fine-grained grouping (~25 effective groups)
# Groups together only true synonyms (same activity, different naming)
# Keeps semantically different activities separate (e.g., ascending vs descending stairs)
LABEL_GROUPS = {
    # Walking variants (same gait pattern)
    # Note: nordic_walking, walking_straight, walking_winding are VTT-ConIoT (zero-shot)
    # but walking exists in training, so these can be evaluated as synonyms
    # KU-HAR: walking_backwards (backwards locomotion, still walking pattern)
    # USC-HAD: walking_forward, walking_left, walking_right (directional walking)
    # HARTH: shuffling (slow, impaired gait pattern — still walking variant)
    'walking': ['walking', 'nordic_walking', 'walking_parking', 'walking_treadmill_flat',
                'walking_treadmill_incline', 'walking_straight', 'walking_winding',
                'walking_backwards', 'walking_forward', 'walking_left', 'walking_right',
                'shuffling'],

    # Stairs - ascending (upward motion pattern)
    'ascending_stairs': ['ascending_stairs', 'climbing_stairs', 'going_up_stairs',
                         'walking_upstairs', 'stairs_up'],

    # Stairs - descending (downward motion pattern)
    'descending_stairs': ['descending_stairs', 'going_down_stairs', 'walking_downstairs',
                          'stairs_down'],

    # Generic stairs (ambiguous direction - WISDM doesn't specify)
    'stairs': ['stairs'],

    # Running/jogging (fast locomotion)
    # USC-HAD: running_forward (directional running)
    'running': ['running', 'jogging', 'running_treadmill', 'running_forward'],

    # Lying/laying (horizontal posture)
    # USC-HAD: sleeping (extended horizontal posture)
    'lying': ['lying', 'laying', 'lying_back', 'lying_right_side', 'laying_back', 'sleeping'],

    # Sitting (seated posture)
    # HARTH: transport_sit (seated in vehicle — same seated posture)
    'sitting': ['sitting', 'sitting_down', 'sitting_chair', 'transport_sit'],

    # Standing (upright posture)
    # HARTH: transport_stand (standing in vehicle — same upright posture)
    'standing': ['standing', 'standing_elevator', 'standing_work', 'transport_stand'],

    # Postural transitions - distinct motion patterns from static postures
    # UniMiB SHAR transitions (whole-body repositioning)
    'transition_to_standing': ['standing_up_from_laying', 'standing_up_from_sitting',
                               'sit_to_stand', 'lie_to_stand', 'lie_to_sit'],
    'transition_to_sitting': ['sitting_down_from_standing', 'stand_to_sit'],
    'transition_to_lying': ['lying_down_from_standing', 'sit_to_lie', 'stand_to_lie'],

    # Talking while stationary (KU-HAR) - distinct from pure posture
    'talking_sitting': ['talking_sitting'],
    'talking_standing': ['talking_standing'],

    # Falling variants (loss of balance - all share sudden acceleration patterns)
    # MobiFall (training) provides fall types
    'falling': ['falling_backward', 'falling_backward_sitting', 'falling_forward',
                'falling_hitting_obstacle', 'falling_left', 'falling_right',
                'falling_with_protection', 'syncope',
                # MobiFall fall types
                'fall_forward', 'fall_backward_knees', 'fall_backward_sitting', 'fall_sideways'],

    # Jumping variants (vertical explosive motion)
    # Note: rope_jumping/rope_skipping are RecGym (zero-shot) but jumping is in training
    # REALDISP: jump_up, jump_front_back, jump_sideways, jump_legs_arms, jump_rope
    # USC-HAD: jumping_up (vertical jump)
    'jumping': ['jumping', 'jump_front_back', 'rope_jumping', 'rope_skipping', 'jumping_down',
                'jump_up', 'jump_sideways', 'jump_legs_arms', 'jump_rope', 'jumping_up'],

    # Eating/drinking (hand-to-mouth repetitive motion)
    'eating': ['eating_chips', 'eating_pasta', 'eating_sandwich', 'eating_soup', 'drinking'],

    # Cycling variants (pedaling motion)
    # DSADS has cycling_horizontal, cycling_vertical
    # HARTH: cycling_sit, cycling_stand (seated vs standing cycling)
    'cycling': ['cycling', 'bicycling', 'cycling_horizontal', 'cycling_vertical',
                'cycling_sit', 'cycling_stand'],

    # Cardio exercise machines (rhythmic full-body motion)
    'cardio_machine': ['exercising_stepper', 'exercising_cross_trainer', 'rowing', 'stairclimber',
                       'elliptical_bike'],

    # Vehicle entry/exit (stepping motion with support)
    'vehicle_entry': ['car_step_in', 'car_step_out'],

    # Elevator (stationary or subtle motion)
    # USC-HAD: elevator_up, elevator_down (directional elevator)
    'elevator': ['moving_elevator', 'elevator_up', 'elevator_down'],

    # Ball sports (throwing, catching, dribbling patterns). NOTE: keyed 'ball_sports'
    # (not 'sports') so CAPTURE-24's free-living 'sports' label does not collide with
    # this group name and get silently merged with ball-handling motions.
    'ball_sports': ['playing_basketball', 'playing_catch', 'dribbling', 'kicking'],

    # Hand/arm activities - fine motor movements (WISDM, MHEALTH activities)
    'typing': ['typing', 'writing'],  # Fine motor, similar wrist patterns
    'grooming': ['brushing_teeth'],
    'clapping': ['clapping'],

    # Household chores (PAMAP2, WISDM activities)
    'ironing': ['ironing'],
    'vacuum_cleaning': ['vacuum_cleaning'],
    'folding_clothes': ['folding_clothes'],

    # Exercise/stretching movements (MHEALTH activities)
    'arm_exercise': ['frontal_elevation_arms', 'raising_hands'],
    'leg_exercise': ['knees_bending'],

    # KU-HAR exercises (floor/body-weight exercises)
    'push_up': ['push_up'],
    'sit_up': ['sit_up'],
    'picking_up': ['picking_up'],  # Bending to pick something up

    # General/free-living sports (KU-HAR table-tennis -> table_tennis;
    # CAPTURE-24 WillettsSpecific 'sports' is the same broad concept).
    'table_tennis': ['table_tennis', 'sports'],

    # Construction/manual work (VTT-ConIoT activities)
    'carrying': ['carrying', 'lifting', 'pushing_cart'],
    'climbing': ['climbing_ladder'],
    'kneeling': ['kneeling_work'],
    'painting': ['roll_painting', 'spraying_paint', 'leveling_paint'],

    # Gym weight exercises (RecGym activities)
    'gym_weights': ['squat', 'bench_press', 'leg_press', 'leg_curl', 'arm_curl', 'adductor_machine'],

    # REALDISP fitness exercises (A1-A33)
    'trunk_twist': ['trunk_twist_arms_out', 'trunk_twist_elbows_bent', 'upper_lower_twist'],
    'waist_exercise': ['waist_bends_forward', 'waist_rotation', 'waist_bend_cross',
                       'lateral_bend', 'lateral_bend_arm_up', 'forward_stretching'],
    'arm_exercise_realdisp': ['lateral_arm_elevation', 'frontal_arm_elevation', 'frontal_hand_claps',
                               'frontal_crossing_arms', 'shoulders_high_rotation', 'shoulders_low_rotation',
                               'arms_inner_rotation'],
    'leg_exercise_realdisp': ['knees_to_breast', 'heels_to_backside', 'knees_bending_crouching',
                               'knees_alternating_forward', 'rotation_on_knees', 'reach_heels_backwards'],
    # Daphnet FoG (Parkinson's gait disorder)
    'freezing_gait': ['freezing_gait'],
}

# LABEL_GROUPS_SIMPLE: Simplified grouping (~10 effective groups)
# For coarser evaluation - groups semantically similar activities
# Use when fine-grained distinctions aren't important or data is limited
LABEL_GROUPS_SIMPLE = {
    # Locomotion - walking pace (includes backwards walking, directional walking, shuffling)
    'walking': ['walking', 'nordic_walking', 'walking_parking', 'walking_treadmill_flat',
                'walking_treadmill_incline', 'walking_straight', 'walking_winding',
                'walking_backwards', 'walking_forward', 'walking_left', 'walking_right',
                'shuffling'],

    # Locomotion - running pace
    'running': ['running', 'jogging', 'running_treadmill', 'running_forward'],

    # Locomotion - stairs (any direction)
    'stairs': ['stairs', 'ascending_stairs', 'climbing_stairs', 'going_up_stairs', 'walking_upstairs',
               'descending_stairs', 'going_down_stairs', 'walking_downstairs',
               'stairs_up', 'stairs_down'],

    # Locomotion - cycling
    'cycling': ['cycling', 'bicycling', 'cycling_horizontal', 'cycling_vertical',
                'cycling_sit', 'cycling_stand'],

    # Stationary - sitting (seated postures only, no transitions)
    'sitting': ['sitting', 'sitting_down', 'sitting_chair', 'transport_sit'],

    # Stationary - standing (upright postures only, no transitions)
    'standing': ['standing', 'standing_elevator', 'moving_elevator', 'standing_work',
                 'elevator_up', 'elevator_down', 'transport_stand'],

    # Stationary - lying (horizontal postures only, no transitions)
    'lying': ['lying', 'laying', 'lying_back', 'lying_right_side', 'laying_back', 'sleeping'],

    # Postural transitions (whole-body repositioning — distinct motion patterns)
    'postural_transition': ['standing_up_from_laying', 'standing_up_from_sitting',
                            'sitting_down_from_standing', 'lying_down_from_standing',
                            'stand_to_sit', 'sit_to_stand', 'sit_to_lie',
                            'lie_to_sit', 'stand_to_lie', 'lie_to_stand'],

    # Talking while stationary (KU-HAR)
    'talking': ['talking_sitting', 'talking_standing'],

    # Dynamic exercise (jumping, body-weight exercises, cardio machines)
    'exercise': ['jumping', 'jumping_up', 'jump_front_back', 'rope_jumping', 'rope_skipping', 'jumping_down',
                 'knees_bending', 'waist_bends_forward', 'frontal_elevation_arms', 'raising_hands',
                 'exercising_stepper', 'exercising_cross_trainer', 'rowing', 'stairclimber', 'climbing_ladder',
                 'squat', 'bench_press', 'leg_press', 'leg_curl', 'arm_curl', 'adductor_machine',
                 'push_up', 'sit_up', 'picking_up',
                 # REALDISP fitness exercises (A1-A33)
                 'jump_up', 'jump_sideways', 'jump_legs_arms', 'jump_rope',
                 'trunk_twist_arms_out', 'trunk_twist_elbows_bent', 'upper_lower_twist',
                 'waist_rotation', 'waist_bend_cross', 'lateral_bend', 'lateral_bend_arm_up',
                 'forward_stretching', 'reach_heels_backwards',
                 'lateral_arm_elevation', 'frontal_arm_elevation', 'frontal_hand_claps',
                 'frontal_crossing_arms', 'shoulders_high_rotation', 'shoulders_low_rotation',
                 'arms_inner_rotation', 'knees_to_breast', 'heels_to_backside',
                 'knees_bending_crouching', 'knees_alternating_forward', 'rotation_on_knees',
                 'elliptical_bike'],

    # Hand/arm activities (fine motor, repetitive arm movements)
    'hand_activity': ['eating_chips', 'eating_pasta', 'eating_sandwich', 'eating_soup',
                      'drinking', 'brushing_teeth', 'typing', 'writing', 'folding_clothes', 'clapping'],

    # Manual work / construction (VTT-ConIoT activities)
    'manual_work': ['carrying', 'lifting', 'pushing_cart', 'kneeling_work',
                    'roll_painting', 'spraying_paint', 'leveling_paint'],

    # Vehicle interaction
    'vehicle': ['car_step_in', 'car_step_out'],

    # Household chores (varied whole-body movements)
    'household': ['ironing', 'vacuum_cleaning'],

    # Ball / general sports (throwing, catching, kicking patterns).
    # Keyed 'ball_sports' to avoid colliding with a literal 'sports' label.
    'ball_sports': ['playing_catch', 'dribbling', 'kicking', 'playing_basketball', 'table_tennis', 'sports'],

    # Falling (loss of balance - grouped for safety applications)
    'falling': ['falling_backward', 'falling_backward_sitting', 'falling_forward',
                'falling_hitting_obstacle', 'falling_left', 'falling_right',
                'falling_with_protection', 'syncope',
                'fall_forward', 'fall_backward_knees', 'fall_backward_sitting', 'fall_sideways'],

    # Gait disorders (Daphnet FoG - Parkinson's patients)
    'gait_disorder': ['freezing_gait'],
}

# Active label groups (change this to switch between groupings)
ACTIVE_LABEL_GROUPS = LABEL_GROUPS  # Use LABEL_GROUPS_SIMPLE for coarser evaluation


def get_label_to_group_mapping(use_simple: bool = False) -> Dict[str, str]:
    """
    Create reverse mapping from individual labels to their group name.
    Labels not in any group map to themselves.

    Args:
        use_simple: If True, use LABEL_GROUPS_SIMPLE instead of LABEL_GROUPS
    """
    label_to_group = {}
    groups = LABEL_GROUPS_SIMPLE if use_simple else LABEL_GROUPS

    # Map grouped labels to their group name
    for group_name, labels in groups.items():
        for label in labels:
            # Note: some labels like 'stairs' may belong to multiple groups
            # We take the first assignment (ascending_stairs comes before descending_stairs)
            if label not in label_to_group:
                label_to_group[label] = group_name

    return label_to_group


def get_group_for_label(label: str, use_simple: bool = False) -> str:
    """
    Get the group name for a given label.
    Returns the label itself if not in any group (singleton group).

    Args:
        label: The activity label
        use_simple: If True, use LABEL_GROUPS_SIMPLE

    Returns:
        Group name or the label itself
    """
    mapping = get_label_to_group_mapping(use_simple)
    return mapping.get(label, label)


def get_group_members(label: str, use_simple: bool = False) -> List[str]:
    """
    Get all labels that are synonyms of the given label.
    Returns list including the label itself.

    Args:
        label: The label to find synonyms for
        use_simple: If True, use LABEL_GROUPS_SIMPLE instead of LABEL_GROUPS
    """
    mapping = get_label_to_group_mapping(use_simple)
    group = mapping.get(label, label)
    groups = LABEL_GROUPS_SIMPLE if use_simple else LABEL_GROUPS

    if group in groups:
        return groups[group]
    else:
        return [label]  # Ungrouped label - only matches itself
