"""
Dataset-specific label augmentation for IMU activity recognition.

Each dataset gets custom synonyms and templates tailored to its activities.
This provides rich, natural language variations for contrastive learning.
"""

import random

# ============================================================================
# UCI-HAR: Basic 6 activities (lab-controlled)
# ============================================================================

UCI_HAR_SYNONYMS = {
    "walking": ["walking", "strolling", "striding", "ambulating", "pacing"],
    "walking_upstairs": ["walking upstairs", "climbing stairs", "ascending stairs", "going upstairs", "stair climbing"],
    "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs", "stair descending"],
    "sitting": ["sitting", "seated", "sitting down", "in a seated position"],
    "standing": ["standing", "standing up", "upright", "in a standing position", "on feet"],
    "laying": ["laying", "lying down", "reclining", "horizontal", "lying", "supine"],
}

UCI_HAR_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "individual {}",
    "subject {}",
    "user {}",
    "{} activity",
    "{} posture",
    "body {}",
]

# ============================================================================
# MHEALTH: Exercise and daily activities (medical monitoring)
# ============================================================================

MHEALTH_SYNONYMS = {
    "walking": ["walking", "strolling", "ambulating", "taking a walk"],
    "jogging": ["jogging", "light running", "slow running", "jog"],
    "running": ["running", "sprinting", "fast running", "run"],
    "cycling": ["cycling", "riding a bike", "pedaling", "biking"],
    "climbing_stairs": ["climbing stairs", "stair climbing", "ascending stairs", "going up stairs"],
    "sitting": ["sitting", "seated", "sitting down"],
    "standing": ["standing", "upright", "standing still"],
    "lying": ["lying", "laying down", "horizontal", "supine", "reclining"],
    "frontal_elevation_arms": ["frontal arm elevation", "raising arms forward", "arm lifting", "frontal arm raise"],
    "knees_bending": ["knee bending", "squatting", "knee flexion", "bending knees"],
    "waist_bends_forward": ["forward waist bend", "bending forward", "torso flexion", "waist bending"],
    "jump_front_back": ["jumping front and back", "forward-backward jumping", "jump exercise"],
}

MHEALTH_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "patient {}",
    "subject performing {}",
    "{} motion",
    "{} exercise",
    "{} movement",
    "health monitoring during {}",
]

# ============================================================================
# PAMAP2: Daily and sports activities (physical activity monitoring)
# ============================================================================

PAMAP2_SYNONYMS = {
    "walking": ["walking", "strolling", "ambulating", "casual walking"],
    "nordic_walking": ["nordic walking", "pole walking", "nordic walk", "walking with poles"],
    "running": ["running", "jogging", "fast running"],
    "cycling": ["cycling", "biking", "riding bicycle", "pedaling"],
    "ascending_stairs": ["ascending stairs", "climbing stairs", "going upstairs", "stair ascent"],
    "descending_stairs": ["descending stairs", "going downstairs", "stair descent", "walking downstairs"],
    "rope_jumping": ["rope jumping", "jump rope", "skipping rope", "jumping rope"],
    "sitting": ["sitting", "seated", "sitting down"],
    "standing": ["standing", "upright", "standing still"],
    "lying": ["lying", "laying down", "horizontal", "reclining"],
    "ironing": ["ironing", "pressing clothes", "ironing clothes"],
    "vacuum_cleaning": ["vacuum cleaning", "vacuuming", "cleaning with vacuum", "using vacuum cleaner"],
}

PAMAP2_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "individual {}",
    "user performing {}",
    "{} activity",
    "physical activity: {}",
    "daily activity: {}",
    "{} behavior",
]

# ============================================================================
# WISDM: Diverse daily activities with hand gestures
# ============================================================================

WISDM_SYNONYMS = {
    "walking": ["walking", "strolling", "ambulating"],
    "jogging": ["jogging", "light jogging", "slow running"],
    "stairs": ["using stairs", "stair activity", "stair movement", "on stairs"],
    "sitting": ["sitting", "seated", "sitting down"],
    "standing": ["standing", "upright", "standing still"],

    # Eating activities
    "eating_pasta": ["eating pasta", "consuming pasta", "having pasta"],
    "eating_chips": ["eating chips", "snacking on chips", "consuming chips", "eating crisps"],
    "eating_sandwich": ["eating sandwich", "having a sandwich", "consuming sandwich"],
    "eating_soup": ["eating soup", "having soup", "consuming soup", "spooning soup"],
    "drinking": ["drinking", "having a drink", "consuming beverage"],

    # Hand activities
    "brushing_teeth": ["brushing teeth", "tooth brushing", "dental hygiene", "oral care"],
    "typing": ["typing", "keyboard typing", "texting", "using keyboard"],
    "writing": ["writing", "handwriting", "writing by hand", "penmanship"],
    "clapping": ["clapping", "hand clapping", "applauding", "clapping hands"],
    "folding_clothes": ["folding clothes", "clothing folding", "folding laundry", "organizing clothes"],

    # Sports activities
    "playing_catch": ["playing catch", "throwing and catching", "tossing ball", "catch game"],
    "dribbling": ["dribbling", "ball dribbling", "dribbling basketball"],
    "kicking": ["kicking", "kicking ball", "foot kicking"],
}

WISDM_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "user {}",
    "individual {}",
    "{} activity",
    "{} gesture",
    "{} action",
    "human {}",
    "smartphone user {}",
]

# ============================================================================
# MotionSense: Phone pocket activities (similar to UCI-HAR)
# ============================================================================

MOTIONSENSE_SYNONYMS = {
    "walking": ["walking", "strolling", "ambulating", "taking steps"],
    "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs", "stair descent", "walking down stairs"],
    "walking_upstairs": ["walking upstairs", "climbing stairs", "going upstairs", "ascending stairs", "walking up stairs"],
    "jogging": ["jogging", "light running", "slow running", "jog", "light jog"],
    "sitting": ["sitting", "seated", "sitting down", "in a seated position"],
    "standing": ["standing", "upright", "standing still", "on feet", "standing up"],
}

MOTIONSENSE_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "individual {}",
    "smartphone user {}",
    "{} activity",
    "mobile phone sensing {}",
    "phone in pocket {}",
]

# ============================================================================
# UniMiB SHAR: ADL and postural transitions
# ============================================================================

UNIMIB_SHAR_SYNONYMS = {
    "standing_up_from_sitting": ["standing up from sitting", "rising from chair", "getting up from seated", "sit-to-stand"],
    "standing_up_from_laying": ["standing up from laying", "getting out of bed", "rising from horizontal", "lay-to-stand"],
    "walking": ["walking", "strolling", "ambulating", "taking steps"],
    "running": ["running", "jogging", "sprinting", "fast movement"],
    "going_up_stairs": ["going up stairs", "climbing stairs", "ascending stairs", "stair ascent"],
    "jumping": ["jumping", "hopping", "leaping", "vertical jump"],
    "going_down_stairs": ["going down stairs", "descending stairs", "walking downstairs", "stair descent"],
    "lying_down_from_standing": ["lying down from standing", "going to bed", "laying down", "stand-to-lay"],
    "sitting_down_from_standing": ["sitting down from standing", "taking a seat", "sitting down", "stand-to-sit"],
    # Falls
    "falling_forward": ["falling forward", "forward fall", "tripping forward"],
    "falling_backward": ["falling backward", "backward fall"],
    "falling_left": ["falling left", "sideways fall left"],
    "falling_right": ["falling right", "sideways fall right"],
    "falling_hitting_obstacle": ["falling hitting obstacle", "fall with collision"],
    "falling_with_protection": ["falling with protection", "protected fall", "bracing fall"],
    "falling_backward_sitting": ["falling backward into sitting", "backward fall sitting"],
    "syncope": ["syncope", "fainting", "loss of consciousness", "blacking out"],
    "sitting_down": ["sitting down", "taking a seat", "sit down"],
}

UNIMIB_SHAR_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "{} transition",
    "postural transition: {}",
    "activity: {}",
    "daily movement: {}",
]

# ============================================================================
# HHAR: Heterogeneity Activity Recognition (multi-device)
# ============================================================================

HHAR_SYNONYMS = {
    "standing": ["standing", "upright", "standing still", "on feet"],
    "sitting": ["sitting", "seated", "sitting down", "in a seated position"],
    "walking": ["walking", "strolling", "ambulating", "taking steps"],
    "cycling": ["cycling", "biking", "riding bicycle", "pedaling"],
    "walking_upstairs": ["walking upstairs", "climbing stairs", "ascending stairs", "going upstairs"],
    "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs", "stair descent"],
}

HHAR_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "individual {}",
    "smartphone user {}",
    "{} activity",
    "heterogeneous device {}",
    "mobile sensing {}",
]

# ============================================================================
# Main augmentation function
# ============================================================================

# Map dataset names to their augmentation configs
DATASET_CONFIGS = {
    "capture24": {
        "synonyms": {
            "sleeping": ["sleeping", "asleep", "in bed", "resting in bed"],
            "sitting": ["sitting", "seated", "sitting down", "in a chair"],
            "standing": ["standing", "upright", "standing still", "on feet"],
            "walking": ["walking", "strolling", "ambulating", "going for a walk"],
            "bicycling": ["bicycling", "cycling", "riding a bike", "biking"],
            "vehicle": ["in a vehicle", "riding in a car", "travelling by vehicle", "in transport"],
            "household_chores": ["household chores", "doing chores", "housework", "domestic tasks"],
            "manual_work": ["manual work", "manual labour", "physical work", "labouring"],
            "sports": ["playing sports", "sports", "athletic activity", "exercising"],
            "mixed_activity": ["mixed activity", "miscellaneous activity", "varied activity", "general activity"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "person is {}"],
    },
    "uci_har": {
        "synonyms": UCI_HAR_SYNONYMS,
        "templates": UCI_HAR_TEMPLATES,
    },
    "mhealth": {
        "synonyms": MHEALTH_SYNONYMS,
        "templates": MHEALTH_TEMPLATES,
    },
    "pamap2": {
        "synonyms": PAMAP2_SYNONYMS,
        "templates": PAMAP2_TEMPLATES,
    },
    "wisdm": {
        "synonyms": WISDM_SYNONYMS,
        "templates": WISDM_TEMPLATES,
    },
    "unimib_shar": {
        "synonyms": UNIMIB_SHAR_SYNONYMS,
        "templates": UNIMIB_SHAR_TEMPLATES,
    },
    "hhar": {
        "synonyms": HHAR_SYNONYMS,
        "templates": HHAR_TEMPLATES,
    },
    "motionsense": {
        "synonyms": MOTIONSENSE_SYNONYMS,
        "templates": MOTIONSENSE_TEMPLATES,
    },
    # New datasets - minimal configs for label retrieval
    # Training datasets
    "dsads": {
        "synonyms": {
            "sitting": ["sitting", "seated", "in a chair", "sitting down"],
            "standing": ["standing", "upright", "on feet", "standing still"],
            "lying_back": ["lying on back", "supine", "reclining on back", "laying face up"],
            "lying_right_side": ["lying on the right side", "right side-lying", "lying on right side", "right lateral recumbent"],
            "stairs_up": ["ascending stairs", "climbing stairs", "going upstairs", "stair climbing"],
            "stairs_down": ["descending stairs", "going downstairs", "walking downstairs", "stair descent"],
            "walking_parking": ["walking in parking lot", "walking outdoors", "outdoor walking", "walking outside"],
            "walking_treadmill_flat": ["walking on treadmill", "treadmill walking", "flat treadmill walk"],
            "walking_treadmill_incline": ["walking on inclined treadmill", "incline treadmill walking", "uphill treadmill"],
            "running_treadmill": ["running on treadmill", "treadmill running", "treadmill jog"],
            "exercising_stepper": ["using stepper machine", "stepper exercise", "step machine"],
            "exercising_cross_trainer": ["using elliptical", "elliptical trainer", "cross trainer exercise"],
            "cycling_horizontal": ["cycling recumbent", "recumbent bike", "horizontal cycling"],
            "cycling_vertical": ["cycling upright", "upright bike", "vertical cycling"],
            "rowing": ["rowing", "rowing machine", "ergometer rowing"],
            "jumping": ["jumping", "hopping", "leaping"],
            "playing_basketball": ["playing basketball", "basketball", "shooting hoops"],
            "moving_elevator": ["riding elevator", "in elevator", "elevator ride"],
            "standing_elevator": ["standing in elevator", "idle in elevator", "waiting in elevator"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "{} movement", "person is {}"],
    },
    "mobiact": {
        "synonyms": {
            "standing": ["standing", "upright", "standing still", "on feet"],
            "walking": ["walking", "strolling", "ambulating", "taking steps"],
            "jogging": ["jogging", "light running", "slow running", "jog"],
            "jumping": ["jumping", "hopping", "leaping"],
            "stairs_up": ["ascending stairs", "climbing stairs", "going upstairs", "stair climbing"],
            "stairs_down": ["descending stairs", "going downstairs", "walking downstairs", "stair descent"],
            "sitting_chair": ["sitting on chair", "seated", "sitting down", "in a chair"],
            "car_step_in": ["getting into car", "entering vehicle", "stepping into car"],
            "car_step_out": ["getting out of car", "exiting vehicle", "stepping out of car"],
            "fall_forward": ["falling forward", "forward fall", "tripping forward"],
            "fall_backward_knees": ["falling backward onto knees", "backward knee fall"],
            "fall_backward_sitting": ["falling backward into sitting", "backward sitting fall"],
            "fall_sideways": ["falling sideways", "lateral fall", "sideways fall"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "{} movement", "person is {}"],
    },
    "hapt": {
        "synonyms": {
            # Basic activities (same as UCI HAR)
            "walking": ["walking", "strolling", "ambulating", "taking steps"],
            "walking_upstairs": ["walking upstairs", "ascending stairs", "climbing stairs", "going upstairs"],
            "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs", "stair descent"],
            "sitting": ["sitting", "seated", "sitting down", "in a chair"],
            "standing": ["standing", "upright", "standing still", "on feet"],
            "lying": ["lying", "lying down", "horizontal", "reclining", "supine"],
            # Postural transitions
            "stand_to_sit": ["standing to sitting", "sit down transition", "lowering to sit"],
            "sit_to_stand": ["sitting to standing", "stand up transition", "rising from chair"],
            "sit_to_lie": ["sitting to lying", "lie down from sitting", "reclining from seated"],
            "lie_to_sit": ["lying to sitting", "sit up from lying", "rising to seated"],
            "stand_to_lie": ["standing to lying", "lie down from standing", "lowering to horizontal"],
            "lie_to_stand": ["lying to standing", "stand up from lying", "rising from horizontal"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "{} transition", "person is {}"],
    },
    "kuhar": {
        "synonyms": {
            # Basic postures
            "standing": ["standing", "upright", "standing still", "on feet"],
            "sitting": ["sitting", "seated", "sitting down", "in a chair"],
            "lying": ["lying", "lying down", "horizontal", "reclining"],
            # Locomotion
            "walking": ["walking", "strolling", "ambulating", "taking steps"],
            "walking_backwards": ["walking backward", "reverse walking", "backing up"],
            "walking_upstairs": ["walking upstairs", "ascending stairs", "climbing stairs", "going upstairs"],
            "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs"],
            "running": ["running", "sprinting", "jogging", "fast movement"],
            "jumping": ["jumping", "hopping", "leaping", "vertical jump"],
            # Transitions
            "standing_up_from_sitting": ["standing up from sitting", "rising from chair", "sit-to-stand"],
            "standing_up_from_laying": ["standing up from laying", "getting up from lying", "lay-to-stand"],
            # Activities
            "picking_up": ["picking up object", "bending to pick up", "retrieving from floor"],
            "push_up": ["doing push-ups", "push-up exercise", "press-up"],
            "sit_up": ["doing sit-ups", "sit-up exercise", "abdominal crunch"],
            "talking_sitting": ["talking while sitting", "seated conversation", "chatting while seated"],
            "talking_standing": ["talking while standing", "standing conversation", "chatting while standing"],
            "table_tennis": ["table tennis", "playing table tennis", "ping pong", "racquet sport"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "{} movement", "person is {}"],
    },
    # sp_sw_har: Timed-Up-and-Go (sit, stand up, walk, turn, sit down); paired phone + watch IMU
    "sp_sw_har": {
        "synonyms": {
            "sitting": ["sitting", "seated", "in a chair", "in a seated position"],
            "sitting_down": ["sitting down", "taking a seat", "sit down", "lowering to sit"],
            "standing_up_from_sitting": ["standing up from sitting", "rising from chair", "getting up from seated", "sit-to-stand"],
            "turning": ["turning", "turning around", "turning about", "changing direction"],
            "walking": ["walking", "strolling", "ambulating", "taking steps"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "{} movement", "person is {}"],
    },
    # nfi_fared: forensic activity set — locomotion, transport and dynamic movements (back + forearm IMU)
    "nfi_fared": {
        "synonyms": {
            "cycling": ["cycling", "biking", "riding a bike", "pedaling"],
            "dragging": ["dragging", "dragging an object", "pulling along the ground", "hauling"],
            "elevator_down": ["going down in an elevator", "riding an elevator down", "descending in a lift", "elevator going down"],
            "elevator_up": ["going up in an elevator", "riding an elevator up", "ascending in a lift", "elevator going up"],
            "escalator_down": ["going down an escalator", "riding an escalator down", "descending on an escalator", "escalator going down"],
            "escalator_up": ["going up an escalator", "riding an escalator up", "ascending on an escalator", "escalator going up"],
            "kicking": ["kicking", "kicking motion", "striking with the foot", "foot kick"],
            "punching": ["punching", "throwing a punch", "striking with the fist", "hitting with a fist"],
            "running": ["running", "sprinting", "fast running", "run"],
            "sitting": ["sitting", "seated", "in a chair", "in a seated position"],
            "standing": ["standing", "upright", "standing still", "on feet"],
            "throwing": ["throwing", "throwing an object", "tossing", "hurling"],
            "vehicle": ["in a vehicle", "riding in a car", "travelling by vehicle", "in transport"],
            "walking": ["walking", "strolling", "ambulating", "taking steps"],
            "walking_downstairs": ["walking downstairs", "descending stairs", "going downstairs", "stair descent"],
            "walking_upstairs": ["walking upstairs", "ascending stairs", "climbing stairs", "going upstairs"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "{} movement", "person is {}"],
    },
    # harmes: fine-grained kitchen/bathroom hand ADLs (dominant-wrist smartwatch IMU)
    "harmes": {
        "synonyms": {
            "applying_hand_cream": ["applying hand cream", "putting on hand lotion", "rubbing in hand cream", "moisturizing the hands"],
            "brushing_teeth": ["brushing teeth", "tooth brushing", "dental hygiene", "oral care"],
            "cleaning_table": ["cleaning the table", "wiping the table", "wiping down a table", "table cleaning"],
            "cutting_vegetables": ["cutting vegetables", "chopping vegetables", "slicing vegetables", "dicing vegetables"],
            "disinfecting_hands": ["disinfecting the hands", "sanitizing the hands", "applying hand sanitizer", "using hand sanitizer"],
            "drinking": ["drinking", "having a drink", "taking a drink", "sipping a drink"],
            "emptying_dishwasher": ["emptying the dishwasher", "unloading the dishwasher", "taking dishes out of the dishwasher", "clearing the dishwasher"],
            "floor_cleaning": ["cleaning the floor", "mopping the floor", "wiping the floor", "floor cleaning"],
            "making_tea": ["making tea", "preparing tea", "brewing tea", "fixing a cup of tea"],
            "putting_away_dishes": ["putting away dishes", "putting dishes away", "storing dishes", "stacking dishes in the cupboard"],
            "vacuum_cleaning": ["vacuum cleaning", "vacuuming", "cleaning with a vacuum", "using a vacuum cleaner"],
            "washing_dishes": ["washing dishes", "doing the dishes", "washing up", "cleaning dishes"],
            "washing_hands": ["washing hands", "hand washing", "washing one's hands", "rinsing the hands"],
            "watering_plants": ["watering plants", "watering the plants", "watering flowers", "giving the plants water"],
            "window_cleaning": ["cleaning windows", "washing windows", "wiping windows", "window cleaning"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "daily activity: {}", "person is {}"],
    },
    # xrf_v2: indoor daily activities (five-position body IMU + AirPods ear IMU)
    "xrf_v2": {
        "synonyms": {
            "answering_phone": ["answering the phone", "picking up the phone", "answering a phone call", "taking a phone call"],
            "cleaning_table": ["cleaning the table", "wiping the table", "wiping down a table", "table cleaning"],
            "cutting_fruit": ["cutting fruit", "slicing fruit", "chopping fruit", "cutting up fruit"],
            "drinking": ["drinking", "having a drink", "taking a drink", "sipping a drink"],
            "eating_fruit": ["eating fruit", "having fruit", "eating a piece of fruit", "consuming fruit"],
            "lying": ["lying", "lying down", "horizontal", "reclining", "supine"],
            "lying_down_from_standing": ["lying down from standing", "lying down", "reclining from standing", "stand-to-lie"],
            "opening_curtains": ["opening the curtains", "drawing the curtains", "pulling open the curtains", "opening curtains"],
            "opening_envelope": ["opening an envelope", "opening the envelope", "tearing open an envelope", "unsealing an envelope"],
            "opening_windows": ["opening the window", "opening a window", "opening windows", "pushing a window open"],
            "picking_fruit": ["picking fruit", "picking up fruit", "selecting fruit", "gathering fruit"],
            "picking_up": ["picking up an object", "bending to pick something up", "picking something up", "retrieving an object"],
            "pouring_water": ["pouring water", "pouring a glass of water", "filling a glass with water", "pouring out water"],
            "reading": ["reading", "reading a book", "reading a document", "reading text"],
            "sitting_down": ["sitting down", "taking a seat", "sit down", "lowering to sit"],
            "standing": ["standing", "upright", "standing still", "on feet"],
            "standing_up_from_lying": ["standing up from lying", "getting up from lying down", "rising from lying", "lie-to-stand"],
            "standing_up_from_sitting": ["standing up from sitting", "rising from chair", "getting up from seated", "sit-to-stand"],
            "stretching": ["stretching", "doing stretches", "stretching the body", "limbering up"],
            "taking_medicine": ["taking medicine", "taking medication", "taking a pill", "swallowing a pill"],
            "throwing_garbage": ["throwing away garbage", "throwing out the trash", "disposing of garbage", "tossing out the trash"],
            "toggling_lamp": ["toggling a lamp", "switching the lamp on and off", "turning a lamp on or off", "flicking the lamp switch"],
            "typing": ["typing", "typing on a keyboard", "keyboard typing", "using the keyboard"],
            "using_mouse": ["using a mouse", "using a computer mouse", "operating a mouse", "moving the mouse"],
            "using_phone": ["using a phone", "using a smartphone", "operating a phone", "using a mobile phone"],
            "walking": ["walking", "strolling", "ambulating", "taking steps"],
            "washing_hands": ["washing hands", "hand washing", "washing one's hands", "rinsing the hands"],
            "watering_plants": ["watering plants", "watering the plants", "watering flowers", "giving the plants water"],
            "writing": ["writing", "handwriting", "writing by hand", "penmanship"],
            "writing_on_blackboard": ["writing on a blackboard", "writing on the board", "writing on a chalkboard", "chalking on a blackboard"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} activity", "daily activity: {}", "person is {}"],
    },
    # Zero-shot datasets
    "realworld": {
        "synonyms": {
            "walking": ["walking"],
            "running": ["running"],
            "sitting": ["sitting"],
            "standing": ["standing"],
            "lying": ["lying"],
            "stairs_up": ["stairs_up", "climbing stairs"],
            "stairs_down": ["stairs_down", "descending stairs"],
            "jumping": ["jumping"],
        },
        "templates": ["{}"],
    },
    "recgym": {
        "synonyms": {
            "walking": ["walking", "strolling", "ambulating", "treadmill walking"],
            "running": ["running", "sprinting", "jogging", "treadmill running"],
            "cycling": ["cycling", "biking", "pedaling", "stationary bike"],
            "stairclimber": ["stair climber machine", "stair stepper", "climbing machine"],
            "rope_skipping": ["jumping rope", "skipping rope", "rope jumping"],
            "squat": ["squatting", "doing squats", "squat exercise", "knee bend"],
            "bench_press": ["bench pressing", "chest press", "barbell press"],
            "arm_curl": ["bicep curl", "arm curling", "dumbbell curl"],
            "leg_curl": ["leg curling", "hamstring curl", "leg curl machine"],
            "leg_press": ["leg pressing", "leg press machine", "lower body press"],
            "adductor_machine": ["adductor exercise", "inner thigh machine", "hip adduction"],
        },
        "templates": ["{}", "person {}", "subject {}", "individual {}", "{} exercise", "{} movement", "gym {}"],
    },
    # Zero-shot test dataset
    "shoaib": {
        "synonyms": {
            "walking": ["walking", "strolling", "ambulating"],
            "running": ["running", "sprinting"],
            "jogging": ["jogging", "light running"],
            "cycling": ["cycling", "biking", "pedaling"],
            "sitting": ["sitting", "seated"],
            "standing": ["standing", "upright"],
            "walking_upstairs": ["walking upstairs", "climbing stairs", "ascending stairs"],
            "walking_downstairs": ["walking downstairs", "descending stairs"],
        },
        "templates": ["{}"],
    },
    # Training datasets
    "opportunity": {
        "synonyms": {
            "standing": ["standing", "upright"],
            "walking": ["walking", "ambulating"],
            "sitting": ["sitting", "seated"],
            "lying": ["lying", "laying down", "horizontal"],
        },
        "templates": ["{}"],
    },
    "harth": {
        "synonyms": {
            "walking": ["walking", "strolling", "ambulating"],
            "running": ["running", "jogging", "sprinting"],
            "shuffling": ["shuffling", "slow walking", "dragging feet"],
            "stairs_up": ["stairs up", "climbing stairs", "ascending stairs"],
            "stairs_down": ["stairs down", "descending stairs", "going downstairs"],
            "standing": ["standing", "upright", "on feet"],
            "sitting": ["sitting", "seated"],
            "lying": ["lying", "lying down", "horizontal", "reclining"],
            "cycling_sit": ["cycling seated", "sitting cycling", "recumbent cycling"],
            "cycling_stand": ["cycling standing", "standing cycling", "upright cycling"],
            "transport_sit": ["sitting in transport", "seated in vehicle", "riding seated"],
            "transport_stand": ["standing in transport", "standing in vehicle", "riding standing"],
        },
        "templates": ["{}"],
    },
}


def augment_label(
    label: str,
    dataset_name: str,
    augmentation_rate: float = 0.8,
    use_synonyms: bool = True,
    use_templates: bool = True,
) -> str:
    """
    Augment a single activity label with dataset-specific synonyms and templates.

    Args:
        label: Original activity label (e.g., "walking")
        dataset_name: Name of dataset (e.g., "uci_har", "mhealth", "pamap2", "wisdm")
        augmentation_rate: Probability of augmenting (0.0 to 1.0)
        use_synonyms: Whether to apply synonym replacement
        use_templates: Whether to apply template wrapping

    Returns:
        Augmented label text

    Examples:
        >>> augment_label("walking", "uci_har")
        "person strolling"  # synonym + template

        >>> augment_label("eating_pasta", "wisdm")
        "user consuming pasta"  # synonym + template

        >>> augment_label("cycling", "pamap2")
        "physical activity: biking"  # synonym + template
    """
    # No augmentation during validation or with probability (1 - augmentation_rate)
    if random.random() > augmentation_rate:
        return label

    # Get dataset-specific config
    if dataset_name not in DATASET_CONFIGS:
        # Fallback to generic template if dataset unknown (de-underscore so multi-word
        # labels like 'household_chores' never reach the text encoder as raw tokens).
        label = label.replace("_", " ")
        if use_templates and random.random() < 0.5:
            return random.choice(["person {}", "{} activity", "human {}"]).format(label)
        return label

    config = DATASET_CONFIGS[dataset_name]
    synonyms = config["synonyms"]
    templates = config["templates"]

    # Step 1: Apply synonym (if available and enabled)
    if use_synonyms and label in synonyms:
        label = random.choice(synonyms[label])
    else:
        label = label.replace("_", " ")   # no synonym -> at least read it naturally

    # Step 2: Apply template (if enabled)
    if use_templates:
        template = random.choice(templates)
        label = template.format(label)

    return label


if __name__ == "__main__":
    # Test augmentation
    print("=" * 70)
    print("Dataset-Specific Label Augmentation Test")
    print("=" * 70)

    test_cases = [
        ("walking", "uci_har"),
        ("eating_pasta", "wisdm"),
        ("cycling", "pamap2"),
        ("jogging", "mhealth"),
        ("standing_up_from_sitting", "unimib_shar"),
        ("jogging", "motionsense"),
    ]

    for label, dataset in test_cases:
        print(f"\n{dataset.upper()}: '{label}'")
        print("Variations:")
        for i in range(5):
            augmented = augment_label(label, dataset, augmentation_rate=1.0)
            print(f"  {i+1}. {augmented}")
