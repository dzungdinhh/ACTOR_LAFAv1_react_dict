"""
Feature Group Indices for CHEEARS Data
Includes both static and longitudinal feature groups for group-based acquisition
"""

# ============================================================================
# STATIC FEATURE GROUPS (shape: N, 22)
# ============================================================================

STATIC_FEATURE_GROUP_INDICES = {
    'race': [16, 17, 18, 19, 20, 21],  # race___0 through race___5
}

STATIC_INDIVIDUAL_FEATURE_INDICES = list(range(16)) + [21]  # Indices not in groups

STATIC_INDIVIDUAL_FEATURE_NAMES = [
    'sex',                      # 0
    'age',                      # 1
    'handedness',               # 2
    'hispanic',                 # 3
    'marital_status',           # 4
    'education',                # 5
    'degree',                   # 6
    'current_employed',         # 7
    'school_year',              # 8
    'militaryaffil',            # 9
    'family_income',            # 10
    'religious_affiliation',    # 11
    'physical_handicap',        # 12
    'cigarette_use',            # 13
    'alcohol_use',              # 14
    'drug_use',                 # 15
    # race is indices 16-21 (group)
]

# ============================================================================
# LONGITUDINAL FEATURE GROUPS (shape: N, T, 149)
# ============================================================================

LONGITUDINAL_FEATURE_GROUP_INDICES = {
    'daily_activities':      [22, 23, 24, 25, 26, 27, 28, 29, 30],  # 9 features
    'daily_experiences':     [31, 32, 33, 34, 35, 36],               # 6 features
    'drink_expectancies':    list(range(37, 66)),                    # 29 features (37-65)
    'drink_motives':         list(range(66, 79)),                    # 13 features (66-78)
    'general_experiences':   [79, 80, 81, 82],                       # 4 features
    'nondrink_expectancies': list(range(83, 109)),                   # 26 features (83-108)
    'nondrink_motives':      list(range(109, 122)),                  # 13 features (109-121)
    'nondrink_plans':        list(range(122, 135)),                  # 13 features (122-134)
    'social_experiences':    [135, 136, 137, 138, 139, 140, 141],    # 7 features
    'day_of_week':           [142, 143, 144, 145, 146, 147, 148],    # 7 features (day_0 to day_6)
}

LONGITUDINAL_INDIVIDUAL_FEATURE_INDICES = list(range(22))  # Indices 0-21

LONGITUDINAL_INDIVIDUAL_FEATURE_NAMES = [
    'happy',                # 0
    'nervous',              # 1
    'angry',                # 2
    'sad',                  # 3
    'excited',              # 4
    'alert',                # 5
    'ashamed',              # 6
    'relaxed',              # 7
    'bored',                # 8
    'content',              # 9
    'stress',               # 10
    'drink_plans',          # 11
    'substance',            # 12
    'dom',                  # 13
    'warm',                 # 14
    'drink_likely',         # 15
    'drink_quantity',       # 16
    'drink_urge',           # 17
    'nondrink_likely',      # 18
    'nondrink_quantity',    # 19
    'nondrink_urge',        # 20
    'nondrink_plan_other',  # 21
]

# ============================================================================
# SUMMARY INFO
# ============================================================================

NUM_STATIC_FEATURES = 22
NUM_STATIC_INDIVIDUAL = 16
NUM_STATIC_GROUPED = 6  # race (6 features)
NUM_STATIC_GROUPS = 1   # 1 group (race)

NUM_LONGITUDINAL_FEATURES = 149
NUM_LONGITUDINAL_INDIVIDUAL = 22
NUM_LONGITUDINAL_GROUPED = 127  # All grouped features
NUM_LONGITUDINAL_GROUPS = 10    # 10 groups

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_static_group_mask(group_name, total_features=22):
    """
    Create binary mask for a static feature group.
    
    Args:
        group_name: Name of the static feature group ('race')
        total_features: Total number of static features (default 22)
        
    Returns:
        numpy array of shape (total_features,) with 1s at group indices
    """
    import numpy as np
    mask = np.zeros(total_features, dtype=np.int32)
    mask[STATIC_FEATURE_GROUP_INDICES[group_name]] = 1
    return mask


def get_longitudinal_group_mask(group_name, total_features=149):
    """
    Create binary mask for a longitudinal feature group.
    
    Args:
        group_name: Name of the feature group
        total_features: Total number of features (default 149)
        
    Returns:
        numpy array of shape (total_features,) with 1s at group indices
    """
    import numpy as np
    mask = np.zeros(total_features, dtype=np.int32)
    mask[LONGITUDINAL_FEATURE_GROUP_INDICES[group_name]] = 1
    return mask


def acquire_static_group(x_static, group_name):
    """
    Extract features for a specific static group.
    
    Args:
        x_static: Static feature array of shape (N, 22) or (22,)
        group_name: Name of the static feature group
        
    Returns:
        Array with only the group's features
    """
    import numpy as np
    indices = STATIC_FEATURE_GROUP_INDICES[group_name]
    return x_static[..., indices]


def acquire_longitudinal_group(x, group_name):
    """
    Extract features for a specific longitudinal group.
    
    Args:
        x: Feature array of shape (N, T, 149) or (N, 149) or (149,)
        group_name: Name of the feature group
        
    Returns:
        Array with only the group's features
    """
    import numpy as np
    indices = LONGITUDINAL_FEATURE_GROUP_INDICES[group_name]
    return x[..., indices]


def print_summary():
    """Print summary of all feature groups."""
    print("="*70)
    print("STATIC FEATURE GROUPS")
    print("="*70)
    print(f"{'Group Name':<30} {'Indices':<20} {'Size'}")
    print("-"*70)
    
    for group_name, indices in STATIC_FEATURE_GROUP_INDICES.items():
        indices_str = f"{indices[0]}-{indices[-1]}"
        print(f"{group_name:<30} {indices_str:<20} {len(indices)}")
    
    print("-"*70)
    print(f"Individual features: {NUM_STATIC_INDIVIDUAL} (indices 0-15)")
    print(f"Grouped features: {NUM_STATIC_GROUPED}")
    print(f"Total static: {NUM_STATIC_FEATURES}")
    print()
    
    print("="*70)
    print("LONGITUDINAL FEATURE GROUPS")
    print("="*70)
    print(f"{'Group Name':<30} {'Indices':<20} {'Size'}")
    print("-"*70)
    
    for group_name, indices in LONGITUDINAL_FEATURE_GROUP_INDICES.items():
        indices_str = f"{indices[0]}-{indices[-1]}"
        print(f"{group_name:<30} {indices_str:<20} {len(indices)}")
    
    print("-"*70)
    print(f"Individual features: {NUM_LONGITUDINAL_INDIVIDUAL} (indices 0-21)")
    print(f"Grouped features: {NUM_LONGITUDINAL_GROUPED}")
    print(f"Total longitudinal: {NUM_LONGITUDINAL_FEATURES}")
    print(f"Number of groups: {NUM_LONGITUDINAL_GROUPS}")


def print_usage():
    """Print usage examples."""
    print()
    print("="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print()
    
    print("1. Import in your model:")
    print("   from feature_groups import (STATIC_FEATURE_GROUP_INDICES,")
    print("                                 LONGITUDINAL_FEATURE_GROUP_INDICES)")
    print()
    
    print("2. Acquire static race features together:")
    print("   race_indices = STATIC_FEATURE_GROUP_INDICES['race']")
    print("   # race_indices = [16, 17, 18, 19, 20, 21]")
    print("   static_mask[race_indices] = gate_value  # Acquire all 6 race features together")
    print()
    
    print("3. Acquire longitudinal day_of_week together:")
    print("   day_indices = LONGITUDINAL_FEATURE_GROUP_INDICES['day_of_week']")
    print("   # day_indices = [142, 143, 144, 145, 146, 147, 148]")
    print("   long_mask[:, :, day_indices] = gate_value  # Acquire all 7 day features together")
    print()
    
    print("4. Group-based acquisition (reduce 149 → 32 decisions):")
    print("   # 22 individual + 10 groups = 32 total acquisition decisions")
    print("   for group_name, indices in LONGITUDINAL_FEATURE_GROUP_INDICES.items():")
    print("       group_gate = planner_output[group_name]")
    print("       mask[:, :, indices] = group_gate")
    print()
    
    print("5. Extract group features:")
    print("   x = data['x']  # (5905, 10, 149)")
    print("   drink_exp = acquire_longitudinal_group(x, 'drink_expectancies')")
    print("   # drink_exp.shape = (5905, 10, 29)")


# Example usage
if __name__ == '__main__':
    print_summary()
    print_usage()
