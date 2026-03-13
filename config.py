"""
Configuration for Gumbel Actor-Critic on Longitudinal Data
"""
import os
from datetime import datetime

#Dataset selection (override via ACTOR_DATASET env var)
DATASET = os.environ.get('ACTOR_DATASET', 'womac')  # 'synthetic', 'cheears_demog', 'cheears_day_context', 'klg', 'womac', 'ILIADD', 'adni'

# Mask type for classifier pretraining (override via ACTOR_MASK_TYPE env var)
# 'uniform' = shared random threshold (P(K=k) = 1/(n+1), default)
# 'bernoulli' = independent coin flip per feature (P(feature)=0.5)
MASK_TYPE = os.environ.get('ACTOR_MASK_TYPE', 'uniform')

# Paths
BASE_DIR = './'  
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
_default_output = os.path.join(BASE_DIR, 'outputs', f'{DATASET}')
_legacy_output = os.path.join(BASE_DIR, f'{DATASET}')
OUTPUT_FOLDER = os.environ.get('ACTOR_OUTPUT_FOLDER', _default_output)
if 'ACTOR_OUTPUT_FOLDER' not in os.environ and (not os.path.exists(OUTPUT_FOLDER)) and os.path.exists(_legacy_output):
    # Backward-compatible fallback for historical runs stored under ./<dataset>/.
    OUTPUT_FOLDER = _legacy_output

# Data paths
SYNTHETIC_DATA_FOLDER = os.path.join(BASE_DIR, 'synthetic_data')
CHEEARS_DATA_FOLDER = os.path.join(BASE_DIR, 'cheears')
CHEEARS_DEMOG_DATA_FOLDER = os.path.join(BASE_DIR, 'cheears_demog')
CHEEARS_DAY_CONTEXT_DATA_FOLDER = os.path.join(BASE_DIR, 'cheears_day_context')
OAI_DATA_FOLDER = os.path.join(BASE_DIR, 'oai_data')
ILIADD_DATA_FOLDER = os.path.join(BASE_DIR, 'ILIADD')
ADNI_DATA_FOLDER = os.path.join(BASE_DIR, 'adni')

# Active data folder (set by DATASET unless ACTOR_DATA_FOLDER override is set)
if DATASET == 'cheears_demog':
    _default_data_folder = CHEEARS_DEMOG_DATA_FOLDER
elif DATASET == 'cheears':
    _default_data_folder = CHEEARS_DATA_FOLDER
elif DATASET == 'cheears_day_context':
    _default_data_folder = CHEEARS_DAY_CONTEXT_DATA_FOLDER
elif DATASET in ('klg', 'womac'):
    _default_data_folder = OAI_DATA_FOLDER
elif DATASET == 'ILIADD':
    _default_data_folder = ILIADD_DATA_FOLDER
elif DATASET == 'adni':
    _default_data_folder = ADNI_DATA_FOLDER
else:
    _default_data_folder = SYNTHETIC_DATA_FOLDER
DATA_FOLDER = os.environ.get('ACTOR_DATA_FOLDER', _default_data_folder)

# Model paths
_cls_suffix = '' if MASK_TYPE == 'uniform' else f'_{MASK_TYPE}'
CLASSIFIER_PATH = os.path.join(OUTPUT_FOLDER, f'classifier{_cls_suffix}.ckpt')


def make_hp_suffix(cost_weight, aux_cost_weight):
    """Build filename suffix from hyperparameters, e.g. '_cw0.01_acw0.001'."""
    parts = [f'cw{cost_weight}']
    if aux_cost_weight is not None:
        parts.append(f'acw{aux_cost_weight}')
    return '_' + '_'.join(parts)


def make_oracle_path(cost_weight):
    """Oracle rollout path for a given cost_weight."""
    return os.path.join(OUTPUT_FOLDER, f'oracle_rollout_cw{cost_weight}.npz')


def make_actor_path(cost_weight, aux_cost_weight, joint=False, baseline='learned'):
    """Actor checkpoint path for given hyperparameters."""
    suffix = make_hp_suffix(cost_weight, aux_cost_weight)
    prefix = 'actor_iterative_joint' if joint else 'actor_iterative'
    if baseline != 'learned':
        prefix = f'{prefix}_baseline_{baseline}'
    return os.path.join(OUTPUT_FOLDER, f'{prefix}{suffix}.ckpt')


def make_eval_path(cost_weight, aux_cost_weight):
    """Evaluation results path for given hyperparameters."""
    suffix = make_hp_suffix(cost_weight, aux_cost_weight)
    return os.path.join(OUTPUT_FOLDER, f'evaluation_results{suffix}.npz')


def make_trajectory_path(cost_weight, aux_cost_weight, joint=False):
    """Trajectory results path for given hyperparameters."""
    suffix = make_hp_suffix(cost_weight, aux_cost_weight)
    prefix = 'trajectory_joint' if joint else 'trajectory'
    return os.path.join(OUTPUT_FOLDER, f'{prefix}{suffix}.npz')


if DATASET == 'cheears_demog':
    NUM_TIME = 10
    NUM_FEAT = 149   # d: features per timestep (including day_of_week)
    NUM_AUX = 22     # number of static/auxiliary features
elif DATASET == 'cheears':
    NUM_TIME = 10
    NUM_FEAT = 149   # d: features per timestep (including day_of_week)
    NUM_AUX = 34
elif DATASET == 'cheears_day_context':
    NUM_TIME = 10
    NUM_FEAT = 142   # v2 day-context longitudinal features (day_of_week moved to static)
    NUM_AUX = 35
elif DATASET in ('klg', 'womac'):
    NUM_TIME = 7
    NUM_FEAT = 17    # d: time-variant features per timestep (27 - 10 static)
    NUM_AUX = 10     # baseline (time-invariant) features
elif DATASET == 'ILIADD':
    NUM_TIME = 10
    NUM_FEAT = 8
    NUM_AUX = 26
elif DATASET == 'adni':
    NUM_TIME = 12
    NUM_FEAT = 4    # FDG, AV45, Hippocampus, Entorhinal
    NUM_AUX = 7     # AGE, PTGENDER, PTEDUCAT, PTETHCAT, PTRACCAT, PTMARRY, FAQ
else:
    NUM_FEAT = 10    # d: time-variant features per timestep
    NUM_TIME = 10
    NUM_AUX = 10     # baseline (time-invariant) features

# group-based acquisition: 22 individual features + 10 one-hot groups = 32 units
if DATASET in ('cheears_demog', 'cheears'):
    NUM_GROUPS = 32
elif DATASET == 'cheears_day_context':
    NUM_GROUPS = 31
else:
    NUM_GROUPS = NUM_FEAT  # no grouping

# ── Per-unit acquisition costs ──────────────────────────────────────────
# Longitudinal: 31 acquirable units (22 individual + 9 groups)
# Edit values to reflect real-world costs (default 1.0 = uniform cost)
if DATASET == 'cheears_demog':
    LONGITUDINAL_FEATURE_COSTS = {
        # Individual features (rows 0-21 of group-to-feat matrix)
        'happy': 1.0,
        'nervous': 1.0,
        'angry': 1.0,
        'sad': 1.0,
        'excited': 1.0,
        'alert': 1.0,
        'ashamed': 1.0,
        'relaxed': 1.0,
        'bored': 1.0,
        'content': 1.0,
        'stress': 1.0,
        'drink_plans': 1.0,
        'substance': 1.0,
        'dom': 1.0,
        'warm': 1.0,
        'drink_likely': 1.0,
        'drink_quantity': 1.0,
        'drink_urge': 1.0,
        'nondrink_likely': 1.0,
        'nondrink_quantity': 1.0,
        'nondrink_urge': 1.0,
        'nondrink_plan_other': 1.0,
        # Feature groups (rows 22-30)
        'daily_activities': 1.0,
        'daily_experiences': 1.0,
        'drink_expectancies': 1.0,
        'drink_motives': 1.0,
        'general_experiences': 1.0,
        'nondrink_expectancies': 1.0,
        'nondrink_motives': 1.0,
        'nondrink_plans': 1.0,
        'social_experiences': 1.0,
        'day_of_week': 1.0,
    }

    # Static/auxiliary features: 22 individual aux gates
    STATIC_FEATURE_COSTS = {
        'sex': 1.0,
        'age': 1.0,
        'handedness': 1.0,
        'hispanic': 1.0,
        'marital_status': 1.0,
        'education': 1.0,
        'degree': 1.0,
        'current_employed': 1.0,
        'school_year': 1.0,
        'militaryaffil': 1.0,
        'family_income': 1.0,
        'religious_affiliation': 1.0,
        'physical_handicap': 1.0,
        'cigarette_use': 1.0,
        'alcohol_use': 1.0,
        'drug_use': 1.0,
        'race_0': 1.0,
        'race_1': 1.0,
        'race_2': 1.0,
        'race_3': 1.0,
        'race_4': 1.0,
        'race_5': 1.0,
    }
elif DATASET == 'cheears':
    #62.9% for majority class
    LONGITUDINAL_FEATURE_COSTS = {
        # Individual features (rows 0-21 of group-to-feat matrix)
        'happy': 1.0,
        'nervous': 1.0,
        'angry': 1.0,
        'sad': 1.0,
        'excited': 1.0,
        'alert': 1.0,
        'ashamed': 1.0,
        'relaxed': 1.0,
        'bored': 1.0,
        'content': 1.0,
        'stress': 1.0,
        'drink_plans': 1.0,
        'substance': 1.0,
        'dom': 1.0,
        'warm': 1.0,
        'drink_likely': 1.0,
        'drink_quantity': 1.0,
        'drink_urge': 1.0,
        'nondrink_likely': 1.0,
        'nondrink_quantity': 1.0,
        'nondrink_urge': 1.0,
        'nondrink_plan_other': 1.0,
        # Feature groups (rows 22-30)
        'daily_activities': 1.0,
        'daily_experiences': 1.0,
        'drink_expectancies': 1.0,
        'drink_motives': 1.0,
        'general_experiences': 1.0,
        'nondrink_expectancies': 1.0,
        'nondrink_motives': 1.0,
        'nondrink_plans': 1.0,
        'social_experiences': 1.0,
        'day_of_week': 1.0,
    }
    STATIC_FEATURE_COSTS = {f'baseline_{i}': 1.0 for i in range(NUM_AUX)}
elif DATASET == 'cheears_day_context':
    LONGITUDINAL_FEATURE_COSTS = {
        'happy': 1.0,
        'nervous': 1.0,
        'angry': 1.0,
        'sad': 1.0,
        'excited': 1.0,
        'alert': 1.0,
        'ashamed': 1.0,
        'relaxed': 1.0,
        'bored': 1.0,
        'content': 1.0,
        'stress': 1.0,
        'drink_plans': 1.0,
        'substance': 1.0,
        'dom': 1.0,
        'warm': 1.0,
        'drink_likely': 1.0,
        'drink_quantity': 1.0,
        'drink_urge': 1.0,
        'nondrink_likely': 1.0,
        'nondrink_quantity': 1.0,
        'nondrink_urge': 1.0,
        'nondrink_plan_other': 1.0,
        'daily_activities': 1.0,
        'daily_experiences': 1.0,
        'drink_expectancies': 1.0,
        'drink_motives': 1.0,
        'general_experiences': 1.0,
        'nondrink_expectancies': 1.0,
        'nondrink_motives': 1.0,
        'nondrink_plans': 1.0,
        'social_experiences': 1.0,
    }
    STATIC_FEATURE_COSTS = {f'baseline_{i}': 1.0 for i in range(34)}
    STATIC_FEATURE_COSTS['day_of_week'] = 1.0
elif DATASET == 'adni':
    LONGITUDINAL_FEATURE_COSTS = {
        'FDG': 1.0,
        'AV45': 1.0,
        'Hippocampus': 0.5,
        'Entorhinal': 0.5,
    }
    STATIC_FEATURE_COSTS = {
        'AGE': 0.3,
        'PTGENDER': 0.3,
        'PTEDUCAT': 0.3,
        'PTETHCAT': 0.3,
        'PTRACCAT': 0.3,
        'PTMARRY': 0.3,
        'FAQ': 0.3,
    }
elif DATASET in ('klg', 'womac'):
    # Original 27-feat costs: [0.3]*11, [0.5]*3, [0.3]*3, [0.8]*9, [1.0]
    # Static indices [0,1,2,3,4,5,6,9,10,16] all fall in 0.3 blocks
    # Longitudinal indices [7,8,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26]
    LONGITUDINAL_FEATURE_COSTS = {
        'long_0': 0.3,   # orig idx 7
        'long_1': 0.3,   # orig idx 8
        'long_2': 0.5,   # orig idx 11
        'long_3': 0.5,   # orig idx 12
        'long_4': 0.5,   # orig idx 13
        'long_5': 0.3,   # orig idx 14
        'long_6': 0.3,   # orig idx 15
        'long_7': 0.8,   # orig idx 17
        'long_8': 0.8,   # orig idx 18
        'long_9': 0.8,   # orig idx 19
        'long_10': 0.8,  # orig idx 20
        'long_11': 0.8,  # orig idx 21
        'long_12': 0.8,  # orig idx 22
        'long_13': 0.8,  # orig idx 23
        'long_14': 0.8,  # orig idx 24
        'long_15': 0.8,  # orig idx 25
        'long_16': 1.0,  # orig idx 26
    }
    STATIC_FEATURE_COSTS = {f'baseline_{i}': 0.3 for i in range(NUM_AUX)}
else:
    LONGITUDINAL_FEATURE_COSTS = {f'feat_{i}': 1.0 for i in range(NUM_GROUPS)}
    STATIC_FEATURE_COSTS = {f'baseline_{i}': 1.0 for i in range(NUM_AUX)}

LONGITUDINAL_COST_VECTOR = list(LONGITUDINAL_FEATURE_COSTS.values())
STATIC_COST_VECTOR = list(STATIC_FEATURE_COSTS.values())

#Classifier config
if DATASET in ('cheears', 'cheears_demog', 'cheears_day_context'):
    CLASSIFIER_CONFIG = {
        'epochs': 500,
        'lr': 1e-3,
        'min_lr': 1e-6,
        'batch_size': 64,
        'patience': 50,
        'hidden_dim': 64,
        'dropout': 0.4,
    }
elif DATASET in ('klg', 'womac', 'adni'):
    CLASSIFIER_CONFIG = {
        'epochs': 500,
        'lr': 1e-3,
        'min_lr': 1e-6,
        'batch_size': 64,
        'patience': 50,
        'hidden_dim': 64,
        'dropout': 0.4,
    }
else:
    CLASSIFIER_CONFIG = {
        'epochs': 200,
        'lr': 1e-3,
        'min_lr': 1e-6,
        'batch_size': 64,
        'patience': 10,
        'hidden_dim': 32,
        'dropout': 0.3,
    }

#Oracle config
ORACLE_CONFIG = {
    'num_samples': 1000,
    'cost_weight': 0.001,
    'time_weight': 0.0,
    'feature_costs': LONGITUDINAL_COST_VECTOR,
    'aux_feature_costs': STATIC_COST_VECTOR,
}

#Actor config
if DATASET in ('cheears', 'cheears_demog', 'cheears_day_context'):
    ACTOR_CONFIG = {
        'epochs': 500,
        'lr': 1e-3,
        'batch_size': 64,
        'val_batch_size': 256,
        'patience': 10,
        'gate_tau': 1.0,
        'init_logit': 0.0,
        'threshold': 0.5,
        'cost_weight': 0.01,
        'time_emb_dim': 64,
        'planner_hidden': [256, 128, 64],
        'init_logit': 0.0,          # 2.0 start aux gates biased toward acquiring
        'aux_cost_weight': 0.001,
        'feature_costs': LONGITUDINAL_COST_VECTOR,
        'aux_feature_costs': STATIC_COST_VECTOR,
    }
elif DATASET in ('klg', 'womac'):
    ACTOR_CONFIG = {
        'epochs': 500,
        'lr': 1e-3,
        'batch_size': 64,
        'val_batch_size': 256,
        'patience': 10,
        'gate_tau': 1.0,
        'threshold': 0.5,
        'cost_weight': 0.05,
        'time_emb_dim': 64,
        'planner_hidden': [256, 128, 64],
        'init_logit': 0.0,
        'aux_cost_weight': 0.001,
        'feature_costs': LONGITUDINAL_COST_VECTOR,
        'aux_feature_costs': STATIC_COST_VECTOR,
    }
elif DATASET == 'adni':
    ACTOR_CONFIG = {
        'epochs': 500,
        'lr': 1e-3,
        'batch_size': 64,
        'val_batch_size': 256,
        'patience': 10,
        'gate_tau': 1.0,
        'threshold': 0.5,
        'cost_weight': 0.05,
        'time_emb_dim': 64,
        'planner_hidden': [256, 128, 64],
        'init_logit': 0.0,
        'aux_cost_weight': 0.001,
        'feature_costs': LONGITUDINAL_COST_VECTOR,
        'aux_feature_costs': STATIC_COST_VECTOR,
    }
else:
    ACTOR_CONFIG = {
        'epochs': 500,
        'lr': 1e-3,
        'batch_size': 64,
        'val_batch_size': 256,
        'patience': 10,
        'gate_tau': 1.0,
        'init_logit': 0.0,
        'threshold': 0.5,
        'cost_weight': 0.05,
        'time_emb_dim': 64,
        'planner_hidden': [512, 256, 128],
        'init_logit': 0.0,          # start aux gates biased toward acquiring
        'aux_cost_weight': 0.001,
        'feature_costs': LONGITUDINAL_COST_VECTOR,
        'aux_feature_costs': STATIC_COST_VECTOR,
    }

# Planner dictionary defaults (used only when planner_mode='dictionary')
ACTOR_CONFIG.update({
    'planner_mode': 'direct',        # direct | dictionary
    'dict_mode': 'global',           # global | timestep
    'dict_num_templates': 30,        # number of canonical plans (B)
    'dict_use_st': True,             # straight-through hard selection
    'dict_tau0': 1.0,                # initial Gumbel-Softmax temperature
    'dict_tau_min': 0.2,             # min annealed temperature
    'dict_tau_gamma': 1.0,           # set <1.0 to enable annealing
    'dict_tau_decay': 0.95,          # per pseudo-epoch multiplicative decay
    'dict_div_lambda': 0.0,          # diversity regularizer weight
    'dict_sparse_lambda': 0.0,       # sparsity regularizer weight
    'dict_sparse_warmup_frac': 0.3,  # initial epoch fraction with zero sparsity
    'dict_init': 'random',           # random | orthogonal | kmeans
    'dict_init_eps': 1e-4,           # clip epsilon before logit init
    'dict_kmeans_seed': 42,
    'dict_kmeans_n_init': 20,
    'dict_lr': 1e-4,                 # TTUR dictionary learning rate
    'grad_clip_norm': 1.0,           # gradient clipping threshold
})

ITERATIVE_ACTOR_CONFIG = {
    **ACTOR_CONFIG,
    'total_batches': 1000,
    'warmup_batches': 50,
    'oracle_mix_ratio': 0.0,
    'eval_every': 50,
    'log_every': 10,
}

JOINT_ITERATIVE_ACTOR_CONFIG = {
    **ITERATIVE_ACTOR_CONFIG,
    'classifier_lr': 1e-4,      # lower LR for classifier fine-tuning
    'anchor_weight': 1.0,       # L2 anchor regularisation strength
    'mask_aug_ratio': 0.5,      # random mask augmentation probability
}
