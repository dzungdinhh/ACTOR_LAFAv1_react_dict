"""
Dataset classes for longitudinal data
"""
import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset


class LongitudinalDataset(TorchDataset):
    """Dataset for longitudinal data with missing values"""

    def __init__(self, x, y, mask, x_static=None, mask_static=None):
        """
        Args:
            x: Features (N, T, d) or (N, T*d)
            y: Labels (N, T)
            mask: Availability mask (N, T, d) or (N, T*d)
            x_static: Static/auxiliary features (N, num_aux) or None
            mask_static: Static availability mask (N, num_aux) or None
        """
        super().__init__()
        self.x = x
        self.y = y
        self.mask = mask
        self.x_static = x_static
        self.mask_static = mask_static

        # Get dimensions
        unique_y = np.unique(y)
        self.y_dim = len(unique_y) - 1 if -1 in unique_y else len(unique_y)

        if x.ndim == 3:
            self.t = x.shape[1]
            self.x_dim = x.shape[2]
        else:
            # Infer from shape
            self.t = y.shape[1] if y.ndim > 1 else 10  # Default to 10
            self.x_dim = x.shape[1] // self.t

        self.num_aux = x_static.shape[1] if x_static is not None else 0

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        mask = self.mask[index]
        if self.x_static is not None:
            return x, y, mask, self.x_static[index], self.mask_static[index]
        return x, y, mask


class ActorDataset(TorchDataset):
    """Dataset for training the actor-critic model"""

    def __init__(self, x, y, mask, time_emb, cur_t):
        """
        Args:
            x: Features (N, F)
            y: Labels (N, T)
            mask: Acquired features mask (N, F)
            time_emb: Time embeddings (N, emb_dim)
            cur_t: Current timestep (N,)
        """
        super().__init__()
        self.x = x
        self.y = y
        self.mask = mask
        self.time_emb = time_emb
        self.cur_t = cur_t

        unique_y = np.unique(y)
        self.y_dim = len(unique_y) - 1 if -1 in unique_y else len(unique_y)
        self.x_dim = x.shape[-1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return (
            self.x[index],
            self.y[index],
            self.mask[index],
            self.time_emb[index],
            self.cur_t[index]
        )


def load_synthetic_data(data_path, num_static=10):
    """Load synthetic longitudinal data from .npz file

    X has shape (N, T, F) where the first `num_static` features are
    time-invariant baselines and the rest are time-variant.  We split
    them into x_static (from timestep 0) and x (time-variant only).

    Args:
        data_path: Path to .npz file with keys 'x' (N,T,F) and 'y' (N,T)
        num_static: Number of leading baseline features (default 10)
    """
    file = np.load(data_path)
    x_raw = file['x'].astype('float32')   # (N, T, F)
    y = file['y'].astype('int64')          # (N, T)

    # Split into static (baselines from t=0) and longitudinal (time-variant)
    x_static = x_raw[:, 0, :num_static]                # (N, num_static)
    x = x_raw[:, :, num_static:]                        # (N, T, num_variant)

    # Availability masks (0 where NaN, 1 otherwise)
    mask_static = np.where(np.isnan(x_static), 0, 1).astype('float32')
    mask = np.where(np.isnan(x), 0, 1).astype('float32')

    # Replace NaN with 0
    x = np.nan_to_num(x)
    x_static = np.nan_to_num(x_static)

    dataset = LongitudinalDataset(x, y, mask, x_static, mask_static)
    return dataset


# Indices of time-invariant (baseline) features in OAI data
OAI_STATIC_INDICES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 16]


def _load_oai_data(data_path, task):
    """Shared loader for OAI datasets (KLG / WOMAC).

    X has shape (N, T, F).  Features at OAI_STATIC_INDICES are baselines
    (taken from t=0); the rest are time-variant longitudinal features.

    Args:
        data_path: Path to .npz file with keys 'x' and the task key
        task: Label key in the file ('KLG' or 'WOMAC')
    """
    file = np.load(data_path)
    x_raw = file['x'].astype('float32')        # (N, T, F)
    y = file[task].astype('float64')            # (N, T)  — float first for NaN handling

    # Clean labels: NaN → -1, negative (except -1) → -1
    y[np.isnan(y)] = -1
    y[(y < 0) & (y != -1)] = -1

    # Task-specific label transforms
    if task == 'WOMAC':
        y = np.where((y >= 0) & (y < 5), 0, y)
        y = np.where(y >= 5, 1, y)
    elif task == 'KLG':
        y = np.where(y >= 1, y - 1, y)

    y = y.astype('int64')

    # Split features into static (baselines from t=0) and longitudinal
    static_idx = np.array(OAI_STATIC_INDICES)
    all_idx = np.arange(x_raw.shape[2])
    long_idx = np.setdiff1d(all_idx, static_idx)

    x_static = x_raw[:, 0, static_idx]          # (N, num_static)
    x = x_raw[:, :, long_idx]                    # (N, T, num_long)

    # Availability masks
    mask_static = np.where(np.isnan(x_static), 0, 1).astype('float32')
    mask = np.where(np.isnan(x), 0, 1).astype('float32')

    # Replace NaN with 0
    x = np.nan_to_num(x)
    x_static = np.nan_to_num(x_static)

    dataset = LongitudinalDataset(x, y, mask, x_static, mask_static)
    return dataset


def load_klg_data(data_path):
    """Load OAI data with KLG labels (classes merged: 0+1→0, 2→1, 3→2, 4→3)."""
    return _load_oai_data(data_path, task='KLG')


def load_womac_data(data_path):
    """Load OAI data with WOMAC labels (binary: <5→0, >=5→1)."""
    return _load_oai_data(data_path, task='WOMAC')


def load_cheears_data(data_path):
    """Load CHEEARS longitudinal data with static features from .npz file"""
    file = np.load(data_path)
    x = file['x'].astype('float32')           # (N, T, d)
    y = file['y'].astype('int64')              # (N, T)
    mask = file['mask'].astype('float32')      # (N, T, d)
    x_static = file['x_static'].astype('float32')      # (N, num_aux)
    mask_static = file['mask_static'].astype('float32') # (N, num_aux)

    # Replace NaN with 0
    x = np.nan_to_num(x)
    x_static = np.nan_to_num(x_static)

    dataset = LongitudinalDataset(x, y, mask, x_static, mask_static)
    return dataset

def load_ILIADD_data(data_path):
    """Load ILIADD longitudinal data with static features from .npz file"""
    file = np.load(data_path)
    x = file['x'].astype('float32')           # (N, T, d)
    y = file['y'].astype('int64')              # (N, T)
    mask = file['mask'].astype('float32')      # (N, T, d)
    x_static = file['x_static'].astype('float32')      # (N, num_aux) or (N, 1, num_aux)
    mask_static = file['mask_static'].astype('float32') # (N, num_aux) or (N, 1, num_aux)

    # Squeeze extra dimension if x_static is 3D (N, 1, num_aux) -> (N, num_aux)
    if x_static.ndim == 3:
        x_static = x_static.squeeze(axis=1)
    if mask_static.ndim == 3:
        mask_static = mask_static.squeeze(axis=1)

    # Replace NaN with 0
    x = np.nan_to_num(x)
    x_static = np.nan_to_num(x_static)

    dataset = LongitudinalDataset(x, y, mask, x_static, mask_static)
    return dataset


ADNI_BASELINE_FEATURES = ["AGE", "PTGENDER", "PTEDUCAT", "PTETHCAT", "PTRACCAT", "PTMARRY", "FAQ"]
ADNI_LONGITUDINAL_FEATURES = ["FDG", "AV45", "Hippocampus", "Entorhinal"]


def load_adni_data(data_path):
    """Load ADNI longitudinal data from .npz file.

    Selects 7 baseline (static) and 4 longitudinal features, separates them,
    and returns a LongitudinalDataset with 5-tuple items.

    Data keys expected: 'x' (N,T,F), 'y' (N,T,1), 'mask' (N,T), 'feat_list'.
    """
    file = np.load(data_path, allow_pickle=True)
    x_raw = file['x'].astype('float32')      # (N, T, F_all)
    y = file['y'].astype('float64')           # (N, T, 1)
    mask_time = file['mask'].astype('float32')  # (N, T)

    # Squeeze label dimension: (N, T, 1) -> (N, T)
    if y.ndim == 3:
        y = y[:, :, 0]

    # Labels: NaN -> -1
    y[np.isnan(y)] = -1
    y = y.astype('int64')

    # Resolve feature indices
    feat_list = [str(f) for f in file['feat_list']]
    baseline_idx = [feat_list.index(f) for f in ADNI_BASELINE_FEATURES]
    long_idx = [feat_list.index(f) for f in ADNI_LONGITUDINAL_FEATURES]

    # Static features from t=0
    x_static = x_raw[:, 0, baseline_idx]                # (N, 7)
    mask_static = np.where(np.isnan(x_static), 0, 1).astype('float32')
    x_static = np.nan_to_num(x_static)

    # Longitudinal features (all timesteps)
    x = x_raw[:, :, long_idx]                            # (N, T, 4)
    # Feature-level mask from NaN + time-level mask
    feat_mask = np.where(np.isnan(x), 0, 1).astype('float32')  # (N, T, 4)
    mask = feat_mask * mask_time[:, :, None]             # zero out missing timesteps
    x = np.nan_to_num(x)

    dataset = LongitudinalDataset(x, y, mask, x_static, mask_static)
    return dataset


def load_oracle_rollout(rollout_path, num_time, num_feat):
    """
    Load oracle rollout data for actor training

    Args:
        rollout_path: Path to .npz file
        num_time: Number of timesteps
        num_feat: Number of features per timestep

    Returns:
        x_all: Features (N, F)
        y_all: Labels (N, T)
        m_pred: Acquired masks (N, F)
        cur_t: Current timesteps (N,)
        x_static: Static features (N, num_aux) or None
        mask_static: Static availability mask (N, num_aux) or None
        aux_mask: Acquired aux mask (N, num_aux) or None
    """
    data = np.load(rollout_path)

    x_all = data['x']  # (N, F)
    y_all = data['y']  # (N, T)
    m_pred = data['mask']  # (N, F)
    cur_t = data['t']  # (N,)

    # Optional aux fields
    x_static = data['x_static'] if 'x_static' in data else None
    mask_static = data['mask_static'] if 'mask_static' in data else None
    aux_mask = data['aux_mask'] if 'aux_mask' in data else None

    print(f"Loaded oracle rollout:")
    print(f"  x shape: {x_all.shape}")
    print(f"  y shape: {y_all.shape}")
    print(f"  mask shape: {m_pred.shape}")
    print(f"  t shape: {cur_t.shape}")
    if x_static is not None:
        print(f"  x_static shape: {x_static.shape}")
        print(f"  aux_mask shape: {aux_mask.shape if aux_mask is not None else 'None'}")

    return x_all, y_all, m_pred, cur_t, x_static, mask_static, aux_mask
