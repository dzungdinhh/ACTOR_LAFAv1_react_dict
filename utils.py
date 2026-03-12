"""
Utility functions for the project
"""
import torch
import numpy as np
import random
import math


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    Create sinusoidal timestep embeddings
    
    Args:
        timesteps: (batch,) int or float tensor
        embedding_dim: dimension of the embedding
        
    Returns:
        embedding: (batch, embedding_dim) float32 tensor
    """
    device = timesteps.device
    half_dim = embedding_dim // 2

    # Compute frequencies
    freq_exponent = -math.log(10000.0) / (half_dim - 1)
    freqs = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=device) * freq_exponent
    )

    # Outer product: (batch, half_dim)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)

    # Sin + cos concat
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    return embedding


def sample_future_data(valid_mask):
    """
    Randomly sample subsets of valid features
    
    Args:
        valid_mask: (B, F) binary mask of valid positions
        
    Returns:
        selected: (B, F) binary mask of selected features
    """
    B, F = valid_mask.shape
    device = valid_mask.device
    
    # Count valid features per sample
    valid_counts = torch.sum(valid_mask, dim=1)
    
    # Random subset sizes (uniform from 0 to valid_count)
    subset_sizes = torch.ceil(
        torch.rand(B, device=device) * valid_counts.float()
    ).long()
    
    # Create random indices and sort
    indices = torch.rand(B, F, device=device)
    indices[valid_mask == 0] = float('inf')
    sorted_indices = torch.argsort(indices, dim=1)
    
    # Create keep mask based on subset sizes
    keep_mask = torch.arange(F, device=device).unsqueeze(0) < subset_sizes.unsqueeze(1)
    
    # Scatter to get selected mask
    selected_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    valid_sorted = torch.gather(valid_mask.bool(), 1, sorted_indices)
    selected_mask.scatter_(1, sorted_indices, valid_sorted & keep_mask)
    
    return selected_mask.float()


def generate_uniform_mask(batch_size, num_features, num_time=None):
    """Generate random uniform masks for pretraining.
    P(K=k) = 1/(n+1) — uniform over number of acquired features."""
    if num_time is None:
        unif = torch.rand(batch_size, num_features)
        ref = torch.rand(batch_size, 1)
    else:
        unif = torch.rand(batch_size, num_time, num_features)
        ref = torch.rand(batch_size, num_time, 1)
    return (unif > ref).float()


def generate_bernoulli_mask(batch_size, num_features, num_time=None, p=0.5):
    """Generate independent Bernoulli(p) masks for pretraining.
    Each feature acquired independently with probability p."""
    if num_time is None:
        return (torch.rand(batch_size, num_features) < p).float()
    else:
        return (torch.rand(batch_size, num_time, num_features) < p).float()


def build_group_to_feat_matrix(num_feat=142):
    """
    Build the (31, 142) group-to-feature expansion matrix for cheears_demog.

    Rows 0-21: individual features (identity mapping)
    Rows 22-30: one-hot groups (daily_activities, daily_experiences, etc.)

    Returns:
        torch.Tensor of shape (31, 142)
    """
    try:
        from cheears_demog.feature_groups import (  # type: ignore
            LONGITUDINAL_INDIVIDUAL_FEATURE_INDICES,
            LONGITUDINAL_FEATURE_GROUP_INDICES,
        )
    except ImportError:
        # Local fallback used in this workspace clone.
        from feature_groups import (  # type: ignore
            LONGITUDINAL_INDIVIDUAL_FEATURE_INDICES,
            LONGITUDINAL_FEATURE_GROUP_INDICES,
        )

    individual = LONGITUDINAL_INDIVIDUAL_FEATURE_INDICES
    # Only include groups whose indices fit within num_feat
    groups = {
        name: idxs
        for name, idxs in LONGITUDINAL_FEATURE_GROUP_INDICES.items()
        if all(i < num_feat for i in idxs)
    }

    num_groups = len(individual) + len(groups)
    matrix = torch.zeros(num_groups, num_feat)

    # Individual features: rows 0-21
    for i, idx in enumerate(individual):
        matrix[i, idx] = 1.0

    # Group features: rows 22-30
    for g, name in enumerate(groups.keys()):
        row = len(individual) + g
        for feat_idx in groups[name]:
            matrix[row, feat_idx] = 1.0

    return matrix


def generate_random_masks_for_cur_t(x_all, y_all, cur_t_all, num_time, num_feat,
                                     num_samples_per_state=1,
                                     num_groups=None, group_to_feat_np=None):
    """
    Generate random acquisition masks for timesteps <= cur_t for data augmentation.

    If num_groups and group_to_feat_np are provided, generates group-level
    random masks and expands to feature level.

    Returns:
        random_masks: (N * num_samples, F) -- feature-level masks
        random_cur_t: (N * num_samples,)
        random_x: (N * num_samples, F)
        random_y: (N * num_samples, T)
    """
    N, F = x_all.shape
    assert F == num_time * num_feat

    use_groups = (num_groups is not None and group_to_feat_np is not None)
    gate_dim = num_groups if use_groups else num_feat

    random_masks = []
    random_cur_t = []
    random_x = []
    random_y = []

    for i in range(N):
        x_instance = x_all[i]
        y_instance = y_all[i]
        cur_t = int(cur_t_all[i])

        for _ in range(num_samples_per_state):
            if use_groups:
                mask_g = np.zeros((num_time, num_groups), dtype=np.float32)
                for t in range(cur_t):
                    for g in range(num_groups):
                        if np.random.rand() > 0.5:
                            mask_g[t, g] = 1.0
                mask_f = (mask_g @ group_to_feat_np).clip(0, 1)
                mask = mask_f.reshape(-1)
            else:
                mask = np.zeros(F, dtype=np.float32)
                mask_2d = mask.reshape(num_time, num_feat)
                for t in range(cur_t):
                    for feat_idx in range(num_feat):
                        if np.random.rand() > 0.5:
                            mask_2d[t, feat_idx] = 1.0
                mask = mask_2d.reshape(-1)

            random_masks.append(mask)
            random_cur_t.append(cur_t)
            random_x.append(x_instance)
            random_y.append(y_instance)

    random_masks = np.stack(random_masks)
    random_cur_t = np.array(random_cur_t)
    random_x = np.stack(random_x)
    random_y = np.stack(random_y)

    return random_masks, random_cur_t, random_x, random_y
