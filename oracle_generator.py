"""
Oracle Rollout Generator for Longitudinal Feature Acquisition
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm
from utils import sample_future_data


class OracleRolloutGenerator:
    """Generate oracle rollouts using the trained predictor"""

    def __init__(self, predictor, mask_layer, num_time, num_feat, device,
                 cost_weight=0.0, time_weight=0.0, feature_costs=None,
                 num_samples=1000, num_aux=0, aux_cost_weight=None,
                 num_groups=None, group_to_feat_matrix=None,
                 aux_feature_costs=None):
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.num_time = num_time
        self.num_feat = num_feat
        self.device = device
        self.cost_weight = cost_weight
        self.time_weight = time_weight
        self.num_samples = num_samples
        self.num_aux = num_aux
        self.aux_cost_weight = aux_cost_weight if aux_cost_weight is not None else cost_weight
        self.num_groups = num_groups if num_groups is not None else num_feat

        # Group-to-feature expansion matrix
        if group_to_feat_matrix is not None:
            self.group_to_feat = group_to_feat_matrix.to(device)
        else:
            self.group_to_feat = torch.eye(num_feat, device=device)

        # Feature costs are per-group
        if feature_costs is None:
            self.feature_costs = torch.ones(self.num_groups, device=device)
        elif isinstance(feature_costs, (list, np.ndarray)):
            self.feature_costs = torch.tensor(feature_costs, device=device, dtype=torch.float32)
        else:
            self.feature_costs = feature_costs.to(device)

        # Expand to (T * num_groups)
        self.feature_costs_flat = self.feature_costs.unsqueeze(0).expand(
            num_time, -1
        ).reshape(-1)

        # Per-aux-feature costs (resize to match num_aux if needed)
        if aux_feature_costs is None:
            self.aux_feature_costs = torch.ones(max(num_aux, 1), device=device)
        elif isinstance(aux_feature_costs, (list, np.ndarray)):
            self.aux_feature_costs = torch.tensor(aux_feature_costs, device=device, dtype=torch.float32)
        else:
            self.aux_feature_costs = aux_feature_costs.to(device)

        if num_aux > 0 and len(self.aux_feature_costs) != num_aux:
            print(f"WARNING: aux_feature_costs length ({len(self.aux_feature_costs)}) != num_aux ({num_aux}), resizing with 1.0 padding")
            new_costs = torch.ones(num_aux, device=device)
            n = min(len(self.aux_feature_costs), num_aux)
            new_costs[:n] = self.aux_feature_costs[:n]
            self.aux_feature_costs = new_costs

        self.predictor.to(device)
        self.predictor.eval()

    def expand_groups_to_feat(self, group_mask):
        """Expand (B, T*num_groups) -> (B, T*num_feat)"""
        B = group_mask.size(0)
        g = group_mask.reshape(B, self.num_time, self.num_groups)
        f = torch.matmul(g, self.group_to_feat)
        return f.reshape(B, self.num_time * self.num_feat)

    def feat_to_groups(self, feat_mask):
        """Collapse (B, T*num_feat) -> (B, T*num_groups), any-in-group -> 1"""
        B = feat_mask.size(0)
        f = feat_mask.reshape(B, self.num_time, self.num_feat)
        g = torch.matmul(f, self.group_to_feat.T)
        g = (g > 0).float()
        return g.reshape(B, self.num_time * self.num_groups)

    def _find_best_aux_mask(self, x_flat, y, x_static, mask_static):
        """Find the best global aux mask by sampling random subsets."""
        B = x_flat.size(0)
        best_obj = float('inf')
        best_mask = torch.zeros(self.num_aux, device=self.device)

        for _ in range(self.num_samples):
            # Sample a random subset size k, then pick exactly k features
            k = torch.randint(0, self.num_aux + 1, (1,)).item()
            perm = torch.randperm(self.num_aux, device=self.device)
            rand_mask = torch.zeros(self.num_aux, device=self.device)
            rand_mask[perm[:k]] = 1.0
            aux_mask = rand_mask.unsqueeze(0).expand(B, -1) * mask_static
            aux_acquired = x_static * aux_mask

            total_ce = 0.0
            m_all = torch.ones(B, self.num_time * self.num_feat, device=self.device)
            for k in range(self.num_time):
                m_k = m_all.clone()
                if k < self.num_time - 1:
                    m_k[:, (k + 1) * self.num_feat:] = 0

                t_frac = (k + 1) / self.num_time
                t_ind = torch.full((B, 1), t_frac, device=self.device)
                x_masked = self.mask_layer(x_flat, m_k)
                x_in = torch.cat([t_ind, x_masked, aux_acquired], dim=1)
                preds = self.predictor(x_in)

                y_target = y[:, k]
                step_loss = torch.nn.functional.cross_entropy(
                    preds, y_target.long(), ignore_index=-1, reduction='mean'
                )
                total_ce += step_loss.item()

            aux_cost = (rand_mask * self.aux_feature_costs).sum().item() * self.aux_cost_weight
            obj = total_ce + aux_cost

            if obj < best_obj:
                best_obj = obj
                best_mask = rand_mask.clone()

        return best_mask

    def select_next_group(self, m_curr_groups, cur_t, selected_groups):
        """
        Select earliest-timestep groups from selected subset.
        """
        B = m_curr_groups.shape[0]
        ng = self.num_groups
        selected_3d = selected_groups.reshape(B, self.num_time, ng)

        has_acq_at_t = selected_3d.sum(dim=2) > 0  # [B, T]
        first_t = torch.argmax(has_acq_at_t.int(), dim=1)  # [B]

        update_mask = torch.zeros_like(selected_3d)
        batch_indices = torch.arange(B, device=self.device)
        valid_rows = selected_groups.sum(dim=1) > 0

        if valid_rows.any():
            update_mask[batch_indices[valid_rows], first_t[valid_rows], :] = \
                selected_3d[batch_indices[valid_rows], first_t[valid_rows], :]

        update_mask_flat = update_mask.reshape(B, -1)
        m_next = m_curr_groups + update_mask_flat

        cur_t_next = cur_t.clone()
        if valid_rows.any():
            cur_t_next[valid_rows] = (first_t[valid_rows] + 1).int()

        return m_next, cur_t_next

    def generate(self, dataloader, save_path=None):
        """Generate oracle rollouts by selecting optimal feature subsets."""
        all_x, all_m, all_t, all_y = [], [], [], []
        all_xs, all_ms = [], []
        global_aux_mask = None

        ng = self.num_groups

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating Oracle Rollouts"):
                # Unpack batch
                if len(batch) == 5:
                    x, y, m_avail, x_static, mask_static = batch
                    x_static = torch.nan_to_num(x_static).float().to(self.device)
                    mask_static = mask_static.float().to(self.device)
                elif len(batch) == 3:
                    x, y, m_avail = batch
                    x_static = None
                    mask_static = None
                else:
                    x, y, m_avail, _ = batch
                    x_static = None
                    mask_static = None

                x = torch.nan_to_num(x).to(self.device)
                y = y.to(self.device)
                m_avail = m_avail.to(self.device)

                B = x.shape[0]

                if x.dim() == 3:
                    x_flat = x.reshape(B, -1)
                else:
                    x_flat = x
                if m_avail.dim() == 3:
                    m_avail_flat = m_avail.reshape(B, -1)
                else:
                    m_avail_flat = m_avail

                # Compute group-level availability
                m_avail_groups = self.feat_to_groups(m_avail_flat.float())

                # Stage 1: find best aux mask (once on first batch)
                aux_acquired = None
                if self.num_aux > 0 and x_static is not None:
                    if global_aux_mask is None:
                        print("Finding aux mask...")
                        global_aux_mask = self._find_best_aux_mask(
                            x_flat, y, x_static, mask_static
                        )
                        n_aux_selected = int(global_aux_mask.sum().item())
                        print(f"selected {n_aux_selected}/{self.num_aux} aux features")

                    aux_mask_expanded = global_aux_mask.unsqueeze(0).expand(B, -1) * mask_static
                    aux_acquired = x_static * aux_mask_expanded

                # Initialize state at group level
                m_curr_groups = torch.zeros(B, self.num_time * ng, dtype=torch.float32, device=self.device)
                m_curr_feat = torch.zeros_like(x_flat, dtype=torch.float32)
                cur_t = torch.zeros(B, dtype=torch.int, device=self.device)
                m_done = torch.zeros(B, dtype=torch.bool, device=self.device)

                for step in range(2 * self.num_time):
                    # Save current state (feature-level masks)
                    active_mask = ~m_done
                    if active_mask.any():
                        all_x.append(x_flat[active_mask].cpu())
                        all_m.append(m_curr_feat[active_mask].cpu())
                        all_y.append(y[active_mask].cpu())
                        all_t.append(cur_t[active_mask].cpu())
                        if x_static is not None:
                            all_xs.append(x_static[active_mask].cpu())
                            all_ms.append(mask_static[active_mask].cpu())

                    # Valid groups to acquire
                    t_grid = torch.arange(self.num_time, device=self.device).unsqueeze(0).expand(B, -1)
                    time_mask = t_grid >= cur_t.unsqueeze(1)
                    time_mask_g = time_mask.unsqueeze(-1).expand(-1, -1, ng).reshape(B, -1)

                    valid_mask_g = (m_avail_groups > 0) & (m_curr_groups == 0) & time_mask_g
                    valid_counts = torch.sum(valid_mask_g, dim=1)

                    if ((valid_counts == 0) | m_done).all():
                        break

                    # Repeat for sampling (at group level)
                    m_curr_g_rep = m_curr_groups.repeat_interleave(self.num_samples, dim=0)
                    m_curr_f_rep = m_curr_feat.repeat_interleave(self.num_samples, dim=0)
                    valid_mask_g_rep = valid_mask_g.repeat_interleave(self.num_samples, dim=0)
                    x_rep = x_flat.repeat_interleave(self.num_samples, dim=0)
                    y_rep = y.repeat_interleave(self.num_samples, dim=0)
                    cur_t_rep = cur_t.repeat_interleave(self.num_samples, dim=0)

                    aux_acquired_rep = None
                    if aux_acquired is not None:
                        aux_acquired_rep = aux_acquired.repeat_interleave(self.num_samples, dim=0)

                    # Sample random group subsets
                    selected_groups = sample_future_data(valid_mask_g_rep)  # (B*S, T*ng)
                    m_cand_groups = m_curr_g_rep + selected_groups
                    # Expand to feature level for prediction
                    m_cand_feat = self.expand_groups_to_feat(m_cand_groups).clamp(0, 1)

                    # Evaluate each candidate
                    total_ce_loss = torch.zeros(B * self.num_samples, device=self.device)
                    m_cand_3d = m_cand_feat.reshape(-1, self.num_time, self.num_feat)

                    for k in range(self.num_time):
                        relevant_mask = (k >= cur_t_rep)
                        if not relevant_mask.any():
                            continue

                        m_k_3d = m_cand_3d.clone()
                        if k < self.num_time - 1:
                            m_k_3d[:, k+1:, :] = 0
                        m_k_flat = m_k_3d.reshape(len(m_cand_feat), -1)

                        t_frac = (k + 1) / self.num_time
                        t_ind = torch.full((len(m_cand_feat), 1), t_frac, device=self.device)

                        x_masked = self.mask_layer(x_rep, m_k_flat)
                        if aux_acquired_rep is not None:
                            x_in = torch.cat([t_ind, x_masked, aux_acquired_rep], dim=1)
                        else:
                            x_in = torch.cat([t_ind, x_masked], dim=1)
                        preds = self.predictor(x_in)

                        y_target = y_rep[:, k]
                        step_loss = torch.nn.functional.cross_entropy(
                            preds, y_target.long(), ignore_index=-1, reduction='none'
                        )
                        total_ce_loss += (step_loss * relevant_mask.float())

                    # Cost at group level
                    acq_cost = (selected_groups * self.feature_costs_flat).sum(dim=1) * self.cost_weight
                    total_obj = total_ce_loss + acq_cost

                    # Select best subset
                    total_obj = total_obj.view(B, self.num_samples)
                    best_indices = torch.argmin(total_obj, dim=1)

                    flat_indices = torch.arange(B, device=self.device) * self.num_samples + best_indices
                    best_subset_g = selected_groups[flat_indices]

                    # Update state at group level
                    m_next_g, cur_t_next = self.select_next_group(m_curr_groups, cur_t, best_subset_g)

                    added = (m_next_g - m_curr_groups).sum(dim=1)
                    m_done = m_done | (added == 0)

                    m_curr_groups = torch.where(m_done.unsqueeze(1).expand_as(m_curr_groups), m_curr_groups, m_next_g)
                    cur_t = torch.where(m_done, cur_t, cur_t_next)
                    # Update feature-level mask from groups
                    m_curr_feat = self.expand_groups_to_feat(m_curr_groups).clamp(0, 1)

        # Create dataset (feature-level masks for compatibility)
        x_all = torch.cat(all_x, dim=0).numpy()
        y_all = torch.cat(all_y, dim=0).numpy()
        m_all = torch.cat(all_m, dim=0).numpy()
        t_all = torch.cat(all_t, dim=0).numpy()

        print(f"\nGenerated {len(x_all)} oracle states")
        print(f"  x shape: {x_all.shape}")
        print(f"  y shape: {y_all.shape}")
        print(f"  mask shape: {m_all.shape}")
        print(f"  t shape: {t_all.shape}")

        if save_path:
            save_dict = dict(x=x_all, y=y_all, mask=m_all, t=t_all)

            if all_xs:
                xs_all = torch.cat(all_xs, dim=0).numpy()
                ms_all = torch.cat(all_ms, dim=0).numpy()
                aux_mask_np = global_aux_mask.cpu().numpy() if global_aux_mask is not None else None
                save_dict['x_static'] = xs_all
                save_dict['mask_static'] = ms_all
                if aux_mask_np is not None:
                    save_dict['aux_mask'] = aux_mask_np
                print(f"  x_static shape: {xs_all.shape}")
                print(f"  aux_mask: {aux_mask_np}")

            np.savez(save_path, **save_dict)
            print(f"Saved oracle rollout to {save_path}")

        dataset = TensorDataset(
            torch.from_numpy(x_all),
            torch.from_numpy(y_all),
            torch.from_numpy(m_all),
            torch.from_numpy(t_all)
        )

        return dataset
