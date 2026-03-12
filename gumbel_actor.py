"""
Gumbel Actor-Critic for Longitudinal Feature Acquisition
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models import GumbelSigmoid, PlannerNet, MaskLayer
from utils import get_timestep_embedding


class GumbelActor(pl.LightningModule):
    """Gumbel Actor-Critic model for feature acquisition"""

    def __init__(self, predictor, num_time, num_feat, config, num_aux=0,
                 num_groups=None, group_to_feat_matrix=None,
                 feature_costs=None, aux_feature_costs=None):
        """
        Args:
            predictor: Trained prediction network (frozen)
            num_time: Number of timesteps
            num_feat: Number of features per timestep
            config: Configuration dict with hyperparameters
            num_aux: Number of auxiliary/static features (0 = no aux)
            num_groups: Acquirable units per timestep (defaults to num_feat)
            group_to_feat_matrix: (num_groups, num_feat) expansion matrix
            feature_costs: Per-group costs, length num_groups (default all 1s)
            aux_feature_costs: Per-aux-feature costs, length num_aux (default all 1s)
        """
        super().__init__()
        self.predictor = predictor
        self.num_time = num_time
        self.num_feat = num_feat
        self.config = config
        self.num_aux = num_aux
        self.num_groups = num_groups if num_groups is not None else num_feat

        self.save_hyperparameters(ignore=['predictor'])
        self.automatic_optimization = False

        # Freeze predictor by default (joint script may unfreeze later)
        self.predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad = False

        # Gumbel sigmoid for direct stochastic group selection
        self.gumbel_sigmoid = GumbelSigmoid(tau=config['gate_tau'])

        # Mask layer (operates at feature level)
        self.mask_layer = MaskLayer(mask_size=num_time * num_feat, append=False)

        # Group-to-feature expansion matrix
        if group_to_feat_matrix is not None:
            self.register_buffer('group_to_feat', group_to_feat_matrix)
        else:
            self.register_buffer('group_to_feat', torch.eye(num_feat))

        # Per-group acquisition costs (length num_groups), tiled across timesteps
        if feature_costs is None:
            feature_costs = config.get('feature_costs', None)
        if feature_costs is not None:
            fc = torch.tensor(feature_costs, dtype=torch.float32)
        else:
            fc = torch.ones(self.num_groups, dtype=torch.float32)
        self.register_buffer('feature_costs', fc)
        self.register_buffer(
            'feature_costs_flat',
            fc.unsqueeze(0).expand(num_time, -1).reshape(-1),
        )

        # Per-aux-feature acquisition costs (length num_aux)
        if aux_feature_costs is None:
            aux_feature_costs = config.get('aux_feature_costs', None)
        if aux_feature_costs is not None and len(aux_feature_costs) > 0:
            afc = torch.tensor(aux_feature_costs, dtype=torch.float32)
        else:
            afc = torch.ones(max(num_aux, 1), dtype=torch.float32)
        if num_aux > 0 and len(afc) != num_aux:
            print(
                "WARNING: aux_feature_costs length "
                f"({len(afc)}) != num_aux ({num_aux}), resizing with 1.0 padding"
            )
            new_afc = torch.ones(num_aux, dtype=torch.float32)
            n = min(len(afc), num_aux)
            new_afc[:n] = afc[:n]
            afc = new_afc
        self.register_buffer('aux_feature_costs', afc)

        # Stage 1: auxiliary feature gate (fixed policy, same for all patients)
        if num_aux > 0:
            aux_init = config.get('aux_init_logit', 0.0)
            self.aux_logits = nn.Parameter(torch.full((num_aux,), aux_init))
            self.aux_cost_weight = config.get('aux_cost_weight', config['cost_weight'])
            if self.aux_cost_weight is None:
                self.aux_cost_weight = config['cost_weight']

        # Planner settings
        self.planner_mode = str(config.get('planner_mode', 'direct')).lower()
        self.dict_mode = str(config.get('dict_mode', 'global')).lower()
        self.dict_num_templates = int(config.get('dict_num_templates', 30))
        self.dict_use_st = bool(config.get('dict_use_st', True))
        self.dict_temperature = float(config.get('dict_tau0', config.get('gate_tau', 1.0)))
        self.dict_div_lambda = float(config.get('dict_div_lambda', 0.0))
        self.dict_sparse_lambda = float(config.get('dict_sparse_lambda', 0.0))

        # Planner network
        time_emb_dim = config['time_emb_dim']
        # Input: x_masked(T*d) + mask(T*d) + time_emb + aux_acquired(num_aux) + aux_gates(num_aux)
        planner_input_dim = (num_time * num_feat) * 2 + time_emb_dim + num_aux * 2

        if self.planner_mode == 'dictionary':
            if self.dict_mode == 'global':
                planner_out_dim = self.dict_num_templates
                self.dictionary_mu = nn.Parameter(
                    torch.empty(self.dict_num_templates, self.num_time, self.num_groups)
                )
                nn.init.normal_(self.dictionary_mu, mean=0.0, std=0.02)
            elif self.dict_mode == 'timestep':
                planner_out_dim = self.num_time * self.dict_num_templates
                self.dictionary_mu_step = nn.Parameter(
                    torch.empty(self.dict_num_templates, self.num_groups)
                )
                nn.init.normal_(self.dictionary_mu_step, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unsupported dict_mode: {self.dict_mode}")
        else:
            planner_out_dim = num_time * self.num_groups  # group-level gates

        self.planner_nn = PlannerNet(
            planner_input_dim,
            planner_out_dim,
            hidden=config['planner_hidden']
        )

        # Optional explicit dictionary initialization
        dict_init = str(config.get('dict_init', 'random')).lower()
        dict_init_masks_path = config.get('dict_init_masks_path', None)
        if self.planner_mode == 'dictionary' and dict_init != 'random':
            if dict_init == 'orthogonal':
                self.initialize_dictionary(init_mode='orthogonal')
            elif dict_init == 'kmeans':
                if not dict_init_masks_path:
                    raise ValueError("dict_init='kmeans' requires dict_init_masks_path")
                with np.load(dict_init_masks_path) as npz:
                    if 'masks' in npz:
                        masks_np = npz['masks']
                    elif 'all_masks' in npz:
                        masks_np = npz['all_masks']
                    else:
                        raise KeyError("K-Means masks file must contain 'masks' or 'all_masks'")
                self.initialize_dictionary(
                    init_mode='kmeans',
                    masks_np=masks_np,
                    eps=float(config.get('dict_init_eps', 1e-4)),
                    seed=int(config.get('dict_kmeans_seed', 42)),
                    n_init=int(config.get('dict_kmeans_n_init', 20)),
                )
            else:
                raise ValueError(f"Unsupported dict_init: {dict_init}")

        # Hyperparameters
        self.lr = config['lr']
        self.threshold = config['threshold']
        self.cost_weight = config['cost_weight']
        self.time_emb_dim = time_emb_dim

    def planner_parameters(self):
        """Parameters optimized for planner behavior."""
        params = list(self.planner_nn.parameters())
        if self.planner_mode == 'dictionary':
            if self.dict_mode == 'global':
                params.append(self.dictionary_mu)
            else:
                params.append(self.dictionary_mu_step)
        return params

    def set_planner_temperature(self, tau):
        """Update planner sampling temperatures."""
        tau = float(tau)
        self.gumbel_sigmoid.tau = tau
        self.dict_temperature = tau

    def expand_group_gates_to_feat_mask(self, group_gates):
        """
        Expand group-level gates to feature-level mask.

        Args:
            group_gates: (B, T * num_groups)
        Returns:
            feat_mask: (B, T * num_feat)
        """
        bsz = group_gates.size(0)
        g = group_gates.reshape(bsz, self.num_time, self.num_groups)
        f = torch.matmul(g, self.group_to_feat)
        return f.reshape(bsz, self.num_time * self.num_feat)

    def feat_mask_to_group_mask(self, feat_mask):
        """
        Collapse feature-level mask to group-level.
        A group is acquired if ANY feature in it is acquired.

        Args:
            feat_mask: (B, T * num_feat)
        Returns:
            group_mask: (B, T * num_groups)
        """
        bsz = feat_mask.size(0)
        f = feat_mask.reshape(bsz, self.num_time, self.num_feat)
        g = torch.matmul(f, self.group_to_feat.T)
        g = (g > 0).float()
        return g.reshape(bsz, self.num_time * self.num_groups)

    def get_aux_gates(self, batch_size, mask_static=None):
        """Apply Gumbel-Sigmoid to aux logits (baseline feature gate)."""
        logits = self.aux_logits.unsqueeze(0).expand(batch_size, -1)
        gates = self.gumbel_sigmoid(logits, hard=True)
        if mask_static is not None:
            gates = gates * mask_static
        return gates

    def after_cur_t_mask(self, cur_t, t_steps, width, device):
        """Create mask for features at timesteps >= cur_t."""
        bsz = cur_t.size(0)
        t = torch.arange(t_steps, device=device).view(1, t_steps)
        allowed_t = t >= cur_t.view(bsz, 1)
        allowed = allowed_t.unsqueeze(-1).expand(bsz, t_steps, width)
        return allowed.reshape(bsz, t_steps * width)

    def predict_with_mask(self, x, mask, aux_acquired=None):
        """Make predictions using the predictor (feature-level mask)."""
        preds = []
        for t in range(self.num_time):
            m_t = mask.clone()
            m_t[:, (t + 1) * self.num_feat:] = 0

            x_t_masked = self.mask_layer(x, m_t)

            t_indicator = torch.full(
                (x.size(0),),
                (t + 1) / self.num_time,
                device=x.device
            ).unsqueeze(1)

            if aux_acquired is not None:
                x_t_masked = torch.cat((t_indicator, x_t_masked, aux_acquired), dim=1)
            else:
                x_t_masked = torch.cat((t_indicator, x_t_masked), dim=1)
            preds.append(self.predictor(x_t_masked))

        return torch.stack(preds, dim=1)

    def _st_hard(self, soft, dim=-1):
        """Straight-through hardening helper."""
        idx = torch.argmax(soft, dim=dim, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(dim, idx, 1.0)
        return hard + soft - soft.detach()

    def _sample_template_weights(self, logits, hard=True, dim=-1):
        """Sample/compute differentiable template-selection weights."""
        if self.training:
            soft = F.gumbel_softmax(logits, tau=self.dict_temperature, hard=False, dim=dim)
        else:
            soft = F.softmax(logits / max(self.dict_temperature, 1e-8), dim=dim)

        if hard and self.dict_use_st:
            return self._st_hard(soft, dim=dim)
        # Non-ST mode keeps relaxed weights so gradients can flow;
        # the final gate thresholding still provides a hard forward mask.
        return soft

    def planner_forward(self, planner_input, valid_mask_g=None, hard=True):
        """
        Forward planner pass returning group-level gates.

        Args:
            planner_input: (B, planner_input_dim)
            valid_mask_g: (B, T*num_groups) or None
            hard: whether to harden samples via ST behavior

        Returns:
            planner_logits: raw planner output logits
            z_groups: planned group-level mask (B, T*num_groups)
        """
        planner_logits = self.planner_nn(planner_input)

        if self.planner_mode != 'dictionary':
            logits = planner_logits
            if valid_mask_g is not None:
                logits = logits.masked_fill(valid_mask_g == 0, float('-inf'))
            z_groups = self.gumbel_sigmoid(logits, hard=hard)
            return planner_logits, z_groups

        if self.dict_mode == 'global':
            z_templates = self._sample_template_weights(planner_logits, hard=hard, dim=-1)
            relaxed_templates = torch.sigmoid(self.dictionary_mu)
            z_groups = torch.einsum('bk,ktg->btg', z_templates, relaxed_templates)
            z_groups = z_groups.reshape(-1, self.num_time * self.num_groups)
        else:
            logits = planner_logits.reshape(-1, self.num_time, self.dict_num_templates)
            z_templates = self._sample_template_weights(logits, hard=hard, dim=-1)
            relaxed_templates = torch.sigmoid(self.dictionary_mu_step)
            z_groups = torch.einsum('btk,kg->btg', z_templates, relaxed_templates)
            z_groups = z_groups.reshape(-1, self.num_time * self.num_groups)

        if valid_mask_g is not None:
            z_groups = z_groups * valid_mask_g.float()

        if hard:
            z_hard = (z_groups > self.threshold).float()
            z_groups = z_hard + z_groups - z_groups.detach()

        return planner_logits, z_groups

    def dictionary_diversity_loss(self):
        """Pairwise cosine-similarity penalty across templates."""
        if self.planner_mode != 'dictionary':
            return torch.tensor(0.0, device=self.device)

        if self.dict_mode == 'global':
            templates = torch.sigmoid(self.dictionary_mu).reshape(self.dict_num_templates, -1)
        else:
            templates = torch.sigmoid(self.dictionary_mu_step).reshape(self.dict_num_templates, -1)

        normed = F.normalize(templates, p=2, dim=1)
        sim = torch.matmul(normed, normed.T)
        return sim.sum() - torch.diag(sim).sum()

    def dictionary_sparsity_loss(self):
        """L1-like sparsity penalty on relaxed dictionary entries."""
        if self.planner_mode != 'dictionary':
            return torch.tensor(0.0, device=self.device)

        if self.dict_mode == 'global':
            return torch.sigmoid(self.dictionary_mu).sum()
        return torch.sigmoid(self.dictionary_mu_step).sum()

    def dictionary_regularization_loss(self):
        """Combined dictionary regularizer (diversity + sparsity)."""
        if self.planner_mode != 'dictionary':
            return torch.tensor(0.0, device=self.device)
        div = self.dict_div_lambda * self.dictionary_diversity_loss()
        sparse = self.dict_sparse_lambda * self.dictionary_sparsity_loss()
        return div + sparse

    def initialize_dictionary(self, init_mode='random', masks_np=None,
                              eps=1e-4, seed=42, n_init=20):
        """Initialize dictionary parameters."""
        if self.planner_mode != 'dictionary':
            return

        init_mode = str(init_mode).lower()

        if init_mode == 'random':
            with torch.no_grad():
                if self.dict_mode == 'global':
                    nn.init.normal_(self.dictionary_mu, mean=0.0, std=0.02)
                else:
                    nn.init.normal_(self.dictionary_mu_step, mean=0.0, std=0.02)
            return

        if init_mode == 'orthogonal':
            with torch.no_grad():
                if self.dict_mode == 'global':
                    flat_dim = self.num_time * self.num_groups
                    if self.dict_num_templates <= flat_dim:
                        w = torch.empty(self.dict_num_templates, flat_dim, device=self.device)
                        nn.init.orthogonal_(w)
                    else:
                        w_t = torch.empty(flat_dim, self.dict_num_templates, device=self.device)
                        nn.init.orthogonal_(w_t)
                        w = w_t.T
                    self.dictionary_mu.copy_(w.reshape(self.dict_num_templates, self.num_time, self.num_groups))
                else:
                    flat_dim = self.num_groups
                    if self.dict_num_templates <= flat_dim:
                        w = torch.empty(self.dict_num_templates, flat_dim, device=self.device)
                        nn.init.orthogonal_(w)
                    else:
                        w_t = torch.empty(flat_dim, self.dict_num_templates, device=self.device)
                        nn.init.orthogonal_(w_t)
                        w = w_t.T
                    self.dictionary_mu_step.copy_(w)
            return

        if init_mode != 'kmeans':
            raise ValueError(f"Unsupported dictionary init mode: {init_mode}")

        if masks_np is None:
            raise ValueError("K-Means dictionary init requires masks_np")

        try:
            from sklearn.cluster import KMeans
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"K-Means init requires scikit-learn: {exc}")

        masks_np = np.asarray(masks_np, dtype=np.float32)
        if masks_np.ndim == 3:
            masks_np = masks_np.reshape(masks_np.shape[0], -1)

        expected_dim = self.num_time * self.num_groups
        if masks_np.ndim != 2 or masks_np.shape[1] != expected_dim:
            raise ValueError(
                f"Expected masks with shape (N, {expected_dim}), got {masks_np.shape}"
            )

        if masks_np.shape[0] < self.dict_num_templates:
            raise ValueError(
                f"Not enough masks for k={self.dict_num_templates}: N={masks_np.shape[0]}"
            )

        if self.dict_mode == 'global':
            km = KMeans(n_clusters=self.dict_num_templates, random_state=int(seed), n_init=int(n_init))
            km.fit(masks_np)
            centroids = km.cluster_centers_.reshape(self.dict_num_templates, self.num_time, self.num_groups)
        else:
            steps = masks_np.reshape(-1, self.num_time, self.num_groups)
            step_samples = steps.reshape(-1, self.num_groups)
            if step_samples.shape[0] < self.dict_num_templates:
                raise ValueError(
                    "Not enough step-wise masks for timestep dictionary init: "
                    f"N={step_samples.shape[0]}, k={self.dict_num_templates}"
                )
            km = KMeans(n_clusters=self.dict_num_templates, random_state=int(seed), n_init=int(n_init))
            km.fit(step_samples)
            centroids = km.cluster_centers_.reshape(self.dict_num_templates, self.num_groups)

        c_clip = np.clip(centroids, float(eps), 1.0 - float(eps))
        mu_init = np.log(c_clip / (1.0 - c_clip))
        if not np.isfinite(mu_init).all():
            raise ValueError("Non-finite values encountered in logit(K-Means centroid) init")

        with torch.no_grad():
            if self.dict_mode == 'global':
                self.dictionary_mu.copy_(
                    torch.tensor(mu_init, dtype=self.dictionary_mu.dtype, device=self.dictionary_mu.device)
                )
            else:
                self.dictionary_mu_step.copy_(
                    torch.tensor(mu_init, dtype=self.dictionary_mu_step.dtype, device=self.dictionary_mu_step.device)
                )

    def training_step(self, batch, batch_idx):
        """Training step."""
        opt = self.optimizers()

        # batch: tuple size 5 (no aux) or size 7 (with aux)
        if len(batch) == 7:
            x, y, mask, time_emb, cur_t, x_static, mask_static = batch
            x_static = x_static.to(self.device)
            mask_static = mask_static.to(self.device)
        else:
            x, y, mask, time_emb, cur_t = batch
            x_static = None
            mask_static = None

        # Reshape
        mask = mask.reshape(-1, self.num_time * self.num_feat)
        x = x.reshape(-1, self.num_time * self.num_feat)
        y = y.reshape(-1, self.num_time)

        x = x.to(self.device)
        mask = mask.to(self.device)
        cur_t = cur_t.to(self.device)
        time_emb = time_emb.to(self.device)
        y = y.to(self.device)

        bsz = x.size(0)
        orig_mask_f = mask.float()

        # Convert feature-level oracle mask to group-level
        orig_mask_g = self.feat_mask_to_group_mask(orig_mask_f)

        # Auxiliary gate
        aux_acquired = None
        aux_gates = None
        aux_cost_loss = torch.tensor(0.0, device=self.device)
        if self.num_aux > 0 and x_static is not None:
            aux_gates = self.get_aux_gates(bsz, mask_static)
            aux_acquired = x_static * aux_gates
            aux_cost_loss = self.aux_cost_weight * (aux_gates * self.aux_feature_costs).sum() / bsz

        # Group-level availability mask
        mask_after_g = self.after_cur_t_mask(
            cur_t, self.num_time, self.num_groups, self.device
        ).float()

        # Forward through planner (input is feature-level)
        x_masked = self.mask_layer(x, mask)
        if aux_acquired is not None:
            planner_input = torch.cat([x_masked, mask, aux_acquired, aux_gates, time_emb], dim=1)
        else:
            planner_input = torch.cat([x_masked, mask, time_emb], dim=1)
        _, z_groups = self.planner_forward(planner_input, hard=True)

        # Group acquisitions at allowed positions
        z_safe_g = z_groups * mask_after_g * (1.0 - orig_mask_g.detach())
        gated_groups = (orig_mask_g.detach() + z_safe_g).clamp(0.0, 1.0)

        # Expand to feature level for prediction
        gated_mask_feat = self.expand_group_gates_to_feat_mask(gated_groups).clamp(0.0, 1.0)

        # Predict with feature-level mask
        y_hat = self.predict_with_mask(x, gated_mask_feat, aux_acquired=aux_acquired)
        _, t_steps, _ = y_hat.shape

        # Select labels for timesteps >= cur_t
        t_idx = torch.arange(t_steps, device=y_hat.device).view(1, t_steps)
        time_ok = (t_idx >= cur_t.view(bsz, 1))
        label_ok = (y != -1)
        use_for_ce = time_ok & label_ok

        y_hat_sel = y_hat[use_for_ce]
        y_sel = y[use_for_ce]

        ce_loss = F.cross_entropy(y_hat_sel, y_sel.long())

        # Cost at group level (weighted by per-group costs)
        cost = (gated_groups * mask_after_g * self.feature_costs_flat).sum() / bsz
        cost_loss = self.cost_weight * cost
        dict_reg = self.dictionary_regularization_loss()

        loss = ce_loss + cost_loss + aux_cost_loss + dict_reg

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/ce_loss', ce_loss, prog_bar=True)
        self.log('train/cost_loss', cost_loss, prog_bar=True)
        self.log('train/cost', cost, prog_bar=True)
        if self.planner_mode == 'dictionary':
            self.log('train/dict_reg', dict_reg, prog_bar=True)
        if self.num_aux > 0 and aux_gates is not None:
            self.log('train/aux_cost', aux_gates.sum() / bsz, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if len(batch) == 7:
            x, y, mask, time_emb, cur_t, x_static, mask_static = batch
            x_static = x_static.to(self.device)
            mask_static = mask_static.to(self.device)
        else:
            x, y, mask, time_emb, cur_t = batch
            x_static = None
            mask_static = None

        mask = mask.reshape(-1, self.num_time * self.num_feat)
        x = x.reshape(-1, self.num_time * self.num_feat)
        y = y.reshape(-1, self.num_time)

        x = x.to(self.device)
        mask = mask.to(self.device)
        cur_t = cur_t.to(self.device)
        time_emb = time_emb.to(self.device)
        y = y.to(self.device)

        bsz = x.size(0)
        orig_mask_f = mask.float()
        orig_mask_g = self.feat_mask_to_group_mask(orig_mask_f)

        # Auxiliary gate
        aux_acquired = None
        aux_gates = None
        aux_cost_loss = torch.tensor(0.0, device=self.device)
        if self.num_aux > 0 and x_static is not None:
            aux_gates = self.get_aux_gates(bsz, mask_static)
            aux_acquired = x_static * aux_gates
            aux_cost_loss = self.aux_cost_weight * (aux_gates * self.aux_feature_costs).sum() / bsz

        mask_after_g = self.after_cur_t_mask(
            cur_t, self.num_time, self.num_groups, self.device
        ).float()

        with torch.no_grad():
            x_masked = self.mask_layer(x, mask)
            if aux_acquired is not None:
                planner_input = torch.cat([x_masked, mask, aux_acquired, aux_gates, time_emb], dim=1)
            else:
                planner_input = torch.cat([x_masked, mask, time_emb], dim=1)
            _, z_groups = self.planner_forward(planner_input, hard=True)

            z_safe_g = z_groups * mask_after_g * (1.0 - orig_mask_g.detach())
            gated_groups = (orig_mask_g.detach() + z_safe_g).clamp(0.0, 1.0)

            gated_mask_feat = self.expand_group_gates_to_feat_mask(gated_groups).clamp(0.0, 1.0)

            y_hat = self.predict_with_mask(x, gated_mask_feat, aux_acquired=aux_acquired)
            _, t_steps, _ = y_hat.shape

            t_idx = torch.arange(t_steps, device=y_hat.device).view(1, t_steps)
            time_ok = (t_idx >= cur_t.view(bsz, 1))
            label_ok = (y != -1)
            use_for_ce = time_ok & label_ok

            y_hat_sel = y_hat[use_for_ce]
            y_sel = y[use_for_ce]

            if len(y_sel) == 0:
                return None

            ce_loss = F.cross_entropy(y_hat_sel, y_sel.long())
            cost = (gated_groups * mask_after_g * self.feature_costs_flat).sum() / bsz
            cost_loss = self.cost_weight * cost
            dict_reg = self.dictionary_regularization_loss()
            loss = ce_loss + cost_loss + aux_cost_loss + dict_reg

        # Log validation metrics
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/ce_loss', ce_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/cost_loss', cost_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/cost', cost, prog_bar=True, on_step=False, on_epoch=True)
        if self.planner_mode == 'dictionary':
            self.log('val/dict_reg', dict_reg, prog_bar=True, on_step=False, on_epoch=True)

        # Accuracy
        y_pred = torch.argmax(y_hat_sel, dim=-1)
        accuracy = (y_pred == y_sel).float().mean()
        self.log('val/accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True)
