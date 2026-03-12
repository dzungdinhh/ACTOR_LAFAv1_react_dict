"""
Iterative Actor Training — Joint (classifier + planner)
step1: Train on oracle data for `warmup_batches` steps
step2: Rollout actor, mix oracle states, train

Unlike train_actor_iterative.py the classifier is *unfrozen* and
fine-tuned jointly with the planner. Three safeguards prevent collapse:

  1. Lower learning rate for the classifier (classifier_lr).
  2. L2 anchor regularisation that penalises drift from the pre-trained
     classifier weights.
  3. Random mask augmentation on the classifier input so it keeps seeing
     diverse acquisition patterns.
"""
import argparse
import json
import os
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    CLASSIFIER_PATH,
    DATASET,
    DATA_FOLDER,
    ITERATIVE_ACTOR_CONFIG,
    MASK_TYPE,
    NUM_FEAT,
    NUM_GROUPS,
    ORACLE_CONFIG,
    make_actor_path,
    make_oracle_path,
)
from dataset import (
    load_ILIADD_data,
    load_adni_data,
    load_cheears_data,
    load_klg_data,
    load_oracle_rollout,
    load_synthetic_data,
    load_womac_data,
)
from evaluate import evaluate_actor, save_results_to_csv
from gumbel_actor import GumbelActor
from models import Predictor
from utils import build_group_to_feat_matrix, generate_uniform_mask, get_timestep_embedding, set_seed


JOINT_DEFAULTS = {
    'classifier_lr': 1e-4,
    'anchor_weight': 1.0,
    'mask_aug_ratio': 0.5,
}


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _load_dataset(dataset_name, data_path):
    if dataset_name in ('cheears', 'cheears_demog', 'cheears_day_context'):
        return load_cheears_data(data_path)
    if dataset_name == 'klg':
        return load_klg_data(data_path)
    if dataset_name == 'womac':
        return load_womac_data(data_path)
    if dataset_name == 'ILIADD':
        return load_ILIADD_data(data_path)
    if dataset_name == 'adni':
        return load_adni_data(data_path)
    return load_synthetic_data(data_path)


def _git_commit_hash(workdir):
    try:
        out = subprocess.check_output(
            ['git', '-C', workdir, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
        )
        return out.decode('utf-8').strip()
    except Exception:
        return ''


def _append_manifest(manifest_path, payload):
    if not manifest_path:
        return
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, sort_keys=True) + '\n')


def rollout_batch(actor, x, y, m_avail, device, num_time, num_feat,
                  x_static=None, mask_static=None):
    """
    Rollout the actor on a single batch of raw data.
    Collects the state BEFORE each action as a training sample.
    """
    actor.eval()
    bsz = x.shape[0]
    ng = actor.num_groups

    x_flat = x.reshape(bsz, -1) if x.dim() == 3 else x
    m_avail_flat = m_avail.reshape(bsz, -1) if m_avail.dim() == 3 else m_avail
    y_flat = y if y.dim() == 2 else y.reshape(bsz, -1)

    sx, sy, sm, st = [], [], [], []
    s_xstatic, s_mstatic = [], []

    m_curr_groups = torch.zeros(bsz, num_time * ng, dtype=torch.float32, device=device)
    m_curr_feat = torch.zeros_like(x_flat, dtype=torch.float32)
    cur_t = torch.zeros(bsz, dtype=torch.int, device=device)
    m_done = torch.zeros(bsz, dtype=torch.bool, device=device)

    m_avail_groups = actor.feat_mask_to_group_mask(m_avail_flat.float())

    aux_acquired = None
    aux_gates = None
    if actor.num_aux > 0 and x_static is not None:
        aux_gates = actor.get_aux_gates(bsz, mask_static)
        aux_acquired = x_static * aux_gates

    with torch.no_grad():
        for _ in range(2 * num_time):
            active = ~m_done
            if active.any():
                sx.append(x_flat[active].cpu().numpy())
                sy.append(y_flat[active].cpu().numpy())
                sm.append(m_curr_feat[active].cpu().numpy())
                st.append(cur_t[active].cpu().numpy())
                if x_static is not None:
                    s_xstatic.append(x_static[active].cpu().numpy())
                    s_mstatic.append(mask_static[active].cpu().numpy())

            t_grid = torch.arange(num_time, device=device).unsqueeze(0).expand(bsz, -1)
            time_ok = t_grid >= cur_t.unsqueeze(1)
            time_flat_g = time_ok.unsqueeze(-1).expand(-1, -1, ng).reshape(bsz, -1)
            valid_g = (m_avail_groups > 0) & (m_curr_groups == 0) & time_flat_g

            if ((valid_g.sum(1) == 0) | m_done).all():
                break

            time_emb = get_timestep_embedding(cur_t, embedding_dim=actor.time_emb_dim)
            x_masked = actor.mask_layer(x_flat, m_curr_feat)

            if aux_acquired is not None:
                inp = torch.cat([x_masked, m_curr_feat, aux_acquired, aux_gates, time_emb], dim=1)
            else:
                inp = torch.cat([x_masked, m_curr_feat, time_emb], dim=1)

            _, z = actor.planner_forward(inp, valid_mask_g=valid_g, hard=True)

            cur_t_group_mask = torch.zeros_like(m_curr_groups)
            for b in range(bsz):
                if not m_done[b]:
                    s, e = cur_t[b] * ng, (cur_t[b] + 1) * ng
                    cur_t_group_mask[b, s:e] = z[b, s:e]

            m_curr_groups = (m_curr_groups + cur_t_group_mask).clamp(0, 1)
            m_curr_feat = actor.expand_group_gates_to_feat_mask(m_curr_groups).clamp(0, 1)

            added = cur_t_group_mask.sum(dim=1)
            for b in range(bsz):
                if added[b] > 0 and not m_done[b]:
                    cur_t[b] = min(cur_t[b] + 1, num_time)
            m_done = m_done | (added == 0)

    result = (
        np.concatenate(sx), np.concatenate(sy),
        np.concatenate(sm), np.concatenate(st),
    )
    if s_xstatic:
        result = result + (np.concatenate(s_xstatic), np.concatenate(s_mstatic))
    return result


def _forward_loss(actor, x, y, mask, cur_t, device,
                  anchor_params, anchor_weight,
                  mask_aug_ratio=0.0,
                  x_static=None, mask_static=None):
    """Forward pass that computes the joint loss (planner + classifier)."""
    t_steps = actor.num_time
    d = actor.num_feat
    ng = actor.num_groups

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1, t_steps * d)
    y_t = torch.as_tensor(y, dtype=torch.int64, device=device).reshape(-1, t_steps)
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).reshape(-1, t_steps * d)
    cur_t_t = torch.as_tensor(cur_t, dtype=torch.int64, device=device)
    time_emb = get_timestep_embedding(cur_t_t, embedding_dim=actor.time_emb_dim)

    bsz = x_t.size(0)
    orig_mask_f = mask_t.float()
    orig_mask_g = actor.feat_mask_to_group_mask(orig_mask_f)
    mask_after_g = actor.after_cur_t_mask(cur_t_t, t_steps, ng, device).float()

    # aux gates
    aux_acquired = None
    aux_gates = None
    aux_cost_loss = torch.tensor(0.0, device=device)
    if actor.num_aux > 0 and x_static is not None:
        xs_t = torch.as_tensor(x_static, dtype=torch.float32, device=device)
        ms_t = torch.as_tensor(mask_static, dtype=torch.float32, device=device)
        aux_gates = actor.get_aux_gates(bsz, ms_t)
        aux_acquired = xs_t * aux_gates
        aux_cost_loss = actor.aux_cost_weight * (aux_gates * actor.aux_feature_costs).sum() / bsz

    # planner forward
    x_masked = actor.mask_layer(x_t, mask_t)
    if aux_acquired is not None:
        inp = torch.cat([x_masked, mask_t, aux_acquired, aux_gates, time_emb], dim=1)
    else:
        inp = torch.cat([x_masked, mask_t, time_emb], dim=1)

    _, z_groups = actor.planner_forward(inp, hard=True)

    z_safe_g = z_groups * mask_after_g * (1.0 - orig_mask_g.detach())
    gated_groups = (orig_mask_g.detach() + z_safe_g).clamp(0, 1)

    gated_mask_feat = actor.expand_group_gates_to_feat_mask(gated_groups).clamp(0, 1)

    # random mask augmentation for classifier robustness
    if actor.training and mask_aug_ratio > 0:
        rand_mask = generate_uniform_mask(bsz, t_steps * d).to(device)
        aug_selector = (torch.rand(bsz, 1, device=device) < mask_aug_ratio).float()
        rand_mask = rand_mask * mask_t
        pred_mask = aug_selector * rand_mask + (1.0 - aug_selector) * gated_mask_feat
    else:
        pred_mask = gated_mask_feat

    y_hat = actor.predict_with_mask(x_t, pred_mask, aux_acquired=aux_acquired)
    _, tp, _ = y_hat.shape

    t_idx = torch.arange(tp, device=device).view(1, tp)
    use = (t_idx >= cur_t_t.view(bsz, 1)) & (y_t != -1)

    ce_loss = F.cross_entropy(y_hat[use], y_t[use].long())

    # cost at group level
    cost = (gated_groups * mask_after_g * actor.feature_costs_flat).sum() / bsz
    cost_loss = actor.cost_weight * cost

    # L2 anchor loss
    anchor_loss = torch.tensor(0.0, device=device)
    if anchor_weight > 0:
        for name, param in actor.predictor.named_parameters():
            anchor_loss = anchor_loss + F.mse_loss(param, anchor_params[name])
    anchor_loss = anchor_weight * anchor_loss

    dict_reg_loss = actor.dictionary_regularization_loss()

    loss = ce_loss + cost_loss + aux_cost_loss + anchor_loss + dict_reg_loss

    return loss, ce_loss, cost, anchor_loss, dict_reg_loss


def train_step(actor, optimizer, x, y, mask, cur_t, device,
               anchor_params, anchor_weight, mask_aug_ratio,
               x_static=None, mask_static=None):
    """1 gradient update (planner + classifier)."""
    actor.train()
    loss, ce_loss, cost, anchor_loss, dict_reg_loss = _forward_loss(
        actor, x, y, mask, cur_t, device,
        anchor_params=anchor_params,
        anchor_weight=anchor_weight,
        mask_aug_ratio=mask_aug_ratio,
        x_static=x_static, mask_static=mask_static,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'ce_loss': ce_loss.item(),
        'cost': cost.item(),
        'anchor': anchor_loss.item(),
        'dict_reg': dict_reg_loss.item(),
    }


def sample_oracle(x_orc, y_orc, m_orc, t_orc, n,
                  xs_orc=None, ms_orc=None):
    idx = np.random.choice(len(x_orc), size=min(n, len(x_orc)), replace=False)
    result = (x_orc[idx], y_orc[idx], m_orc[idx], t_orc[idx])
    if xs_orc is not None:
        result = result + (xs_orc[idx], ms_orc[idx])
    return result


def mix_data(rollout_data, oracle_data, oracle_ratio):
    n_rollout = len(rollout_data[0])
    n_oracle = max(1, int(n_rollout * oracle_ratio / (1.0 - oracle_ratio)))

    n_arrays = len(rollout_data)
    idx = np.random.choice(len(oracle_data[0]),
                           size=min(n_oracle, len(oracle_data[0])),
                           replace=False)
    return tuple(
        np.concatenate([rollout_data[i], oracle_data[i][idx]])
        for i in range(n_arrays)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost_weight', type=float, default=None)
    parser.add_argument('--aux_cost_weight', type=float, default=None)
    parser.add_argument('--classifier_lr', type=float, default=None,
                        help='Learning rate for the classifier (default: 1e-4)')
    parser.add_argument('--anchor_weight', type=float, default=None,
                        help='L2 anchor regularisation strength (default: 1.0)')
    parser.add_argument('--mask_aug_ratio', type=float, default=None,
                        help='Fraction of samples that get random mask augmentation (default: 0.5)')
    parser.add_argument('--baseline', type=str, default='learned',
                        choices=['learned', 'all', 'none'],
                        help='Baseline feature mode: learned (default), all, or none')
    parser.add_argument('--warmup_batches', type=int, default=None,
                        help='Number of oracle warmup batches (overrides config, 0=no warmup)')
    parser.add_argument('--total_batches', type=int, default=None,
                        help='Total training batches (overrides config)')
    parser.add_argument('--save_suffix', type=str, default=None,
                        help='Extra suffix appended to checkpoint filename')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='CSV path for results (default: LAFA_ACTORS - all.csv)')
    parser.add_argument('--method_suffix', type=str, default=None,
                        help='Suffix appended to method name in CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_folder', type=str, default=None,
                        help='Override dataset split root (train/val/test .npz files)')
    parser.add_argument('--manifest_path', type=str, default=None,
                        help='JSONL path to append run metadata')

    # Planner dictionary args
    parser.add_argument('--planner_mode', type=str, default=None, choices=['direct', 'dictionary'])
    parser.add_argument('--dict_mode', type=str, default=None, choices=['global', 'timestep'])
    parser.add_argument('--dict_num_templates', type=int, default=None)
    parser.add_argument('--dict_use_st', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--dict_tau0', type=float, default=None)
    parser.add_argument('--dict_tau_min', type=float, default=None)
    parser.add_argument('--dict_tau_gamma', type=float, default=None)
    parser.add_argument('--dict_div_lambda', type=float, default=None)
    parser.add_argument('--dict_sparse_lambda', type=float, default=None)
    parser.add_argument('--dict_init', type=str, default=None, choices=['random', 'orthogonal', 'kmeans'])
    parser.add_argument('--dict_init_masks_path', type=str, default=None)
    parser.add_argument('--dict_init_eps', type=float, default=None)
    parser.add_argument('--dict_kmeans_seed', type=int, default=None)
    parser.add_argument('--dict_kmeans_n_init', type=int, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    config = {**ITERATIVE_ACTOR_CONFIG}
    if args.cost_weight is not None:
        config['cost_weight'] = args.cost_weight

    # Enforce cw == acw everywhere.
    if args.aux_cost_weight is not None and args.aux_cost_weight != config['cost_weight']:
        raise ValueError(
            f"Expected cw == acw, got cost_weight={config['cost_weight']} "
            f"and aux_cost_weight={args.aux_cost_weight}"
        )
    config['aux_cost_weight'] = config['cost_weight']

    if args.warmup_batches is not None:
        config['warmup_batches'] = args.warmup_batches
    if args.total_batches is not None:
        config['total_batches'] = args.total_batches

    # Planner overrides
    for key in (
        'planner_mode', 'dict_mode', 'dict_num_templates', 'dict_use_st',
        'dict_tau0', 'dict_tau_min', 'dict_tau_gamma',
        'dict_div_lambda', 'dict_sparse_lambda',
        'dict_init', 'dict_init_masks_path', 'dict_init_eps',
        'dict_kmeans_seed', 'dict_kmeans_n_init',
    ):
        val = getattr(args, key)
        if val is not None:
            config[key] = val

    # Joint-specific hyper-parameters
    classifier_lr = args.classifier_lr or config.get('classifier_lr', JOINT_DEFAULTS['classifier_lr'])
    anchor_weight = args.anchor_weight if args.anchor_weight is not None else config.get('anchor_weight', JOINT_DEFAULTS['anchor_weight'])
    mask_aug_ratio = args.mask_aug_ratio if args.mask_aug_ratio is not None else config.get('mask_aug_ratio', JOINT_DEFAULTS['mask_aug_ratio'])

    data_folder = args.data_folder if args.data_folder else DATA_FOLDER

    # Paths
    oracle_rollout_path = make_oracle_path(config['cost_weight'])
    actor_save_path = make_actor_path(config['cost_weight'], config.get('aux_cost_weight'),
                                      joint=True, baseline=args.baseline)
    if args.save_suffix:
        actor_save_path = actor_save_path.replace('.ckpt', f'_{args.save_suffix}.ckpt')

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_PATH}. "
            "Please run train_classifier.py first."
        )
    if not os.path.exists(oracle_rollout_path):
        raise FileNotFoundError(
            f"Oracle rollout not found at {oracle_rollout_path}. "
            "Please run generate_oracle.py first."
        )

    # Load classifier (will be UNFROZEN)
    print("Loading pre-trained classifier (will be fine-tuned jointly)...")
    cls_ckpt = torch.load(CLASSIFIER_PATH, map_location='cpu')
    num_time = cls_ckpt['num_time']
    num_feat = cls_ckpt['num_feat']
    num_aux = cls_ckpt.get('num_aux', 0)

    predictor = Predictor(
        d_in=num_time * num_feat + num_aux,
        d_out=cls_ckpt['y_dim'],
        hidden=cls_ckpt['config']['hidden_dim'],
        dropout=cls_ckpt['config']['dropout'],
    )
    predictor.load_state_dict(cls_ckpt['predictor'])

    # group config
    group_to_feat_matrix = None
    if NUM_GROUPS != NUM_FEAT:
        group_to_feat_matrix = build_group_to_feat_matrix(num_feat)
        print(f"Group-based acquisition: {NUM_GROUPS} groups -> {num_feat} features")

    # oracle rollout data
    print("Loading oracle rollout...")
    oracle_data = load_oracle_rollout(
        oracle_rollout_path, num_time=num_time, num_feat=num_feat,
    )
    x_orc, y_orc, m_orc, t_orc = oracle_data[:4]
    xs_orc = oracle_data[4]
    ms_orc = oracle_data[5]
    print(f"Oracle: {len(x_orc)} states")

    # raw training data
    train_data_path = os.path.join(data_folder, 'train_data.npz')
    train_dataset = _load_dataset(DATASET, train_data_path)
    raw_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"Train samples: {len(train_dataset)}")

    # validation and test data
    val_data_path = os.path.join(data_folder, 'val_data.npz')
    test_data_path = os.path.join(data_folder, 'test_data.npz')
    test_dataset = _load_dataset(DATASET, test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=config['val_batch_size'], shuffle=False)

    device = _get_device()
    print(f"Device: {device}")

    # Build actor
    actor = GumbelActor(
        predictor=predictor, num_time=num_time, num_feat=num_feat,
        config=config, num_aux=num_aux,
        num_groups=NUM_GROUPS, group_to_feat_matrix=group_to_feat_matrix,
    ).to(device)

    # Unfreeze classifier
    for p in actor.predictor.parameters():
        p.requires_grad = True

    # Keep frozen copy of pre-trained weights for anchor loss
    anchor_params = {
        name: param.clone().detach().to(device)
        for name, param in actor.predictor.named_parameters()
    }

    # Override baseline feature gates if requested
    if args.baseline == 'all' and num_aux > 0:
        actor.aux_logits.data.fill_(100.0)
        actor.aux_logits.requires_grad = False
        print("Forcing ALL baseline features (fixed, not learned)")
    elif args.baseline == 'none' and num_aux > 0:
        actor.aux_logits.data.fill_(-100.0)
        actor.aux_logits.requires_grad = False
        print("Forcing NO baseline features (fixed, not learned)")

    planner_params = actor.planner_parameters()
    if num_aux > 0 and args.baseline == 'learned':
        planner_params.append(actor.aux_logits)

    optimizer = torch.optim.Adam([
        {'params': planner_params, 'lr': config['lr']},
        {'params': list(actor.predictor.parameters()), 'lr': classifier_lr},
    ])

    total_batches = config.get('total_batches', 1000)
    warmup_batches = config.get('warmup_batches', 50)
    oracle_mix = config.get('oracle_mix_ratio', 0.3)
    log_every = config.get('log_every', 10)
    bs = config['batch_size']

    has_aux = num_aux > 0 and xs_orc is not None

    print(
        f"\nConfig: total={total_batches} warmup={warmup_batches} mix={oracle_mix} "
        f"cw={config['cost_weight']} acw={config['aux_cost_weight']} "
        f"num_aux={num_aux} num_groups={NUM_GROUPS} planner_mode={actor.planner_mode}"
    )
    print(
        f"Joint: classifier_lr={classifier_lr} anchor_weight={anchor_weight} "
        f"mask_aug={mask_aug_ratio} seed={args.seed}"
    )

    _append_manifest(args.manifest_path, {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'event': 'run_start',
        'dataset': DATASET,
        'data_folder': data_folder,
        'split_paths': {
            'train': train_data_path,
            'val': val_data_path,
            'test': test_data_path,
        },
        'planner_mode': actor.planner_mode,
        'dict_mode': actor.dict_mode if actor.planner_mode == 'dictionary' else '',
        'dict_num_templates': int(actor.dict_num_templates) if actor.planner_mode == 'dictionary' else 0,
        'baseline': args.baseline,
        'cw': float(config['cost_weight']),
        'acw': float(config['aux_cost_weight']),
        'seed': int(args.seed),
        'actor_save_path': actor_save_path,
        'csv_path': args.csv_path,
        'git_commit': _git_commit_hash(os.path.dirname(__file__)),
    })

    # Global step across entire self-iterative loop
    global_train_step = 0

    def _set_global_tau(step_idx):
        tau0 = float(config.get('dict_tau0', config.get('gate_tau', 1.0)))
        tau_min = float(config.get('dict_tau_min', tau0))
        gamma = float(config.get('dict_tau_gamma', 1.0))
        tau = max(tau_min, tau0 * (gamma ** step_idx))
        actor.set_planner_temperature(tau)
        return tau

    # Oracle warm-up
    print(f"\nOracle warm-up ({warmup_batches} batches)...")
    for batch_idx in range(1, warmup_batches + 1):
        if has_aux:
            ox, oy, om, ot, oxs, oms = sample_oracle(
                x_orc, y_orc, m_orc, t_orc, bs, xs_orc, ms_orc,
            )
        else:
            ox, oy, om, ot = sample_oracle(x_orc, y_orc, m_orc, t_orc, bs)
            oxs, oms = None, None

        tau = _set_global_tau(global_train_step)
        metrics = train_step(
            actor, optimizer, ox, oy, om, ot, device,
            anchor_params=anchor_params,
            anchor_weight=anchor_weight,
            mask_aug_ratio=mask_aug_ratio,
            x_static=oxs, mask_static=oms,
        )
        global_train_step += 1

        if batch_idx % log_every == 0:
            print(
                f"Batch {batch_idx:4d} (oracle): loss={metrics['loss']:.4f} "
                f"ce={metrics['ce_loss']:.4f} cost={metrics['cost']:.2f} "
                f"anchor={metrics['anchor']:.4f} dict_reg={metrics['dict_reg']:.4f} "
                f"tau={tau:.4f}"
            )

    # Actor rollout + oracle mix
    print(f"\nActor rollout + {oracle_mix:.0%} oracle mix ({total_batches - warmup_batches} batches)")
    raw_iter = iter(raw_loader)

    for batch_idx in range(warmup_batches + 1, total_batches + 1):
        try:
            raw_batch = next(raw_iter)
        except StopIteration:
            raw_iter = iter(raw_loader)
            raw_batch = next(raw_iter)

        if len(raw_batch) == 5:
            x_raw, y_raw, m_avail, xs_raw, ms_raw = raw_batch
            xs_raw = torch.nan_to_num(xs_raw).float().to(device)
            ms_raw = ms_raw.float().to(device)
        else:
            x_raw, y_raw, m_avail = raw_batch
            xs_raw = None
            ms_raw = None

        x_raw = torch.nan_to_num(x_raw).to(device)
        y_raw = y_raw.to(device)
        m_avail = m_avail.to(device)

        rollout_result = rollout_batch(
            actor, x_raw, y_raw, m_avail, device, num_time, num_feat,
            x_static=xs_raw, mask_static=ms_raw,
        )
        r_x, r_y, r_m, r_t = rollout_result[:4]
        r_xs = rollout_result[4] if len(rollout_result) > 4 else None
        r_ms = rollout_result[5] if len(rollout_result) > 5 else None

        if oracle_mix > 0:
            if has_aux:
                rollout_arrays = (r_x, r_y, r_m, r_t, r_xs, r_ms)
                oracle_arrays = (x_orc, y_orc, m_orc, t_orc, xs_orc, ms_orc)
            else:
                rollout_arrays = (r_x, r_y, r_m, r_t)
                oracle_arrays = (x_orc, y_orc, m_orc, t_orc)
            mixed = mix_data(rollout_arrays, oracle_arrays, oracle_mix)
            m_x, m_y, m_m, m_t = mixed[:4]
            m_xs = mixed[4] if len(mixed) > 4 else None
            m_ms = mixed[5] if len(mixed) > 5 else None
        else:
            m_x, m_y, m_m, m_t = r_x, r_y, r_m, r_t
            m_xs, m_ms = r_xs, r_ms

        tau = _set_global_tau(global_train_step)
        metrics = train_step(
            actor, optimizer, m_x, m_y, m_m, m_t, device,
            anchor_params=anchor_params,
            anchor_weight=anchor_weight,
            mask_aug_ratio=mask_aug_ratio,
            x_static=m_xs, mask_static=m_ms,
        )
        global_train_step += 1

        if batch_idx % log_every == 0:
            print(
                f"Batch {batch_idx:4d}: loss={metrics['loss']:.4f} "
                f"ce={metrics['ce_loss']:.4f} cost={metrics['cost']:.2f} "
                f"anchor={metrics['anchor']:.4f} dict_reg={metrics['dict_reg']:.4f} "
                f"tau={tau:.4f}"
            )

    # Eval
    actor.eval()
    results, _, _, _ = evaluate_actor(
        actor, test_loader, device, num_time, num_feat,
        feature_costs=config.get('feature_costs'),
        aux_feature_costs=config.get('aux_feature_costs'),
    )
    print(
        f"\nTest: acc={results['accuracy']:.4f} long_cost={results['avg_long_cost']:.2f} "
        f"aux_cost={results['avg_aux_cost']:.2f} total_cost={results['avg_cost']:.2f}"
    )

    save_results_to_csv(
        results,
        DATASET,
        joint=True,
        cost_weight=config['cost_weight'],
        aux_cost_weight=config.get('aux_cost_weight'),
        mask_type=MASK_TYPE,
        csv_path=args.csv_path,
        method_suffix=args.method_suffix,
        baseline=args.baseline,
    )

    # Save (includes updated classifier weights)
    torch.save({
        'state_dict': actor.state_dict(),
        'predictor': actor.predictor.state_dict(),
        'config': config,
        'num_time': num_time,
        'num_feat': num_feat,
        'num_groups': NUM_GROUPS,
        'num_aux': num_aux,
        'y_dim': cls_ckpt['y_dim'],
        'joint': True,
        'classifier_lr': classifier_lr,
        'anchor_weight': anchor_weight,
        'mask_aug_ratio': mask_aug_ratio,
        'global_train_step': global_train_step,
    }, actor_save_path)

    _append_manifest(args.manifest_path, {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'event': 'run_end',
        'dataset': DATASET,
        'baseline': args.baseline,
        'planner_mode': actor.planner_mode,
        'dict_mode': actor.dict_mode if actor.planner_mode == 'dictionary' else '',
        'dict_num_templates': int(actor.dict_num_templates) if actor.planner_mode == 'dictionary' else 0,
        'cw': float(config['cost_weight']),
        'acw': float(config['aux_cost_weight']),
        'seed': int(args.seed),
        'global_train_step': int(global_train_step),
        'actor_save_path': actor_save_path,
        'results': {
            'accuracy': float(results['accuracy']),
            'auroc': float(results['auroc']),
            'auprc': float(results['auprc']),
            'avg_cost': float(results['avg_cost']),
            'avg_long_cost': float(results['avg_long_cost']),
            'avg_aux_cost': float(results['avg_aux_cost']),
        },
    })

    print(f"Model saved to {actor_save_path}")


if __name__ == '__main__':
    main()
