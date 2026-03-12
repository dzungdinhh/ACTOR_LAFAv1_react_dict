"""
Iterative Actor Training 
step1:Train on oracle data for `warmup_batches` steps
step2:Rollout actor, mix oracle states, train
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from config import (
    DATA_FOLDER, CLASSIFIER_PATH,
    ITERATIVE_ACTOR_CONFIG, DATASET,
    NUM_GROUPS, NUM_FEAT, ORACLE_CONFIG,
    make_oracle_path, make_actor_path,
)
from dataset import load_ILIADD_data, load_adni_data, load_oracle_rollout, load_synthetic_data, load_cheears_data, load_klg_data, load_womac_data
from models import Predictor
from gumbel_actor import GumbelActor
from evaluate import evaluate_actor, save_results_to_csv
from utils import (
    set_seed, get_timestep_embedding, generate_random_masks_for_cur_t,
    build_group_to_feat_matrix,
)


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


#rollout one batch

def rollout_batch(actor, x, y, m_avail, device, num_time, num_feat,
                  x_static=None, mask_static=None):
    """
    Rollout the actor on a single batch of raw data.
    Collects the state BEFORE each action as a training sample.
    Saves feature-level masks for training compatibility.
    """
    actor.eval()
    B = x.shape[0]
    ng = actor.num_groups

    x_flat = x.reshape(B, -1) if x.dim() == 3 else x
    m_avail_flat = m_avail.reshape(B, -1) if m_avail.dim() == 3 else m_avail
    y_flat = y if y.dim() == 2 else y.reshape(B, -1)

    sx, sy, sm, st = [], [], [], []
    s_xstatic, s_mstatic = [], []

    # Track at both levels
    m_curr_groups = torch.zeros(B, num_time * ng, dtype=torch.float32, device=device)
    m_curr_feat = torch.zeros_like(x_flat, dtype=torch.float32)
    cur_t = torch.zeros(B, dtype=torch.int, device=device)
    m_done = torch.zeros(B, dtype=torch.bool, device=device)

    # Group-level availability
    m_avail_groups = actor.feat_mask_to_group_mask(m_avail_flat.float())

    # aux gates
    aux_acquired = None
    aux_gates = None
    if actor.num_aux > 0 and x_static is not None:
        aux_gates = actor.get_aux_gates(B, mask_static)
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

            # Valid groups
            t_grid = torch.arange(num_time, device=device).unsqueeze(0).expand(B, -1)
            time_ok = t_grid >= cur_t.unsqueeze(1)
            time_flat_g = time_ok.unsqueeze(-1).expand(-1, -1, ng).reshape(B, -1)
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

            # Only current timestep groups
            cur_t_group_mask = torch.zeros_like(m_curr_groups)
            for b in range(B):
                if not m_done[b]:
                    s, e = cur_t[b] * ng, (cur_t[b] + 1) * ng
                    cur_t_group_mask[b, s:e] = z[b, s:e]

            m_curr_groups = (m_curr_groups + cur_t_group_mask).clamp(0, 1)
            m_curr_feat = actor.expand_group_gates_to_feat_mask(m_curr_groups).clamp(0, 1)

            added = cur_t_group_mask.sum(dim=1)
            for b in range(B):
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


# train
def _forward_loss(actor, x, y, mask, cur_t, device,
                  x_static=None, mask_static=None):
    """Shared forward pass that computes loss without optimizer step."""
    T = actor.num_time
    d = actor.num_feat
    ng = actor.num_groups

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1, T * d)
    y_t = torch.as_tensor(y, dtype=torch.int64, device=device).reshape(-1, T)
    mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).reshape(-1, T * d)
    cur_t_t = torch.as_tensor(cur_t, dtype=torch.int64, device=device)
    time_emb = get_timestep_embedding(cur_t_t, embedding_dim=actor.time_emb_dim)

    B = x_t.size(0)
    orig_mask_f = mask_t.float()
    # Convert to group level
    orig_mask_g = actor.feat_mask_to_group_mask(orig_mask_f)
    mask_after_g = actor.after_cur_t_mask(cur_t_t, T, ng, device).float()

    # aux gates
    aux_acquired = None
    aux_gates = None
    aux_cost_loss = 0.0
    if actor.num_aux > 0 and x_static is not None:
        xs_t = torch.as_tensor(x_static, dtype=torch.float32, device=device)
        ms_t = torch.as_tensor(mask_static, dtype=torch.float32, device=device)
        aux_gates = actor.get_aux_gates(B, ms_t)
        aux_acquired = xs_t * aux_gates
        aux_cost_loss = actor.aux_cost_weight * (aux_gates * actor.aux_feature_costs).sum() / B

    # Planner input is feature-level
    x_masked = actor.mask_layer(x_t, mask_t)
    if aux_acquired is not None:
        inp = torch.cat([x_masked, mask_t, aux_acquired, aux_gates, time_emb], dim=1)
    else:
        inp = torch.cat([x_masked, mask_t, time_emb], dim=1)

    # Group-level gates
    _, z_groups = actor.planner_forward(inp, hard=True)

    z_safe_g = z_groups * mask_after_g * (1.0 - orig_mask_g.detach())
    gated_groups = (orig_mask_g.detach() + z_safe_g).clamp(0, 1)

    # Expand for prediction
    gated_mask_feat = actor.expand_group_gates_to_feat_mask(gated_groups).clamp(0, 1)

    y_hat = actor.predict_with_mask(x_t, gated_mask_feat, aux_acquired=aux_acquired)
    _, Tp, _ = y_hat.shape

    t_idx = torch.arange(Tp, device=device).view(1, Tp)
    use = (t_idx >= cur_t_t.view(B, 1)) & (y_t != -1)

    ce_loss = F.cross_entropy(y_hat[use], y_t[use].long())
    #cost at group level, weighted by per-group costs
    cost = (gated_groups * mask_after_g * actor.feature_costs_flat).sum() / B
    cost_loss = actor.cost_weight * cost
    dict_reg_loss = actor.dictionary_regularization_loss()
    loss = ce_loss + cost_loss + aux_cost_loss + dict_reg_loss

    return loss, ce_loss, cost, dict_reg_loss


def train_step(actor, optimizer, x, y, mask, cur_t, device,
               x_static=None, mask_static=None):
    """1 gradient update"""
    actor.train()
    loss, ce_loss, cost, dict_reg = _forward_loss(
        actor, x, y, mask, cur_t, device,
        x_static=x_static, mask_static=mask_static,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'ce_loss': ce_loss.item(),
        'cost': cost.item(),
        'dict_reg': dict_reg.item(),
    }


def val_loss_on_rollout(actor, val_loader, device, num_time, num_feat):
    """Rollout actor on val data, compute loss on collected states."""
    actor.eval()
    all_losses = []

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 5:
                x, y, m_avail, x_static, mask_static = batch
                x_static = torch.nan_to_num(x_static).float().to(device)
                mask_static = mask_static.float().to(device)
            else:
                x, y, m_avail = batch
                x_static = None
                mask_static = None

            x = torch.nan_to_num(x).to(device)
            y = y.to(device)
            m_avail = m_avail.to(device)

            rollout_result = rollout_batch(
                actor, x, y, m_avail, device, num_time, num_feat,
                x_static=x_static, mask_static=mask_static,
            )
            r_x, r_y, r_m, r_t = rollout_result[:4]
            r_xs = rollout_result[4] if len(rollout_result) > 4 else None
            r_ms = rollout_result[5] if len(rollout_result) > 5 else None

            loss, _, _, _ = _forward_loss(
                actor, r_x, r_y, r_m, r_t, device,
                x_static=r_xs, mask_static=r_ms,
            )
            all_losses.append(loss.item())

    return float(np.mean(all_losses))


#helpers

def sample_oracle(x_orc, y_orc, m_orc, t_orc, n,
                  xs_orc=None, ms_orc=None):
    """sample n random states from the oracle pool"""
    idx = np.random.choice(len(x_orc), size=min(n, len(x_orc)), replace=False)
    result = (x_orc[idx], y_orc[idx], m_orc[idx], t_orc[idx])
    if xs_orc is not None:
        result = result + (xs_orc[idx], ms_orc[idx])
    return result


def mix_data(rollout_data, oracle_data, oracle_ratio):
    """mix actor rollout states with oracle states"""
    n_rollout = len(rollout_data[0])
    n_oracle = max(1, int(n_rollout * oracle_ratio / (1.0 - oracle_ratio)))

    n_arrays = len(rollout_data)
    idx = np.random.choice(len(oracle_data[0]), size=min(n_oracle, len(oracle_data[0])), replace=False)

    result = tuple(
        np.concatenate([rollout_data[i], oracle_data[i][idx]])
        for i in range(n_arrays)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost_weight', type=float, default=None,
                        help='Longitudinal feature cost weight (overrides config)')
    parser.add_argument('--aux_cost_weight', type=float, default=None,
                        help='Auxiliary feature cost weight (overrides config)')
    parser.add_argument('--baseline', type=str, default='learned',
                        choices=['learned', 'all', 'none'],
                        help='Baseline feature mode: learned (default), all, or none')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='CSV path for results (default: LAFA_ACTORS - all.csv)')
    parser.add_argument('--warmup_batches', type=int, default=None,
                        help='Number of oracle warmup batches (overrides config, 0=no warmup)')
    parser.add_argument('--total_batches', type=int, default=None,
                        help='Total training batches (overrides config)')
    parser.add_argument('--save_suffix', type=str, default=None,
                        help='Extra suffix appended to checkpoint filename (e.g. "warmup0")')
    parser.add_argument('--method_suffix', type=str, default=None,
                        help='Suffix appended to method name in CSV (e.g. "_warmup")')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='Override dataset split root (train/val/test .npz files)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    config = {**ITERATIVE_ACTOR_CONFIG}
    if args.cost_weight is not None:
        config['cost_weight'] = args.cost_weight
    if args.aux_cost_weight is not None and args.aux_cost_weight != config['cost_weight']:
        raise ValueError(
            f"Expected cw == acw, got cost_weight={config['cost_weight']} and aux_cost_weight={args.aux_cost_weight}"
        )
    config['aux_cost_weight'] = config['cost_weight']
    if args.warmup_batches is not None:
        config['warmup_batches'] = args.warmup_batches
    if args.total_batches is not None:
        config['total_batches'] = args.total_batches

    # Build HP-specific paths
    oracle_cost_weight = args.cost_weight if args.cost_weight is not None else ORACLE_CONFIG['cost_weight']
    oracle_rollout_path = make_oracle_path(oracle_cost_weight)
    actor_save_path = make_actor_path(config['cost_weight'], config.get('aux_cost_weight'),
                                      baseline=args.baseline)
    if args.save_suffix:
        actor_save_path = actor_save_path.replace('.ckpt', f'_{args.save_suffix}.ckpt')

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_PATH}. "
            f"Please run train_classifier.py first."
        )
    if not os.path.exists(oracle_rollout_path):
        raise FileNotFoundError(
            f"Oracle rollout not found at {oracle_rollout_path}. "
            f"Please run generate_oracle.py first."
        )

    #load classifier (frozen)
    print("Loading pre-trained classifier...")
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

    #group config(for one hot feat groups)
    group_to_feat_matrix = None
    group_to_feat_np = None
    if NUM_GROUPS != NUM_FEAT:
        group_to_feat_matrix = build_group_to_feat_matrix(num_feat)
        group_to_feat_np = group_to_feat_matrix.numpy()
        print(f"this is group-based acquisition: {NUM_GROUPS} groups -> {num_feat} features")

    # oracle rollout data
    print("oracle rollout...")
    oracle_data = load_oracle_rollout(
        oracle_rollout_path, num_time=num_time, num_feat=num_feat,
    )
    x_orc, y_orc, m_orc, t_orc = oracle_data[:4]
    xs_orc = oracle_data[4]
    ms_orc = oracle_data[5]
    #random data augmentation
    # rand_m, rand_t, rand_x, rand_y = generate_random_masks_for_cur_t(
    #     x_orc, y_orc, t_orc, num_time, num_feat, num_samples_per_state=1,
    #     num_groups=NUM_GROUPS if group_to_feat_np is not None else None,
    #     group_to_feat_np=group_to_feat_np,
    # )
    # x_orc = np.concatenate([x_orc, rand_x])
    # y_orc = np.concatenate([y_orc, rand_y])
    # m_orc = np.concatenate([m_orc, rand_m])
    # t_orc = np.concatenate([t_orc, rand_t])

    # if xs_orc is not None: #uncomment if using random aug
    #     xs_orc = np.concatenate([xs_orc, xs_orc])
    #     ms_orc = np.concatenate([ms_orc, ms_orc])

    print(f"oracle: {len(x_orc)} states")

    #load data
    data_folder = args.data_folder if args.data_folder else DATA_FOLDER
    train_data_path = os.path.join(data_folder, 'train_data.npz')
    if DATASET in ('cheears', 'cheears_demog'):
        train_dataset = load_cheears_data(train_data_path)
    elif DATASET == 'klg':
        train_dataset = load_klg_data(train_data_path)
    elif DATASET == 'womac':
        train_dataset = load_womac_data(train_data_path)
    elif DATASET == 'ILIADD':
        train_dataset = load_ILIADD_data(train_data_path)
    elif DATASET == 'adni':
        train_dataset = load_adni_data(train_data_path)
    else:
        train_dataset = load_synthetic_data(train_data_path)
    raw_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"Train samples: {len(train_dataset)}")


    test_data_path = os.path.join(data_folder, 'test_data.npz')
    if DATASET in ('cheears', 'cheears_demog'):
        test_dataset = load_cheears_data(test_data_path)
    elif DATASET == 'klg':
        test_dataset = load_klg_data(test_data_path)
    elif DATASET == 'womac':
        test_dataset = load_womac_data(test_data_path)
    elif DATASET == 'ILIADD':
        test_dataset = load_ILIADD_data(test_data_path)
    elif DATASET == 'adni':
        test_dataset = load_adni_data(test_data_path)
    else:
        test_dataset = load_synthetic_data(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=config['val_batch_size'], shuffle=False)

    device = _get_device()
    print(f"Device: {device}")

    actor = GumbelActor(
        predictor=predictor, num_time=num_time, num_feat=num_feat,
        config=config, num_aux=num_aux,
        num_groups=NUM_GROUPS, group_to_feat_matrix=group_to_feat_matrix,
    ).to(device)

    # Override baseline feature gates if requested
    if args.baseline == 'all' and num_aux > 0:
        actor.aux_logits.data.fill_(100.0)
        actor.aux_logits.requires_grad = False
        print("Forcing ALL baseline features (fixed, not learned)")
    elif args.baseline == 'none' and num_aux > 0:
        actor.aux_logits.data.fill_(-100.0)
        actor.aux_logits.requires_grad = False
        print("Forcing NO baseline features (fixed, not learned)")

    params = actor.planner_parameters()
    if num_aux > 0 and args.baseline == 'learned':
        params.append(actor.aux_logits)
    optimizer = torch.optim.Adam(params, lr=config['lr'])

    total_batches = config.get('total_batches', 1000)
    warmup_batches = config.get('warmup_batches', 50)
    oracle_mix = config.get('oracle_mix_ratio', 0.3)
    log_every = config.get('log_every', 10)
    bs = config['batch_size']

    has_aux = num_aux > 0 and xs_orc is not None

    print(f"\nConfig: total={total_batches}  warmup={warmup_batches}  "
          f"mix={oracle_mix}  cw={config['cost_weight']}  "
          f"num_aux={num_aux}  num_groups={NUM_GROUPS}")

    #oracle warm-up

    print(f"\n oracle warm-up ({warmup_batches} batches)...")
    for batch_idx in range(1, warmup_batches + 1):
        if has_aux:
            ox, oy, om, ot, oxs, oms = sample_oracle(
                x_orc, y_orc, m_orc, t_orc, bs, xs_orc, ms_orc,
            )
        else:
            ox, oy, om, ot = sample_oracle(x_orc, y_orc, m_orc, t_orc, bs)
            oxs, oms = None, None

        metrics = train_step(
            actor, optimizer, ox, oy, om, ot, device,
            x_static=oxs, mask_static=oms,
        )

        if batch_idx % log_every == 0:
            print(f"Batch {batch_idx:4d} (oracle): loss={metrics['loss']:.4f}  "
                  f"ce={metrics['ce_loss']:.4f}  cost={metrics['cost']:.2f}")

    # actor rollout(can mix if oracle_mix >0)

    print(f"\nactor rollout + {oracle_mix:.0%} oracle mix  "
          f"({total_batches - warmup_batches} batches)")
    raw_iter = iter(raw_loader)

    for batch_idx in range(warmup_batches + 1, total_batches + 1):
        try:
            raw_batch = next(raw_iter)
        except StopIteration:
            raw_iter = iter(raw_loader)
            raw_batch = next(raw_iter)

        # Unpack raw batch
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

        metrics = train_step(
            actor, optimizer, m_x, m_y, m_m, m_t, device,
            x_static=m_xs, mask_static=m_ms,
        )

        if batch_idx % log_every == 0:
            print(f"Batch {batch_idx:4d}: loss={metrics['loss']:.4f}  "
                  f"ce={metrics['ce_loss']:.4f}  cost={metrics['cost']:.2f}")

    # eval
    actor.eval()
    results, _, _, _ = evaluate_actor(
        actor, test_loader, device, num_time, num_feat,
        feature_costs=config.get('feature_costs'),
        aux_feature_costs=config.get('aux_feature_costs'),
    )
    print(f"\nTest: acc={results['accuracy']:.4f}  "
          f"long_cost={results['avg_long_cost']:.2f}  "
          f"aux_cost={results['avg_aux_cost']:.2f}  "
          f"total_cost={results['avg_cost']:.2f}")

    save_results_to_csv(results, DATASET, joint=False,
                        cost_weight=config['cost_weight'],
                        aux_cost_weight=config.get('aux_cost_weight'),
                        csv_path=args.csv_path,
                        baseline=args.baseline,
                        method_suffix=args.method_suffix)

    torch.save({
        'state_dict': actor.state_dict(),
        'config': config,
        'num_time': num_time,
        'num_feat': num_feat,
        'num_groups': NUM_GROUPS,
        'num_aux': num_aux,
        'y_dim': cls_ckpt['y_dim'],
    }, actor_save_path)

    print(f"Model saved to {actor_save_path}")


if __name__ == '__main__':
    main()
