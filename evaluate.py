"""
Evaluate the trained Gumbel Actor-Critic model
"""
import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics import AUROC
from torchmetrics.classification import AveragePrecision
from utils import get_timestep_embedding
from config import (
    DATA_FOLDER, CLASSIFIER_PATH,
    NUM_TIME, NUM_FEAT, DATASET,
    ACTOR_CONFIG, make_actor_path, make_eval_path, make_trajectory_path,
)
from dataset import load_ILIADD_data, load_adni_data, load_synthetic_data, load_cheears_data, load_klg_data, load_womac_data
from models import Predictor, MaskLayer
from gumbel_actor import GumbelActor
from utils import set_seed, build_group_to_feat_matrix

ALL_RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'LAFA_ACTORS - all.csv')


def evaluate_actor(actor, dataloader, device, num_time, num_feat,
                   feature_costs=None, aux_feature_costs=None):
    """
    Evaluate the actor on a dataset

    Returns:
        results: Dict with metrics
        all_masks: List of acquisition masks (group-level)
        all_preds: List of predictions
        all_labels: List of true labels
    """
    actor.eval()
    actor = actor.to(device)

    ng = actor.num_groups

    # Build cost vectors (numpy, for evaluation metrics)
    if feature_costs is not None:
        fc = np.array(feature_costs, dtype=np.float32)
    else:
        fc = np.ones(ng, dtype=np.float32)
    # tile across timesteps: (T * ng,)
    feature_costs_flat_np = np.tile(fc, num_time)

    if aux_feature_costs is not None:
        afc_t = torch.tensor(aux_feature_costs, dtype=torch.float32, device=device)
    else:
        afc_t = torch.ones(max(actor.num_aux, 1), dtype=torch.float32, device=device)

    all_masks = []
    all_preds = []
    all_labels = []
    total_aux_cost = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Unpack: 3-tuple (no aux) or 5-tuple (with aux)
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

            B = x.shape[0]

            # Flatten if needed
            if x.dim() == 3:
                x_flat = x.reshape(B, -1)
                m_avail_flat = m_avail.reshape(B, -1)
            else:
                x_flat = x
                m_avail_flat = m_avail

            # Compute group-level availability
            m_avail_groups = actor.feat_mask_to_group_mask(m_avail_flat.float())

            # Stage 1: acquire aux features
            aux_acquired = None
            aux_gates = None
            if actor.num_aux > 0 and x_static is not None:
                aux_gates = actor.get_aux_gates(B, mask_static)
                aux_acquired = x_static * aux_gates
                total_aux_cost += (aux_gates * afc_t).sum().item()

            # init state at both group and feature level
            m_curr_groups = torch.zeros(B, num_time * ng, dtype=torch.float32, device=device)
            m_curr_feat = torch.zeros(B, num_time * num_feat, dtype=torch.float32, device=device)
            cur_t = torch.zeros(B, dtype=torch.int, device=device)
            m_done = torch.zeros(B, dtype=torch.bool, device=device)

            #iteratively acquire longitudinal features
            for step in range(2 * num_time):
                t_grid = torch.arange(num_time, device=device).unsqueeze(0).expand(B, -1)
                time_mask = t_grid >= cur_t.unsqueeze(1)
                time_mask_g = time_mask.unsqueeze(-1).expand(-1, -1, ng).reshape(B, -1)

                valid_mask_g = (m_avail_groups > 0) & (m_curr_groups == 0) & time_mask_g
                valid_counts = torch.sum(valid_mask_g, dim=1)

                if ((valid_counts == 0) | m_done).all():
                    break


                
                time_emb = get_timestep_embedding(cur_t, embedding_dim=actor.time_emb_dim)

                # Planner input (feature-level)
                x_masked = actor.mask_layer(x_flat, m_curr_feat)
                if aux_acquired is not None:
                    planner_input = torch.cat([x_masked, m_curr_feat, aux_acquired, aux_gates, time_emb], dim=1)
                else:
                    planner_input = torch.cat([x_masked, m_curr_feat, time_emb], dim=1)

                # Unified planner forward (supports direct and dictionary modes)
                _, z_groups = actor.planner_forward(
                    planner_input, valid_mask_g=valid_mask_g, hard=True
                )

                #update group mask - only acquire at current timestep
                cur_t_mask_g = torch.zeros_like(m_curr_groups)
                for b in range(B):
                    if not m_done[b]:
                        t_start = cur_t[b] * ng
                        t_end = (cur_t[b] + 1) * ng
                        cur_t_mask_g[b, t_start:t_end] = z_groups[b, t_start:t_end]

                #update group state
                m_curr_groups = (m_curr_groups + cur_t_mask_g).clamp(0, 1)
                #expand to feature level
                m_curr_feat = actor.expand_group_gates_to_feat_mask(m_curr_groups).clamp(0, 1)

                #update timestep
                added = cur_t_mask_g.sum(dim=1)
                for b in range(B):
                    if added[b] > 0 and not m_done[b]:
                        cur_t[b] = min(cur_t[b] + 1, num_time)

                #check if done
                m_done = m_done | (added == 0)

            #make final predictions
            y_hat = actor.predict_with_mask(x_flat, m_curr_feat, aux_acquired=aux_acquired)

            #store results
            all_masks.append(m_curr_groups.cpu().numpy())  # store group-level masks
            all_preds.append(y_hat.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            total_samples += B

    all_masks = np.concatenate(all_masks, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_preds_flat = all_preds.reshape(-1, all_preds.shape[-1])
    all_labels_flat = all_labels.reshape(-1)

    valid_mask = all_labels_flat != -1
    all_preds_flat = all_preds_flat[valid_mask]
    all_labels_flat = all_labels_flat[valid_mask]

    pred_classes = np.argmax(all_preds_flat, axis=-1)
    accuracy = (pred_classes == all_labels_flat).mean()

    # AUROC and AUPRC via torchmetrics
    y_dim = all_preds_flat.shape[-1]
    preds_t = torch.from_numpy(all_preds_flat).float()
    labels_t = torch.from_numpy(all_labels_flat).long()

    if y_dim == 2:
        auroc_metric = AUROC(task='binary')
        auprc_metric = AveragePrecision(task='binary')
        # Use probability of positive class for binary metrics
        auroc = auroc_metric(preds_t[:, 1], labels_t).item()
        auprc = auprc_metric(preds_t[:, 1], labels_t).item()
    else:
        auroc_metric = AUROC(task='multiclass', num_classes=y_dim)
        auprc_metric = AveragePrecision(task='multiclass', num_classes=y_dim)
        auroc = auroc_metric(preds_t, labels_t).item()
        auprc = auprc_metric(preds_t, labels_t).item()

    avg_long_cost = (all_masks * feature_costs_flat_np).sum(axis=1).mean()

    avg_aux_cost = total_aux_cost / total_samples if total_samples > 0 else 0.0

    results = {
        'accuracy': accuracy,
        'auroc': auroc,
        'auprc': auprc,
        'avg_cost': avg_long_cost + avg_aux_cost,
        'avg_long_cost': avg_long_cost,
        'avg_aux_cost': avg_aux_cost,
        'total_samples': len(all_labels),
    }

    return results, all_masks, all_preds, all_labels


def generate_trajectory(actor, dataloader, device, num_time, num_feat,
                        feature_costs=None, aux_feature_costs=None):
    """
    Run actor's acquisition loop and record one row per acquisition step.

    Returns dict with (N = total acquisition steps across all samples):
        cur_t:      (N,)          timestep at this acquisition step
        x:          (N, T*d)      raw longitudinal features
        m_x:        (N, T*d)      cumulative feature-level mask at this step
        y:          (N, T)        labels
        x_static:   (N, num_aux)  baseline/static features (if present)
        aux_gates:  (num_aux,)    actor's learned auxiliary gates (if present)
    """
    actor.eval()
    actor = actor.to(device)
    ng = actor.num_groups
    max_steps = 2 * num_time

    all_cur_t = []
    all_x = []
    all_m_x = []
    all_y = []
    all_x_static = []
    global_aux_gates = None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating trajectories"):
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
            B = x.shape[0]

            if x.dim() == 3:
                x_flat = x.reshape(B, -1)
                m_avail_flat = m_avail.reshape(B, -1)
            else:
                x_flat = x
                m_avail_flat = m_avail

            m_avail_groups = actor.feat_mask_to_group_mask(m_avail_flat.float())

            # Stage 1: acquire aux features
            aux_acquired = None
            aux_gates = None
            if actor.num_aux > 0 and x_static is not None:
                aux_gates = actor.get_aux_gates(B, mask_static)
                aux_acquired = x_static * aux_gates
                if global_aux_gates is None:
                    global_aux_gates = aux_gates[0].cpu().numpy()

            # Init state
            m_curr_groups = torch.zeros(B, num_time * ng, dtype=torch.float32, device=device)
            m_curr_feat = torch.zeros(B, num_time * num_feat, dtype=torch.float32, device=device)
            cur_t = torch.zeros(B, dtype=torch.int, device=device)
            m_done = torch.zeros(B, dtype=torch.bool, device=device)

            # Record step 0 (before any acquisition) for active samples
            active = ~m_done
            all_cur_t.append(cur_t[active].cpu().numpy())
            all_x.append(x_flat[active].cpu().numpy())
            all_m_x.append(m_curr_feat[active].cpu().numpy())
            all_y.append(y[active].cpu().numpy())
            if x_static is not None:
                all_x_static.append(x_static[active].cpu().numpy())

            # Iterative acquisition
            for step in range(max_steps):
                t_grid = torch.arange(num_time, device=device).unsqueeze(0).expand(B, -1)
                time_mask = t_grid >= cur_t.unsqueeze(1)
                time_mask_g = time_mask.unsqueeze(-1).expand(-1, -1, ng).reshape(B, -1)

                valid_mask_g = (m_avail_groups > 0) & (m_curr_groups == 0) & time_mask_g
                valid_counts = torch.sum(valid_mask_g, dim=1)

                if ((valid_counts == 0) | m_done).all():
                    break

                time_emb = get_timestep_embedding(cur_t, embedding_dim=actor.time_emb_dim)

                x_masked = actor.mask_layer(x_flat, m_curr_feat)
                if aux_acquired is not None:
                    planner_input = torch.cat([x_masked, m_curr_feat, aux_acquired, aux_gates, time_emb], dim=1)
                else:
                    planner_input = torch.cat([x_masked, m_curr_feat, time_emb], dim=1)

                _, z_groups = actor.planner_forward(
                    planner_input, valid_mask_g=valid_mask_g, hard=True
                )

                cur_t_mask_g = torch.zeros_like(m_curr_groups)
                for b in range(B):
                    if not m_done[b]:
                        t_start = cur_t[b] * ng
                        t_end = (cur_t[b] + 1) * ng
                        cur_t_mask_g[b, t_start:t_end] = z_groups[b, t_start:t_end]

                m_curr_groups = (m_curr_groups + cur_t_mask_g).clamp(0, 1)
                m_curr_feat = actor.expand_group_gates_to_feat_mask(m_curr_groups).clamp(0, 1)

                added = cur_t_mask_g.sum(dim=1)
                for b in range(B):
                    if added[b] > 0 and not m_done[b]:
                        cur_t[b] = min(cur_t[b] + 1, num_time)

                m_done = m_done | (added == 0)

                # Record state for samples that were active this step
                active = (added > 0)
                if active.any():
                    all_cur_t.append(cur_t[active].cpu().numpy())
                    all_x.append(x_flat[active].cpu().numpy())
                    all_m_x.append(m_curr_feat[active].cpu().numpy())
                    all_y.append(y[active].cpu().numpy())
                    if x_static is not None:
                        all_x_static.append(x_static[active].cpu().numpy())

    result = {
        'cur_t': np.concatenate(all_cur_t, axis=0),
        'x': np.concatenate(all_x, axis=0),
        'm_x': np.concatenate(all_m_x, axis=0),
        'y': np.concatenate(all_y, axis=0),
    }

    if all_x_static:
        result['x_static'] = np.concatenate(all_x_static, axis=0)
    if global_aux_gates is not None:
        result['aux_gates'] = global_aux_gates

    return result


def save_results_to_csv(results, dataset, joint=False,
                        cost_weight=None, aux_cost_weight=None,
                        csv_path=None, mask_type='uniform',
                        baseline='learned', method_suffix=None):
    """
    Append one row to the shared results CSV (same format as LAFA_ACTORS - all.csv).

    Columns: method, data, (empty), ACC, AUROC, AUPRC, total_cost, long_cost, aux_cost, cw, acw

    Args:
        results: dict returned by evaluate_actor
        dataset: dataset name string (e.g. 'womac', 'klg', 'cheears', 'synthetic')
        joint:   True if the actor was trained jointly with the classifier
        cost_weight: longitudinal cost weight used for training
        aux_cost_weight: auxiliary cost weight used for training
        csv_path: path to CSV file (defaults to ALL_RESULTS_CSV)
        mask_type: classifier mask type ('uniform' or 'bernoulli')
        baseline: baseline feature mode ('learned', 'all', 'none')
    """
    if csv_path is None:
        csv_path = ALL_RESULTS_CSV

    method = 'ACTOR_joint' if joint else 'ACTOR'
    if method_suffix:
        method = f'{method}{method_suffix}'
    if mask_type != 'uniform':
        method = f'{method}_{mask_type}'
    if baseline != 'learned':
        method = f'{method}_baseline_{baseline}'

    row = [
        method,
        dataset,
        '',                                             # empty column
        results.get('accuracy', ''),
        results.get('auroc', ''),
        results.get('auprc', ''),
        results.get('avg_cost', ''),
        results.get('avg_long_cost', ''),
        results.get('avg_aux_cost', ''),
        cost_weight if cost_weight is not None else '',
        aux_cost_weight if aux_cost_weight is not None else '',
    ]

    file_exists = os.path.exists(csv_path)
    # Ensure existing file ends with a newline before appending
    if file_exists and os.path.getsize(csv_path) > 0:
        with open(csv_path, 'rb') as f:
            f.seek(-1, 2)
            if f.read(1) != b'\n':
                with open(csv_path, 'a') as fa:
                    fa.write('\n')
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['method', 'data', '', 'ACC', 'AUROC', 'AUPRC',
                             'total_cost', 'long_cost', 'aux_cost', 'cw', 'acw'])
        writer.writerow(row)

    print(f"Results appended to {csv_path}")


def _get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _load_actor_state_dict(actor, ckpt):
    """Load actor state dict, handling both checkpoint key formats"""
    if 'state_dict' in ckpt:
        actor.load_state_dict(ckpt['state_dict'])
    elif 'actor' in ckpt:
        actor.load_state_dict(ckpt['actor'])
    else:
        raise KeyError(f"Checkpoint has no 'state_dict' or 'actor' key. Keys: {list(ckpt.keys())}")


def evaluate_actor_from_path(actor_path, test_data_path=None, data_folder=None,
                             print_results=True, baseline='learned'):
    """
    Evaluate actor from checkpoint path

    Args:
        actor_path: Path to actor checkpoint
        test_data_path: Path to test data (optional, defaults to config)
        print_results: Whether to print results

    Returns:
        dict: Results dictionary with accuracy and avg_cost
    """
    if test_data_path is None:
        test_root = data_folder if data_folder is not None else DATA_FOLDER
        test_data_path = os.path.join(test_root, 'test_data.npz')

    print(f"*** classifier: {CLASSIFIER_PATH}")
    print(f"*** actor: {actor_path}")
    print(f"*** test_data: {test_data_path}")

    # Load classifier
    classifier_ckpt = torch.load(CLASSIFIER_PATH, map_location='cpu')

    num_time = classifier_ckpt['num_time']
    num_feat = classifier_ckpt['num_feat']
    num_aux = classifier_ckpt.get('num_aux', 0)

    predictor = Predictor(
        d_in=num_time * num_feat + num_aux,
        d_out=classifier_ckpt['y_dim'],
        hidden=classifier_ckpt['config']['hidden_dim'],
        dropout=classifier_ckpt['config']['dropout']
    )
    predictor.load_state_dict(classifier_ckpt['predictor'])

    # Load actor
    actor_ckpt = torch.load(actor_path, map_location='cpu')

    # Build group-to-feature matrix if needed
    num_groups = actor_ckpt.get('num_groups', num_feat)
    group_to_feat_matrix = None
    if num_groups != num_feat:
        group_to_feat_matrix = build_group_to_feat_matrix(num_feat)

    actor = GumbelActor(
        predictor=predictor,
        num_time=num_time,
        num_feat=num_feat,
        config=actor_ckpt['config'],
        num_aux=num_aux,
        num_groups=num_groups,
        group_to_feat_matrix=group_to_feat_matrix,
    )
    _load_actor_state_dict(actor, actor_ckpt)

    if baseline == 'all' and actor.num_aux > 0:
        actor.aux_logits.data.fill_(100.0)
    elif baseline == 'none' and actor.num_aux > 0:
        actor.aux_logits.data.fill_(-100.0)

    # Load test data
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
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate
    device = _get_device()
    actor_config = actor_ckpt.get('config', {})
    results, masks, preds, labels = evaluate_actor(
        actor, test_loader, device, num_time, num_feat,
        feature_costs=actor_config.get('feature_costs'),
        aux_feature_costs=actor_config.get('aux_feature_costs'),
    )

    if print_results:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"AUROC:    {results['auroc']:.4f}")
        print(f"AUPRC:    {results['auprc']:.4f}")
        print(f"Average Cost (total): {results['avg_cost']:.2f}")
        print(f"  Longitudinal (groups): {results['avg_long_cost']:.2f}")
        print(f"  Auxiliary: {results['avg_aux_cost']:.2f}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Num groups: {num_groups}")
        print("="*60)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', type=str, default=None,
                        help='Path to actor checkpoint (default: derived from config HPs)')
    parser.add_argument('--cost_weight', type=float, default=None,
                        help='cost_weight used to locate checkpoint (overrides config)')
    parser.add_argument('--aux_cost_weight', type=float, default=None,
                        help='aux_cost_weight used to locate checkpoint (overrides config)')
    parser.add_argument('--joint', action='store_true', default=False,
                        help='Evaluate joint actor checkpoint')
    parser.add_argument('--trajectory', action='store_true', default=False,
                        help='Generate and save step-by-step acquisition trajectory')
    parser.add_argument('--baseline', type=str, default='learned',
                        choices=['learned', 'all', 'none'],
                        help='Baseline feature mode: learned (default), all, or none')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='Override dataset split root (train/val/test .npz files)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    cw = args.cost_weight if args.cost_weight is not None else ACTOR_CONFIG['cost_weight']
    if args.aux_cost_weight is not None and args.aux_cost_weight != cw:
        raise ValueError(
            f"Expected cw == acw, got cost_weight={cw} and aux_cost_weight={args.aux_cost_weight}"
        )
    acw = cw

    data_folder = args.data_folder if args.data_folder else DATA_FOLDER

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_PATH}. "
            f"Please run train_classifier.py first."
        )

    # Use explicit path or derive from HPs
    actor_path = args.actor_path if args.actor_path else make_actor_path(cw, acw, joint=args.joint, baseline=args.baseline)
    if not os.path.exists(actor_path):
        raise FileNotFoundError(
            f"Actor not found at {actor_path}. "
            f"Please run train_actor_iterative.py first."
        )

    print(f"*** classifier: {CLASSIFIER_PATH}")
    print(f"*** actor: {actor_path}")
    print(f"*** test_data: {os.path.join(data_folder, 'test_data.npz')}")

    # Evaluate using the shared function
    print(f"Evaluating {actor_path} on test set...")
    results = evaluate_actor_from_path(
        actor_path,
        data_folder=data_folder,
        print_results=True,
        baseline=args.baseline,
    )

    # Also save detailed results
    print("\nLoading detailed results for saving...")
    classifier_ckpt = torch.load(CLASSIFIER_PATH, map_location='cpu')
    num_time = classifier_ckpt['num_time']
    num_feat = classifier_ckpt['num_feat']
    num_aux = classifier_ckpt.get('num_aux', 0)

    predictor = Predictor(
        d_in=num_time * num_feat + num_aux,
        d_out=classifier_ckpt['y_dim'],
        hidden=classifier_ckpt['config']['hidden_dim'],
        dropout=classifier_ckpt['config']['dropout']
    )
    predictor.load_state_dict(classifier_ckpt['predictor'])

    actor_ckpt = torch.load(actor_path, map_location='cpu')
    num_groups = actor_ckpt.get('num_groups', num_feat)
    group_to_feat_matrix = None
    if num_groups != num_feat:
        group_to_feat_matrix = build_group_to_feat_matrix(num_feat)

    actor = GumbelActor(
        predictor=predictor,
        num_time=num_time,
        num_feat=num_feat,
        config=actor_ckpt['config'],
        num_aux=num_aux,
        num_groups=num_groups,
        group_to_feat_matrix=group_to_feat_matrix,
    )
    _load_actor_state_dict(actor, actor_ckpt)

    # Override baseline feature gates if requested
    if args.baseline == 'all' and actor.num_aux > 0:
        actor.aux_logits.data.fill_(100.0)
        print("Forcing ALL baseline features to be acquired")
    elif args.baseline == 'none' and actor.num_aux > 0:
        actor.aux_logits.data.fill_(-100.0)
        print("Forcing NO baseline features to be acquired")

    if DATASET in ('cheears', 'cheears_demog'):
        test_dataset = load_cheears_data(os.path.join(data_folder, 'test_data.npz'))
    elif DATASET == 'klg':
        test_dataset = load_klg_data(os.path.join(data_folder, 'test_data.npz'))
    elif DATASET == 'womac':
        test_dataset = load_womac_data(os.path.join(data_folder, 'test_data.npz'))
    elif DATASET == 'ILIADD':
        test_dataset = load_ILIADD_data(os.path.join(data_folder, 'test_data.npz'))
    elif DATASET == 'adni':
        test_dataset = load_adni_data(os.path.join(data_folder, 'test_data.npz'))
    else:
        test_dataset = load_synthetic_data(os.path.join(data_folder, 'test_data.npz'))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = _get_device()
    actor_cfg = actor_ckpt.get('config', {})
    _, masks, preds, labels = evaluate_actor(
        actor, test_loader, device, num_time, num_feat,
        feature_costs=actor_cfg.get('feature_costs'),
        aux_feature_costs=actor_cfg.get('aux_feature_costs'),
    )

    results_path = make_eval_path(cw, acw)
    np.savez(
        results_path,
        masks=masks,
        predictions=preds,
        labels=labels,
        accuracy=results['accuracy'],
        auroc=results['auroc'],
        auprc=results['auprc'],
        avg_cost=results['avg_cost'],
        avg_long_cost=results['avg_long_cost'],
        avg_aux_cost=results['avg_aux_cost'],
    )
    print(f"\nResults saved to {results_path}")

    # save_results_to_csv(results, DATASET, joint=args.joint,
    #                     cost_weight=cw, aux_cost_weight=acw)

    if args.trajectory:
        print("\nGenerating step-by-step trajectory...")
        test_loader2 = DataLoader(test_dataset, batch_size=64, shuffle=False)
        traj = generate_trajectory(
            actor, test_loader2, device, num_time, num_feat,
            feature_costs=actor_cfg.get('feature_costs'),
            aux_feature_costs=actor_cfg.get('aux_feature_costs'),
        )
        traj_path = make_trajectory_path(cw, acw, joint=args.joint)
        np.savez(traj_path, **traj)
        print(f"Trajectory saved to {traj_path}")
        print(f"  N (total steps): {traj['cur_t'].shape[0]}")
        print(f"  cur_t: {traj['cur_t'].shape}")
        print(f"  x:     {traj['x'].shape}")
        print(f"  m_x:   {traj['m_x'].shape}")
        print(f"  y:     {traj['y'].shape}")
        if 'x_static' in traj:
            print(f"  x_static:  {traj['x_static'].shape}")
        if 'aux_gates' in traj:
            print(f"  aux_gates: {traj['aux_gates'].shape}")


if __name__ == '__main__':
    main()
