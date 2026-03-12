"""
Generate oracle rollouts using the trained classifier
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader

from config import (
    DATA_FOLDER, CLASSIFIER_PATH, ORACLE_CONFIG,
    DATASET, ACTOR_CONFIG, NUM_GROUPS, NUM_FEAT,
    make_oracle_path,
)
from dataset import load_ILIADD_data, load_adni_data, load_synthetic_data, load_cheears_data, load_klg_data, load_womac_data
from models import Predictor, MaskLayer
from oracle_generator import OracleRolloutGenerator
from utils import set_seed, build_group_to_feat_matrix


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost_weight', type=float, default=None,
                        help='Feature cost weight (overrides config)')
    parser.add_argument('--aux_cost_weight', type=float, default=None,
                        help='Auxiliary feature cost weight (overrides config)')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='Override dataset split root (train/val/test .npz files)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Check if classifier exists
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f"Classifier not found at {CLASSIFIER_PATH}. "
            f"Please run train_classifier.py first."
        )

    # Load classifier
    print("Loading trained classifier...")
    print("*** ", CLASSIFIER_PATH)
    checkpoint = torch.load(CLASSIFIER_PATH, map_location='cpu')

    num_time = checkpoint['num_time']
    num_feat = checkpoint['num_feat']
    num_aux = checkpoint.get('num_aux', 0)

    predictor = Predictor(
        d_in=num_time * num_feat + num_aux,
        d_out=checkpoint['y_dim'],
        hidden=checkpoint['config']['hidden_dim'],
        dropout=checkpoint['config']['dropout']
    )
    predictor.load_state_dict(checkpoint['predictor'])

    mask_layer = MaskLayer(
        mask_size=num_time * num_feat,
        append=False
    )
    mask_layer.load_state_dict(checkpoint['mask_layer'])

    print(f"Loaded classifier with {num_time} timesteps, {num_feat} features per timestep, {num_aux} aux features")

    # Load training data for oracle generation
    data_folder = args.data_folder if args.data_folder else DATA_FOLDER
    print("\nLoading training data...")
    if DATASET in ('cheears', 'cheears_demog'):
        train_dataset = load_cheears_data(os.path.join(data_folder, 'train_data.npz'))
    elif DATASET == 'klg':
        train_dataset = load_klg_data(os.path.join(data_folder, 'train_data.npz'))
    elif DATASET == 'womac':
        train_dataset = load_womac_data(os.path.join(data_folder, 'train_data.npz'))
    elif DATASET == 'ILIADD':
        train_dataset = load_ILIADD_data(os.path.join(data_folder, 'train_data.npz'))
    elif DATASET == 'adni':
        train_dataset = load_adni_data(os.path.join(data_folder, 'train_data.npz'))
    else:
        train_dataset = load_synthetic_data(os.path.join(data_folder, 'train_data.npz'))

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False
    )

    # Create oracle generator
    print("\nCreating oracle generator...")
    device = _get_device()
    print(f"Device: {device}")

    cost_weight = args.cost_weight if args.cost_weight is not None else ORACLE_CONFIG['cost_weight']

    # Skip if oracle rollout already exists
    save_path = make_oracle_path(cost_weight)
    if os.path.exists(save_path):
        print(f"Oracle rollout already exists at {save_path}, skipping.")
        return

    if args.aux_cost_weight is not None and args.aux_cost_weight != cost_weight:
        raise ValueError(
            f"Expected cw == acw, got cost_weight={cost_weight} and aux_cost_weight={args.aux_cost_weight}"
        )
    aux_cost_weight = cost_weight

    # Build group-to-feature matrix if using groups
    group_to_feat_matrix = None
    if NUM_GROUPS != NUM_FEAT:
        group_to_feat_matrix = build_group_to_feat_matrix(num_feat)
        print(f"Using group-based acquisition: {NUM_GROUPS} groups -> {num_feat} features")

    oracle_gen = OracleRolloutGenerator(
        predictor=predictor,
        mask_layer=mask_layer,
        num_time=num_time,
        num_feat=num_feat,
        device=device,
        cost_weight=cost_weight,
        time_weight=ORACLE_CONFIG['time_weight'],
        feature_costs=ORACLE_CONFIG['feature_costs'],
        num_samples=ORACLE_CONFIG['num_samples'],
        num_aux=num_aux,
        aux_cost_weight=aux_cost_weight,
        num_groups=NUM_GROUPS,
        group_to_feat_matrix=group_to_feat_matrix,
        aux_feature_costs=ORACLE_CONFIG.get('aux_feature_costs'),
    )

    # Generate rollouts
    print(f"\nGenerating oracle rollouts...")
    oracle_dataset = oracle_gen.generate(
        dataloader=train_loader,
        save_path=save_path
    )

    print(f"\nOracle rollout generation complete!")
    print(f"Data saved to {save_path}")
    print(f"Total states collected: {len(oracle_dataset)}")


if __name__ == '__main__':
    main()
