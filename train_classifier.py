"""
Train the prediction network (classifier)
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchmetrics import Accuracy

from config import (
    DATA_FOLDER, OUTPUT_FOLDER, CLASSIFIER_CONFIG, CLASSIFIER_PATH,
    DATASET, NUM_AUX, MASK_TYPE,
)
from dataset import load_synthetic_data, load_cheears_data, load_klg_data, load_womac_data, load_ILIADD_data, load_adni_data
from models import Predictor, MaskLayer
from classifier_trainer import ClassifierTrainer
from utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default=None,
                        help='Override dataset split root (train/val/test .npz files)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_retrain', action='store_true',
                        help='Retrain even if classifier checkpoint already exists')
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if os.path.exists(CLASSIFIER_PATH) and not args.force_retrain:
        print(f"Classifier already exists at {CLASSIFIER_PATH}; skipping (use --force_retrain to override).")
        return

    data_folder = args.data_folder if args.data_folder else DATA_FOLDER

    # Load data
    print("Loading data...")
    if DATASET in ('cheears', 'cheears_demog'):
        train_dataset = load_cheears_data(os.path.join(data_folder, 'train_data.npz'))
        val_dataset = load_cheears_data(os.path.join(data_folder, 'val_data.npz'))
    elif DATASET == 'klg':
        train_dataset = load_klg_data(os.path.join(data_folder, 'train_data.npz'))
        val_dataset = load_klg_data(os.path.join(data_folder, 'val_data.npz'))
    elif DATASET == 'womac':
        train_dataset = load_womac_data(os.path.join(data_folder, 'train_data.npz'))
        val_dataset = load_womac_data(os.path.join(data_folder, 'val_data.npz'))
    elif DATASET == 'ILIADD':
        train_dataset = load_ILIADD_data(os.path.join(data_folder, 'train_data.npz'))
        val_dataset = load_ILIADD_data(os.path.join(data_folder, 'val_data.npz'))
    elif DATASET == 'adni':
        train_dataset = load_adni_data(os.path.join(data_folder, 'train_data.npz'))
        val_dataset = load_adni_data(os.path.join(data_folder, 'val_data.npz'))
    else:
        train_dataset = load_synthetic_data(os.path.join(data_folder, 'train_data.npz'))
        val_dataset = load_synthetic_data(os.path.join(data_folder, 'val_data.npz'))

    num_aux = train_dataset.num_aux

    print(f"Dataset: {DATASET}")
    print(f"Mask type: {MASK_TYPE}")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Number of timesteps: {train_dataset.t}")
    print(f"Number of features per timestep: {train_dataset.x_dim}")
    print(f"Number of aux features: {num_aux}")
    print(f"Number of classes: {train_dataset.y_dim}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CLASSIFIER_CONFIG['batch_size'],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CLASSIFIER_CONFIG['batch_size'],
        shuffle=False
    )

    # Predictor input dimension: T*d + num_aux
    d_in = train_dataset.x_dim * train_dataset.t + num_aux

    # Create model components
    print(f"\nCreating model (d_in={d_in})...")
    predictor = Predictor(
        d_in=d_in,
        d_out=train_dataset.y_dim,
        hidden=CLASSIFIER_CONFIG['hidden_dim'],
        dropout=CLASSIFIER_CONFIG['dropout']
    )

    mask_layer = MaskLayer(
        mask_size=train_dataset.x_dim * train_dataset.t,
        append=False
    )

    # Loss and metric (with class weights for imbalanced data)
    y_flat = train_dataset.y.flatten()
    y_valid = y_flat[y_flat != -1]
    class_counts = np.bincount(y_valid, minlength=train_dataset.y_dim)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * train_dataset.y_dim
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights.round(3)}")
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32),
        reduction='mean', ignore_index=-1
    )

    if train_dataset.y_dim == 2:
        val_metric = Accuracy(task='binary')
    else:
        val_metric = Accuracy(task='multiclass', num_classes=train_dataset.y_dim)

    # Create trainer module
    model = ClassifierTrainer(
        config=CLASSIFIER_CONFIG,
        predictor=predictor,
        mask_layer=mask_layer,
        num_time=train_dataset.t,
        num_feat=train_dataset.x_dim,
        loss_fn=loss_fn,
        val_metric=val_metric,
        num_aux=num_aux,
    )

    # Create PyTorch Lightning trainer
    print("\nTraining classifier...")
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=CLASSIFIER_CONFIG['epochs'],
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val/loss',
                patience=CLASSIFIER_CONFIG['patience'],
                mode='min'
            ),
            pl.callbacks.ModelCheckpoint(
                monitor='val/loss',
                dirpath=OUTPUT_FOLDER,
                filename='classifier-{epoch:02d}-{val/loss:.4f}',
                save_top_k=1,
                mode='min'
            )
        ],
        log_every_n_steps=10
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    torch.save({
        'predictor': predictor.state_dict(),
        'mask_layer': mask_layer.state_dict(),
        'config': CLASSIFIER_CONFIG,
        'num_time': train_dataset.t,
        'num_feat': train_dataset.x_dim,
        'num_aux': num_aux,
        'y_dim': train_dataset.y_dim
    }, CLASSIFIER_PATH)

    print(f"\nClassifier training complete!")
    print(f"Model saved to {CLASSIFIER_PATH}")


if __name__ == '__main__':
    main()
