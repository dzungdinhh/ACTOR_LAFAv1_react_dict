"""
Training the prediction network (classifier) on longitudinal data
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models import Predictor, MaskLayer
from utils import generate_uniform_mask, generate_bernoulli_mask
from config import MASK_TYPE


class ClassifierTrainer(pl.LightningModule):
    """PyTorch Lightning module for training the classifier"""

    def __init__(self, config, predictor, mask_layer, num_time, num_feat,
                 loss_fn, val_metric, num_aux=0):
        """
        Args:
            config: Training configuration dict
            predictor: Prediction network
            mask_layer: MaskLayer instance
            num_time: Number of timesteps
            num_feat: Number of features per timestep
            loss_fn: Loss function
            val_metric: Validation metric function
            num_aux: Number of auxiliary/static features (0 = no aux)
        """
        super().__init__()
        self.config = config
        self.predictor = predictor
        self.mask_layer = mask_layer
        self.mask_size = self.mask_layer.mask_size
        self.num_time = num_time
        self.num_feat = num_feat
        self.num_aux = num_aux
        self.loss_fn = loss_fn
        self.val_metric = val_metric

        self.mask_type = MASK_TYPE
        self.num_bad_epochs = 0
        self.validation_step_outputs = []

    def load_batch(self, batch):
        """Prepare batch data"""
        if len(batch) == 5:
            x, y, m, x_static, mask_static = batch
            x_static = torch.nan_to_num(x_static).float()
            mask_static = mask_static.float()
        else:
            x, y, m = batch
            x_static = None
            mask_static = None

        x = torch.nan_to_num(x)
        x = x.reshape(len(x), -1)  # [B, T, d] -> [B, T*d]
        y = y.reshape(-1)  # [B, T] -> [B*T]
        m = m.reshape(len(m), -1)  # [B, T, d] -> [B, T*d]
        return x, y, m, x_static, mask_static

    def _generate_mask(self, batch_size, num_features, num_time=None):
        """Dispatch to configured mask generator."""
        if self.mask_type == 'bernoulli':
            return generate_bernoulli_mask(batch_size, num_features, num_time=num_time)
        return generate_uniform_mask(batch_size, num_features, num_time=num_time)

    def longitudinal_prediction(self, x, mask, x_static=None, mask_static=None):
        """
        Make predictions at each timestep using masked features

        Args:
            x: Features (B, T*d)
            mask: Acquisition mask (B, T*d)
            x_static: Static features (B, num_aux) or None
            mask_static: Static availability mask (B, num_aux) or None

        Returns:
            pred: Predictions (B, T, num_classes)
        """
        # Prepare aux features if present
        aux_acquired = None
        if x_static is not None and self.num_aux > 0:
            # Random masking for aux during classifier training
            aux_mask = self._generate_mask(len(x), self.num_aux).to(self.device)
            aux_mask = aux_mask * mask_static  # can't acquire unavailable features
            aux_acquired = x_static * aux_mask  # (B, num_aux)

        pred = []
        for t in range(self.num_time):
            # Mask out future timepoints
            m_t = mask.clone()
            m_t[:, (t + 1) * self.num_feat:] = 0

            # Apply mask
            x_t_masked = self.mask_layer(x, m_t)

            # Add time indicator
            t_indicator = torch.full(
                (len(x),), (t + 1) / self.num_time, device=self.device
            ).unsqueeze(1)

            # Build predictor input
            if aux_acquired is not None:
                x_in = torch.cat((t_indicator, x_t_masked, aux_acquired), dim=1)
            else:
                x_in = torch.cat((t_indicator, x_t_masked), dim=1)

            # Predict
            pred.append(self.predictor(x_in))

        pred = torch.stack(pred, dim=1)  # [B, T, num_classes]
        return pred

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y, m, x_static, mask_static = self.load_batch(batch)

        # Random masking for pretraining
        m_random = self._generate_mask(len(x), self.mask_size).to(self.device)

        # Predict
        pred = self.longitudinal_prediction(
            x, m_random * m, x_static=x_static, mask_static=mask_static
        )
        pred = pred.reshape(-1, pred.shape[-1])

        # Filter out invalid labels
        valid_mask = (y != -1)
        pred = pred[valid_mask]
        y = y[valid_mask]

        # Compute loss
        loss = self.loss_fn(pred, y)

        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y, m, x_static, mask_static = self.load_batch(batch)

        # Random masking
        m_random = self._generate_mask(len(x), self.mask_size).to(self.device)

        # Predict
        pred = self.longitudinal_prediction(
            x, m_random * m, x_static=x_static, mask_static=mask_static
        )
        pred = pred.reshape(-1, pred.shape[-1])

        # Filter out invalid labels
        valid_mask = (y != -1)
        pred = pred[valid_mask]
        y = y[valid_mask]

        self.validation_step_outputs.append((pred, y))
        return pred, y

    def on_validation_epoch_end(self):
        """Aggregate validation results"""
        outputs = self.validation_step_outputs
        pred_list, y_list = zip(*outputs)
        pred = torch.cat(pred_list)
        y = torch.cat(y_list)

        # Compute loss
        loss = self.loss_fn(pred, y)

        # Compute metric (handle binary vs multiclass)
        if pred.shape[1] == 2:
            val_perf = self.val_metric(pred[:, 1], y)
        else:
            val_perf = self.val_metric(pred, y)

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/metric', val_perf, prog_bar=True)

        # Early stopping check
        sch = self.lr_schedulers()
        if loss < sch.best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        patience = self.config.get('patience', 10)
        if self.num_bad_epochs > patience:
            self.trainer.should_stop = True

        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.config['lr']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.2,
            patience=self.config.get('patience', 10),
            min_lr=self.config.get('min_lr', 1e-6)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
