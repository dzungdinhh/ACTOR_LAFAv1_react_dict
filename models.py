"""
Neural network models for longitudinal feature acquisition
"""
import torch
import torch.nn as nn


class MaskLayer(nn.Module):
    """Layer for applying masks to features"""
    
    def __init__(self, mask_size, append=True):
        super().__init__()
        self.append = append
        self.mask_size = mask_size
    
    def forward(self, x, mask):
        """
        Args:
            x: Features (B, F)
            mask: Binary mask (B, F)
        Returns:
            Masked features (B, F) or (B, 2*F) if append=True
        """
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
        return out


class Predictor(nn.Module):
    """Prediction network for longitudinal outcomes"""
    
    def __init__(self, d_in, d_out, hidden=32, dropout=0.3):
        """
        Args:
            d_in: Input dimension (will add 1 for time indicator)
            d_out: Output dimension (number of classes)
            hidden: Hidden layer size
            dropout: Dropout probability
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_in + 1, hidden),  # +1 for time indicator
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features including time indicator (B, d_in + 1)
        Returns:
            logits: (B, d_out)
        """
        return self.model(x)


class GumbelSigmoid(nn.Module):
    """Gumbel-Sigmoid for differentiable binary sampling"""
    
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = float(tau)
    
    def _gumbel(self, shape, device):
        """Sample from Gumbel(0, 1)"""
        u = torch.rand(shape, device=device).clamp_(1e-8, 1-1e-8)
        return -torch.log(-torch.log(u))
    
    def forward(self, logits, hard=True):
        """
        Args:
            logits: (B, D) logits
            hard: If True, use straight-through estimator
        Returns:
            Binary-ish gates (B, D)
        """
        if self.training:
            g = self._gumbel(logits.shape, logits.device)
            z_soft = torch.sigmoid((logits + g) / self.tau)
        else:
            z_soft = torch.sigmoid(logits)
        
        if hard:
            z_hard = (z_soft > 0.5).float()
            return z_soft + (z_hard - z_soft).detach()
        return z_soft


class PlannerNet(nn.Module):
    """Network that outputs logits for feature acquisition"""
    
    def __init__(self, input_dim, out_dim, hidden=(512, 256, 128)):
        """
        Args:
            input_dim: Input dimension (x_masked + mask + time_emb)
            out_dim: Output dimension (number of features)
            hidden: Tuple of hidden layer sizes
        """
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Args:
            state: (B, input_dim) concatenation of [x_masked, mask, time_emb]
        Returns:
            logits: (B, out_dim) logits for each feature
        """
        return self.net(state)
