import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptionLoss(nn.Module):
    """
    Masked Cross Entropy Loss with optional Attention Regularization and Label Smoothing.
    Ignores padding tokens (<pad>).
    """
    def __init__(self, pad_idx: int, alpha_c: float = 1.0, label_smoothing: float = 0.0):
        super(CaptionLoss, self).__init__()
        self.pad_idx = pad_idx
        self.alpha_c = alpha_c  # Doubly stochastic attention regularization parameter
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=label_smoothing,
            reduction="mean"
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        alphas: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: (batch_size, seq_len - 1, vocab_size)
            targets: (batch_size, seq_len) -- target token indices (starts with <start>, targets start at index 1)
            alphas: (batch_size, seq_len - 1, num_pixels) optional attention weights
        Returns:
            loss: scalar tensor
        """
        # Target tokens for prediction correspond to index 1 onwards
        targets_slice = targets[:, 1:].contiguous()

        # Reshape for CrossEntropyLoss: (batch_size * (seq_len - 1), vocab_size) vs (batch_size * (seq_len - 1),)
        batch_size, seq_len, vocab_size = predictions.shape
        predictions_flat = predictions.view(-1, vocab_size)
        targets_flat = targets_slice.view(-1)

        loss = self.criterion(predictions_flat, targets_flat)

        # Doubly stochastic attention regularization penalty: sum_t(alpha_t) approx 1
        if alphas is not None and self.alpha_c > 0 and alphas.sum() > 0:
            att_reg = self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            loss = loss + att_reg

        return loss
