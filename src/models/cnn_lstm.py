import torch
import torch.nn as nn
from typing import Tuple

from .encoder import CNNEncoder
from .decoder import AttentionDecoder

class CNNLSTMCaptioner(nn.Module):
    """
    Complete End-to-End Image Captioning System:
    Pretrained CNN Encoder + Bahdanau Visual Attention + LSTM Decoder.
    """
    def __init__(
        self,
        vocab_size: int,
        backbone: str = "resnet50",
        embed_size: int = 256,
        hidden_size: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.5,
        freeze_encoder: bool = True
    ):
        super(CNNLSTMCaptioner, self).__init__()
        self.encoder = CNNEncoder(
            backbone=backbone,
            embed_size=embed_size,
            freeze_encoder=freeze_encoder
        )
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            encoder_dim=self.encoder.encoder_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            images: (batch_size, 3, H, W)
            captions: (batch_size, max_len)
            lengths: (batch_size,) sequence lengths
            teacher_forcing_ratio: float
        Returns:
            predictions: (batch_size, max_len - 1, vocab_size)
            alphas: (batch_size, max_len - 1, num_pixels)
        """
        encoder_out = self.encoder(images)
        predictions, alphas = self.decoder(
            encoder_out=encoder_out,
            captions=captions,
            lengths=lengths,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        return predictions, alphas
