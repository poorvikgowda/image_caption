import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from .encoder import CNNEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CNNTransformerCaptioner(nn.Module):
    """
    Modern Encoder-Decoder Architecture for Image Captioning:
    CNN Visual Feature Extractor + Transformer Decoder with Cross-Attention.
    """
    def __init__(
        self,
        vocab_size: int,
        backbone: str = "resnet50",
        embed_size: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        freeze_encoder: bool = True
    ):
        super(CNNTransformerCaptioner, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # Visual Encoder
        self.encoder = CNNEncoder(backbone=backbone, freeze_encoder=freeze_encoder)
        self.visual_projection = nn.Linear(self.encoder.encoder_dim, embed_size)

        # Word Embedding & Positional Encoding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(d_model=embed_size, dropout=dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )

        # Final projection to vocabulary logits
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Causal mask for decoder to prevent looking at future tokens.
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        pad_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: (batch_size, 3, H, W)
            captions: (batch_size, seq_len)
        Returns:
            predictions: (batch_size, seq_len - 1, vocab_size)
            dummy_alphas: (batch_size, seq_len - 1, num_pixels) for compatibility
        """
        batch_size = images.size(0)
        device = images.device

        # Extract visual features -> (batch_size, num_pixels, embed_size)
        visual_feats = self.encoder(images)
        memory = self.visual_projection(visual_feats)

        # Decoder input sequence (exclude the last token for teacher forcing prediction)
        tgt_input = captions[:, :-1]
        tgt_seq_len = tgt_input.size(1)

        # Embed target tokens + add positional encoding
        tgt_embed = self.pos_encoder(self.embedding(tgt_input) * math.sqrt(self.embed_size))

        # Causal mask and padding mask
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device=device)
        tgt_key_padding_mask = (tgt_input == pad_idx)

        # Run Transformer Decoder
        decoder_output = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        predictions = self.fc_out(decoder_output)

        # Dummy attention weights for output compatibility
        num_pixels = memory.size(1)
        alphas = torch.zeros(batch_size, tgt_seq_len, num_pixels, device=device)

        return predictions, alphas
