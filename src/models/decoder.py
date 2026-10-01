import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .attention import BahdanauAttention

class AttentionDecoder(nn.Module):
    """
    Attention-based LSTM Decoder for Image Captioning.
    Generates word sequence step-by-step conditioned on visual features and previous word.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        encoder_dim: int = 2048,
        attention_dim: int = 256,
        dropout: float = 0.5
    ):
        super(AttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.encoder_dim = encoder_dim

        # Word Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

        # Visual Attention
        self.attention = BahdanauAttention(
            encoder_dim=encoder_dim,
            hidden_size=hidden_size,
            attention_dim=attention_dim
        )

        # Initial hidden state and cell state linear maps from mean spatial features
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)

        # Sigmoid attention gate (to dynamically balance context vector)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # LSTM Cell: input is [word_embed, gated_context]
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)

        # Linear output layer to vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM (h_0, c_0) from the spatial mean of encoder outputs.
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)
        h = torch.tanh(self.init_h(mean_encoder_out))
        c = torch.tanh(self.init_c(mean_encoder_out))
        return h, c

    def forward_step(
        self,
        prev_word: torch.Tensor,
        encoder_out: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute single decoding step.
        Args:
            prev_word: (batch_size,) integer word indices
            encoder_out: (batch_size, num_pixels, encoder_dim)
            h: previous hidden state (batch_size, hidden_size)
            c: previous cell state (batch_size, hidden_size)
        Returns:
            scores: (batch_size, vocab_size) prediction logits
            h: updated hidden state
            c: updated cell state
            alpha: attention weights (batch_size, num_pixels)
        """
        # 1. Embed previous word: (batch_size, embed_size)
        embeddings = self.embedding(prev_word)

        # 2. Compute visual attention context vector and alpha weights
        context, alpha = self.attention(encoder_out, h)

        # 3. Gated context
        gate = self.sigmoid(self.f_beta(h))
        gated_context = gate * context

        # 4. Pass through LSTM cell
        lstm_input = torch.cat([embeddings, gated_context], dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))

        # 5. Project to vocabulary logits
        scores = self.fc(self.dropout(h))

        return scores, h, c, alpha

    def forward(
        self,
        encoder_out: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass over sequence of length max_seq_len.
        Args:
            encoder_out: (batch_size, num_pixels, encoder_dim)
            captions: (batch_size, max_seq_len) ground truth caption indices
            lengths: (batch_size,) actual sequence lengths
            teacher_forcing_ratio: float probability of feeding ground truth token vs prediction
        Returns:
            predictions: (batch_size, max_seq_len - 1, vocab_size)
            alphas: (batch_size, max_seq_len - 1, num_pixels)
        """
        batch_size = encoder_out.size(0)
        max_seq_len = captions.size(1)

        h, c = self.init_hidden_state(encoder_out)

        # We predict tokens starting from position 1 to max_seq_len - 1
        predictions = torch.zeros(batch_size, max_seq_len - 1, self.vocab_size, device=encoder_out.device)
        alphas = torch.zeros(batch_size, max_seq_len - 1, encoder_out.size(1), device=encoder_out.device)

        # Input at step 0 is <start> token (column 0 of captions)
        word = captions[:, 0]

        for t in range(max_seq_len - 1):
            scores, h, c, alpha = self.forward_step(word, encoder_out, h, c)
            predictions[:, t, :] = scores
            alphas[:, t, :] = alpha

            # Teacher forcing logic
            use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio) if self.training else True
            if use_teacher_forcing:
                word = captions[:, t + 1]
            else:
                word = scores.argmax(dim=1)

        return predictions, alphas
