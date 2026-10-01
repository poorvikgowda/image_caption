import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Visual Attention Mechanism.
    Calculates attention weights over spatial image features based on the previous decoder hidden state.
    """
    def __init__(self, encoder_dim: int, hidden_size: int, attention_dim: int):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Linear projection for spatial features
        self.decoder_att = nn.Linear(hidden_size, attention_dim)  # Linear projection for decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)              # Linear layer to compute scalar energy
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)                          # Softmax across spatial locations (pixels)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        Args:
            encoder_out: spatial features (batch_size, num_pixels, encoder_dim)
            decoder_hidden: decoder hidden state (batch_size, hidden_size) or (1, batch_size, hidden_size)
        Returns:
            context_vector: weighted sum of features (batch_size, encoder_dim)
            alpha: attention weights (batch_size, num_pixels)
        """
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden.squeeze(0)

        # Project spatial features -> (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)
        
        # Project decoder hidden state -> (batch_size, 1, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        
        # Additive attention score -> (batch_size, num_pixels)
        energy = self.full_att(self.relu(att1 + att2)).squeeze(-1)
        
        # Calculate attention weights (batch_size, num_pixels)
        alpha = self.softmax(energy)
        
        # Context vector = sum_i(alpha_i * encoder_out_i) -> (batch_size, encoder_dim)
        context_vector = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)
        
        return context_vector, alpha
