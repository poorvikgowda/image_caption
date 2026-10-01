import pytest
import torch

from src.models.encoder import CNNEncoder
from src.models.attention import BahdanauAttention
from src.models.decoder import AttentionDecoder
from src.models.cnn_lstm import CNNLSTMCaptioner
from src.models.transformer_decoder import CNNTransformerCaptioner

def test_cnn_encoder_forward():
    encoder = CNNEncoder(backbone="resnet50", freeze_encoder=True)
    images = torch.randn(2, 3, 224, 224)
    features = encoder(images)
    assert features.shape == (2, 49, 2048)

def test_bahdanau_attention():
    att = BahdanauAttention(encoder_dim=2048, hidden_size=512, attention_dim=256)
    encoder_out = torch.randn(2, 49, 2048)
    h = torch.randn(2, 512)
    context, alpha = att(encoder_out, h)
    assert context.shape == (2, 2048)
    assert alpha.shape == (2, 49)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(2), atol=1e-5)

def test_cnn_lstm_forward():
    model = CNNLSTMCaptioner(vocab_size=100, embed_size=128, hidden_size=256, attention_dim=128)
    images = torch.randn(2, 3, 224, 224)
    captions = torch.randint(0, 100, (2, 10))
    lengths = torch.tensor([10, 8])

    predictions, alphas = model(images, captions, lengths)
    assert predictions.shape == (2, 9, 100)
    assert alphas.shape == (2, 9, 49)

def test_cnn_transformer_forward():
    model = CNNTransformerCaptioner(vocab_size=100, embed_size=128, nhead=4, num_decoder_layers=2)
    images = torch.randn(2, 3, 224, 224)
    captions = torch.randint(0, 100, (2, 10))

    predictions, alphas = model(images, captions)
    assert predictions.shape == (2, 9, 100)
