import pytest
import torch
from PIL import Image

from src.data.vocabulary import Vocabulary
from src.models.cnn_lstm import CNNLSTMCaptioner
from src.inference.generate import CaptionGenerator

def test_inference_decoding_strategies():
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocabulary([["a", "dog", "runs", "fast"]])

    model = CNNLSTMCaptioner(vocab_size=len(vocab), embed_size=64, hidden_size=128, attention_dim=64)
    device = torch.device("cpu")

    generator = CaptionGenerator(model=model, vocab=vocab, device=device, max_length=10)

    # Synthetic image
    img = Image.new("RGB", (224, 224), color="red")

    # Greedy Decoding test
    greedy_res = generator.generate_greedy(img)
    assert "caption" in greedy_res
    assert isinstance(greedy_res["caption"], str)
    assert len(greedy_res["tokens"]) <= 11

    # Beam Search Decoding test
    beam_res = generator.generate_beam_search(img, beam_size=3)
    assert "caption" in beam_res
    assert isinstance(beam_res["caption"], str)
    assert "score" in beam_res
