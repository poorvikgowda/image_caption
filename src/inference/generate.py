import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, Tuple, List, Dict, Any

from src.data.vocabulary import Vocabulary
from src.data.transforms import get_transforms
from src.inference.beam_search import beam_search_decoding
from src.training.checkpoint import load_checkpoint
from src.models.cnn_lstm import CNNLSTMCaptioner

class CaptionGenerator:
    """
    High-level API for generating captions from raw images using Greedy Decoding or Beam Search.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        vocab: Vocabulary,
        device: torch.device,
        max_length: int = 30
    ):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
        self.max_length = max_length
        self.transform = get_transforms(split="val")

    def _load_image(self, image_input: Union[str, Image.Image]) -> Tuple[Image.Image, torch.Tensor]:
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path string or PIL Image object.")

        img_tensor = self.transform(pil_img).unsqueeze(0)  # (1, 3, H, W)
        return pil_img, img_tensor

    @torch.no_grad()
    def generate_greedy(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Generate caption using Greedy (Argmax) Decoding.
        """
        pil_img, img_tensor = self._load_image(image_input)
        img_tensor = img_tensor.to(self.device)

        if hasattr(self.model, "encoder"):
            encoder_out = self.model.encoder(img_tensor)
            h, c = self.model.decoder.init_hidden_state(encoder_out)
        else:
            raise NotImplementedError("Greedy decoding for non-CNN-LSTM model using fallback.")

        tokens = [self.vocab.start_idx]
        alphas = []
        word = torch.tensor([self.vocab.start_idx], device=self.device)

        for t in range(self.max_length):
            scores, h, c, alpha = self.model.decoder.forward_step(word, encoder_out, h, c)
            predicted_idx = scores.argmax(dim=1).item()
            
            tokens.append(predicted_idx)
            alphas.append(alpha.squeeze(0))

            if predicted_idx == self.vocab.end_idx:
                break

            word = torch.tensor([predicted_idx], device=self.device)

        caption = self.vocab.decode_to_sentence(tokens, remove_specials=True)
        return {
            "caption": caption,
            "tokens": tokens,
            "alphas": alphas,
            "image": pil_img,
            "decoding": "greedy"
        }

    def generate_beam_search(
        self,
        image_input: Union[str, Image.Image],
        beam_size: int = 3,
        length_penalty: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate caption using Beam Search Decoding.
        """
        pil_img, img_tensor = self._load_image(image_input)
        tokens, score, alphas = beam_search_decoding(
            model=self.model,
            image_tensor=img_tensor,
            vocab=self.vocab,
            beam_size=beam_size,
            max_length=self.max_length,
            length_penalty=length_penalty,
            device=self.device
        )
        caption = self.vocab.decode_to_sentence(tokens, remove_specials=True)
        return {
            "caption": caption,
            "tokens": tokens,
            "score": score,
            "alphas": alphas,
            "image": pil_img,
            "decoding": f"beam_search (k={beam_size})"
        }
