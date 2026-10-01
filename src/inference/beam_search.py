import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any

class BeamCandidate:
    """
    Represents a single hypothesis during beam search.
    """
    def __init__(
        self,
        tokens: List[int],
        log_prob: float,
        h: torch.Tensor,
        c: torch.Tensor,
        alphas: List[torch.Tensor]
    ):
        self.tokens = tokens
        self.log_prob = log_prob
        self.h = h
        self.c = c
        self.alphas = alphas

    def score(self, length_penalty: float = 0.7) -> float:
        """
        Length-normalized log-probability score to prevent bias towards short captions.
        Score = log_prob / (len(tokens) ** length_penalty)
        """
        length = len(self.tokens)
        if length == 0:
            return -float("inf")
        penalty = (length ** length_penalty)
        return self.log_prob / penalty

def beam_search_decoding(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    vocab: Any,
    beam_size: int = 3,
    max_length: int = 30,
    length_penalty: float = 0.7,
    device: torch.device = None
) -> Tuple[List[int], float, List[torch.Tensor]]:
    """
    Perform Beam Search Decoding to generate caption token sequence given an image tensor.
    Args:
        model: Trained CNN-LSTM Captioning Model
        image_tensor: (1, 3, H, W)
        vocab: Vocabulary instance
        beam_size: int beam width (k)
        max_length: max tokens to generate
        length_penalty: length normalization factor
        device: torch device
    Returns:
        best_tokens: List of token indices
        best_score: normalized log-prob score
        alphas: List of spatial attention tensors (1, num_pixels) for each generated token
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)  # (1, num_pixels, encoder_dim)
        h, c = model.decoder.init_hidden_state(encoder_out)

        start_candidate = BeamCandidate(
            tokens=[vocab.start_idx],
            log_prob=0.0,
            h=h,
            c=c,
            alphas=[]
        )

        completed_candidates: List[BeamCandidate] = []
        candidates: List[BeamCandidate] = [start_candidate]

        for step in range(max_length):
            new_candidates: List[BeamCandidate] = []

            for cand in candidates:
                # If candidate already ended with <end> token, keep in completed list
                if cand.tokens[-1] == vocab.end_idx:
                    completed_candidates.append(cand)
                    continue

                prev_word = torch.tensor([cand.tokens[-1]], device=device, dtype=torch.long)
                scores, new_h, new_c, alpha = model.decoder.forward_step(
                    prev_word, encoder_out, cand.h, cand.c
                )

                log_probs = F.log_softmax(scores, dim=1).squeeze(0)  # (vocab_size,)
                topk_log_probs, topk_indices = torch.topk(log_probs, k=beam_size)

                for k in range(beam_size):
                    token_idx = topk_indices[k].item()
                    token_log_prob = topk_log_probs[k].item()

                    new_cand = BeamCandidate(
                        tokens=cand.tokens + [token_idx],
                        log_prob=cand.log_prob + token_log_prob,
                        h=new_h,
                        c=new_c,
                        alphas=cand.alphas + [alpha.squeeze(0)]
                    )
                    new_candidates.append(new_cand)

            if not new_candidates:
                break

            # Sort new candidates by length-normalized score and retain top beam_size
            new_candidates.sort(key=lambda c: c.score(length_penalty), reverse=True)
            candidates = new_candidates[:beam_size]

            # Early stopping if beam_size candidates have already finished
            if len(completed_candidates) >= beam_size:
                break

        # Combine all candidates and find the single best candidate
        all_candidates = completed_candidates + candidates
        all_candidates.sort(key=lambda c: c.score(length_penalty), reverse=True)

        best_candidate = all_candidates[0]
        return best_candidate.tokens, best_candidate.score(length_penalty), best_candidate.alphas
