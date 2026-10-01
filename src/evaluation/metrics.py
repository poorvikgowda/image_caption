import math
from collections import Counter
from typing import List, Dict, Any
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

# Ensure nltk resources exist safely
try:
    nltk.download('wordnet', quiet=True)
    from nltk.translate.meteor_score import meteor_score
except Exception:
    meteor_score = None

def compute_bleu_scores(
    references: List[List[List[str]]],
    hypotheses: List[List[str]]
) -> Dict[str, float]:
    """
    Compute Corpus BLEU-1, BLEU-2, BLEU-3, BLEU-4 using NLTK SmoothingFunction.
    Args:
        references: List of reference token lists per image, e.g. [[[ref1_w1, ref1_w2], [ref2_w1, ref2_w2]], ...]
        hypotheses: List of predicted token lists per image, e.g. [[pred_w1, pred_w2], ...]
    """
    smooth = SmoothingFunction().method1
    b1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    b2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    b3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    b4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    return {
        "BLEU-1": float(b1),
        "BLEU-2": float(b2),
        "BLEU-3": float(b3),
        "BLEU-4": float(b4)
    }

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence between two token lists."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def compute_rouge_l(references: List[List[List[str]]], hypotheses: List[List[str]]) -> float:
    """
    Compute average ROUGE-L score based on Longest Common Subsequence precision/recall.
    """
    scores = []
    for refs, hyp in zip(references, hypotheses):
        if not hyp:
            scores.append(0.0)
            continue
        max_r_score = 0.0
        for ref in refs:
            if not ref:
                continue
            lcs = _lcs_length(hyp, ref)
            rec = lcs / len(ref)
            prec = lcs / len(hyp)
            if rec + prec > 0:
                f1 = (2 * rec * prec) / (rec + prec)
            else:
                f1 = 0.0
            if f1 > max_r_score:
                max_r_score = f1
        scores.append(max_r_score)
    return float(sum(scores) / max(1, len(scores)))

def compute_meteor(references: List[List[List[str]]], hypotheses: List[List[str]]) -> float:
    """
    Compute average METEOR score.
    Uses NLTK meteor_score if wordnet available, else exact word match F1 fallback.
    """
    scores = []
    for refs, hyp in zip(references, hypotheses):
        if meteor_score is not None:
            try:
                score = meteor_score(refs, hyp)
                scores.append(score)
                continue
            except Exception:
                pass
        
        # Word overlap harmonic F1 fallback
        max_f1 = 0.0
        hyp_set = set(hyp)
        for ref in refs:
            ref_set = set(ref)
            intersection = hyp_set.intersection(ref_set)
            if not intersection:
                continue
            prec = len(intersection) / max(1, len(hyp))
            rec = len(intersection) / max(1, len(ref))
            f1 = (2 * prec * rec) / max(1e-6, (prec + rec))
            if f1 > max_f1:
                max_f1 = f1
        scores.append(max_f1)

    return float(sum(scores) / max(1, len(scores)))

def compute_cider(references: List[List[List[str]]], hypotheses: List[List[str]], n: int = 4) -> float:
    """
    Compute CIDEr (Consensus-based Image Description Evaluation) score using TF-IDF n-gram matches.
    """
    # Count n-grams for references and hypotheses
    def get_ngrams(tokens: List[str], n_val: int) -> Counter:
        return Counter([tuple(tokens[i:i + n_val]) for i in range(len(tokens) - n_val + 1)])

    cider_scores = []
    for refs, hyp in zip(references, hypotheses):
        n_scores = []
        for n_i in range(1, n + 1):
            hyp_ngrams = get_ngrams(hyp, n_i)
            if not hyp_ngrams:
                n_scores.append(0.0)
                continue

            ref_ngrams_list = [get_ngrams(r, n_i) for r in refs]

            # Compute TF-IDF match with references
            score_i = 0.0
            for ref_ngrams in ref_ngrams_list:
                if not ref_ngrams:
                    continue
                common = hyp_ngrams & ref_ngrams
                num_common = sum(common.values())
                score_i += num_common / max(1.0, math.sqrt(sum(hyp_ngrams.values()) * sum(ref_ngrams.values())))
            score_i /= max(1, len(refs))
            n_scores.append(score_i)

        cider_scores.append(sum(n_scores) / n * 10.0)

    return float(sum(cider_scores) / max(1, len(cider_scores)))

def compute_all_metrics(
    references: List[List[List[str]]],
    hypotheses: List[List[str]]
) -> Dict[str, float]:
    """
    Compute all standard Image Captioning metrics.
    """
    bleu_dict = compute_bleu_scores(references, hypotheses)
    rouge_l = compute_rouge_l(references, hypotheses)
    meteor = compute_meteor(references, hypotheses)
    cider = compute_cider(references, hypotheses)

    metrics = {
        **bleu_dict,
        "METEOR": round(meteor, 4),
        "ROUGE-L": round(rouge_l, 4),
        "CIDEr": round(cider, 4)
    }
    return metrics
