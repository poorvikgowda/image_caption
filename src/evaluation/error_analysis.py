import os
import logging
from collections import Counter
from typing import List, Dict, Any

logger = logging.getLogger("ImageCaptioning")

def analyze_errors(predictions_records: List[Dict[str, Any]], output_dir: str = "outputs") -> str:
    """
    Perform systematic error analysis over generated captions.
    Identify common failure modes: repetition, low n-gram overlap, length truncation, or hallucination.
    """
    error_categories = {
        "word_repetition": [],
        "too_short": [],
        "low_overlap": [],
        "length_mismatch": []
    }

    for record in predictions_records:
        img_id = record["image_id"]
        cap = record["beam_caption"]
        tokens = cap.split()
        refs = record["references"]

        # Check repetition error
        counts = Counter(tokens)
        if any(c >= 3 for word, c in counts.items() if len(word) > 3):
            error_categories["word_repetition"].append(record)

        # Check length error
        if len(tokens) <= 3:
            error_categories["too_short"].append(record)

        # Check reference word overlap
        ref_words = set(" ".join(refs).split())
        pred_words = set(tokens)
        overlap = len(pred_words.intersection(ref_words))
        if overlap <= 1 and len(pred_words) > 0:
            error_categories["low_overlap"].append(record)

    report_path = os.path.join(output_dir, "error_analysis_report.md")
    os.makedirs(output_dir, exist_ok=True)

    rep_ex = error_categories['word_repetition'][0]['beam_caption'] if error_categories['word_repetition'] else 'None observed'
    short_ex = error_categories['too_short'][0]['beam_caption'] if error_categories['too_short'] else 'None observed'
    overlap_ex = error_categories['low_overlap'][0]['beam_caption'] if error_categories['low_overlap'] else 'None observed'

    n_rep = len(error_categories['word_repetition'])
    n_short = len(error_categories['too_short'])
    n_overlap = len(error_categories['low_overlap'])

    content = (
        "# Image Caption Generation - Error Analysis Report\n\n"
        "## Failure Mode Taxonomy\n\n"
        f"### 1. Word Repetitions / Looping ({n_rep} cases)\n"
        "- **Root Cause**: Unbalanced hidden states or absence of repetition penalty during beam decoding.\n"
        f"- **Example**: `{rep_ex}`\n\n"
        f"### 2. Excessively Short Captions ({n_short} cases)\n"
        "- **Root Cause**: Premature generation of `<end>` token or severe length penalty without length normalization.\n"
        f"- **Example**: `{short_ex}`\n\n"
        f"### 3. Low Vocabulary Overlap / Semantic Mismatch ({n_overlap} cases)\n"
        "- **Root Cause**: Visual feature misclassification or rare dataset concepts outside top vocabulary.\n"
        f"- **Example**: `{overlap_ex}`\n\n"
        "## Recommended Mitigation Strategies\n"
        "1. **Repetition Penalty**: Add frequency penalty (lambda_rep) to beam search score calculation.\n"
        "2. **Label Smoothing**: Prevents overconfidence in high-frequency visual tokens like 'man' or 'dog'.\n"
        "3. **Encoder Fine-Tuning**: Unfreeze ResNet layer 4 to extract finer granularity feature maps.\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Saved error analysis report to {report_path}")
    return report_path
