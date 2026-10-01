import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import pandas as pd
from typing import List, Dict, Any

from src.inference.generate import CaptionGenerator
from src.data.vocabulary import Vocabulary

def generate_qualitative_examples(
    model: torch.nn.Module,
    vocab: Vocabulary,
    test_df: pd.DataFrame,
    image_dir: str,
    device: torch.device,
    save_dir: str = "outputs/qualitative",
    num_samples: int = 15
) -> List[str]:
    """
    Generate side-by-side qualitative comparison cards for test set images.
    """
    os.makedirs(save_dir, exist_ok=True)
    generator = CaptionGenerator(model=model, vocab=vocab, device=device)

    unique_imgs = test_df["image_id"].unique()[:num_samples]
    output_paths = []

    for idx, img_id in enumerate(unique_imgs):
        img_path = os.path.join(image_dir, img_id)
        if not os.path.exists(img_path):
            continue

        # Get ground truth captions for this image
        gt_captions = test_df[test_df["image_id"] == img_id]["caption"].tolist()

        # Generate Greedy and Beam Search captions
        greedy_res = generator.generate_greedy(img_path)
        beam_res = generator.generate_beam_search(img_path, beam_size=3)

        pil_img = Image.open(img_path).convert("RGB")

        # Create qualitative visual card
        fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.3]})

        ax_img.imshow(pil_img)
        ax_img.set_title(f"Test Image: {img_id}", fontsize=12, fontweight="bold")
        ax_img.axis("off")

        # Text information panel
        txt_content = f"GROUND TRUTH CAPTIONS:\n"
        for i, gt in enumerate(gt_captions[:3], 1):
            txt_content += f"  {i}. {gt}\n"

        txt_content += f"\nGENERATED CAPTIONS:\n"
        txt_content += f"  • Greedy: \"{greedy_res['caption']}\"\n"
        txt_content += f"  • Beam Search (k=3): \"{beam_res['caption']}\" (Score: {beam_res['score']:.3f})\n"

        ax_txt.text(0.05, 0.95, txt_content, transform=ax_txt.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f8", alpha=0.9))
        ax_txt.axis("off")

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"sample_{idx+1:02d}_{img_id}")
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        output_paths.append(out_path)

    return output_paths
