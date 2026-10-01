import os
import argparse
import yaml
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.data.preprocessing import parse_captions_file, split_dataset
from src.data.vocabulary import Vocabulary

logger = setup_logger()

SYNTHETIC_CAPTIONS = [
    "A dog running through a green grassy field.",
    "A man riding a bicycle along a city street.",
    "A child playing with a red ball outdoors.",
    "A woman sitting on a bench in a quiet park.",
    "A cat resting on a soft blue cushion inside.",
    "A person standing on top of a snowy mountain peak.",
    "A group of friends laughing near a campfire at night.",
    "A cute puppy sleeping peacefully on the floor.",
    "An airplane flying high in the clear blue sky.",
    "A chef preparing delicious food in a busy kitchen."
]

def create_synthetic_dataset(data_dir: str, num_images: int = 50):
    """
    Generate synthetic images and captions for testing, CI/CD, and pipeline verification.
    """
    logger.info(f"Generating synthetic dataset with {num_images} images in {data_dir}...")
    img_dir = os.path.join(data_dir, "Images")
    os.makedirs(img_dir, exist_ok=True)
    captions_file = os.path.join(data_dir, "captions.txt")

    records = []

    for i in range(num_images):
        img_name = f"synthetic_{i:04d}.jpg"
        img_path = os.path.join(img_dir, img_name)

        # Generate a colorful synthetic pattern image (256x256)
        color = (
            (i * 37) % 256,
            (i * 73) % 256,
            (i * 109) % 256
        )
        img = Image.new("RGB", (256, 256), color=color)
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 200], fill=((color[0]+100)%256, (color[1]+100)%256, (color[2]+100)%256))
        draw.ellipse([80, 80, 170, 170], fill=((color[0]+180)%256, (color[1]+50)%256, (color[2]+200)%256))
        img.save(img_path)

        # Assign 5 captions per image
        for cap_idx in range(5):
            base_cap = SYNTHETIC_CAPTIONS[(i + cap_idx) % len(SYNTHETIC_CAPTIONS)]
            records.append({"image_id": img_name, "caption": base_cap})

    # Write captions.txt
    df = pd.DataFrame(records)
    df.to_csv(captions_file, index=False)
    logger.info(f"Successfully created synthetic dataset at {data_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset and build vocabulary.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dummy dataset if real dataset is not found")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed_everything(config["experiment"]["seed"])

    image_dir = config["dataset"]["image_dir"]
    captions_file = config["dataset"]["captions_file"]
    processed_dir = config["dataset"]["processed_dir"]
    data_dir = config["dataset"]["data_dir"]

    # Auto-detect real captions file location (e.g. data/flickr8k/Images/captions.txt or data/flickr8k/captions.txt)
    candidates = [
        os.path.join(data_dir, "Images", "captions.txt"),
        os.path.join(data_dir, "captions.txt"),
        os.path.join(data_dir, "captions.csv"),
    ]
    
    real_captions_found = None
    for cand in candidates:
        if os.path.exists(cand):
            with open(cand, "r", encoding="utf-8", errors="ignore") as f:
                line_count = len(f.readlines())
            if line_count > 1000:  # Real Flickr8k has ~40,457 lines
                real_captions_found = cand
                logger.info(f"Found real Flickr8k dataset with {line_count} captions at {cand}")
                break

    if real_captions_found:
        captions_file = real_captions_found
        image_dir = os.path.join(data_dir, "Images")
    elif not (os.path.exists(captions_file) and os.path.exists(image_dir)):
        logger.warning(f"Real dataset not found at {data_dir}. Creating synthetic dataset for execution/testing...")
        create_synthetic_dataset(data_dir, num_images=50)
        image_dir = os.path.join(data_dir, "Images")
        captions_file = os.path.join(data_dir, "captions.txt")

    # Parse captions
    df = parse_captions_file(captions_file)

    # Split dataset
    splits = config["dataset"]["splits"]
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=splits["train_ratio"],
        val_ratio=splits["val_ratio"],
        test_ratio=splits["test_ratio"],
        seed=config["experiment"]["seed"]
    )

    # Save processed split CSVs
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    logger.info(f"Saved split CSVs to {processed_dir}")

    # Build Vocabulary ONLY from training set captions to avoid data leakage
    tokenized_train_captions = [cap.split() for cap in train_df["caption"].tolist()]
    vocab = Vocabulary(min_freq=config["dataset"]["min_freq"])
    vocab.build_vocabulary(tokenized_train_captions)

    vocab_path = os.path.join(processed_dir, "vocab.json")
    vocab.save(vocab_path)
    logger.info(f"Built vocabulary with size {len(vocab)} (min_freq={vocab.min_freq}) and saved to {vocab_path}")

if __name__ == "__main__":
    main()
