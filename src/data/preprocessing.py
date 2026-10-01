import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger("ImageCaptioning")

def clean_caption(caption: str) -> str:
    """
    Clean text: lowercase, remove special characters except basic punctuation, format whitespace.
    """
    if not isinstance(caption, str):
        return ""
    caption = caption.lower()
    # Replace any sequence of non-alphanumeric (except spaces) with space or remove unwanted punctuation
    caption = re.sub(r"[^\w\s]", "", caption)
    # Remove single character words like 'a' or 'i' except valid ones if needed, or keep clean words
    words = [w for w in caption.strip().split() if len(w) > 0 and w.isalpha()]
    return " ".join(words)

def parse_captions_file(captions_file: str) -> pd.DataFrame:
    """
    Parse Flickr8k/Flickr30k captions text/csv file into a standardized DataFrame.
    DataFrame schema: ['image_id', 'caption']
    """
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found at {captions_file}")

    records = []
    with open(captions_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Check header
    first_line = lines[0].strip()
    start_idx = 0
    if "image" in first_line.lower() and "caption" in first_line.lower():
        start_idx = 1

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        # Support CSV comma separation or tab/space separation
        if "," in line:
            parts = line.split(",", 1)
        elif "\t" in line:
            parts = line.split("\t", 1)
        else:
            parts = line.split(" ", 1)

        if len(parts) < 2:
            continue

        image_id = parts[0].strip()
        caption = parts[1].strip()

        # Handle image_name#0 format (e.g. 1000268201_693b08cb0e.jpg#0)
        if "#" in image_id:
            image_id = image_id.split("#")[0]

        cleaned_cap = clean_caption(caption)
        if cleaned_cap:
            records.append({"image_id": image_id, "caption": cleaned_cap})

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} captions across {df['image_id'].nunique()} unique images from {captions_file}.")
    return df

def split_dataset(
    df: pd.DataFrame, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset deterministically by UNIQUE IMAGE ID to avoid data leakage.
    """
    unique_images = np.array(df["image_id"].unique())
    np.random.seed(seed)
    np.random.shuffle(unique_images)

    n_total = len(unique_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_imgs = set(unique_images[:n_train])
    val_imgs = set(unique_images[n_train:n_train + n_val])
    test_imgs = set(unique_images[n_train + n_val:])

    train_df = df[df["image_id"].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df["image_id"].isin(val_imgs)].reset_index(drop=True)
    test_df = df[df["image_id"].isin(test_imgs)].reset_index(drop=True)

    logger.info(f"Split results: Train={len(train_imgs)} imgs ({len(train_df)} caps), "
                f"Val={len(val_imgs)} imgs ({len(val_df)} caps), "
                f"Test={len(test_imgs)} imgs ({len(test_df)} caps)")

    return train_df, val_df, test_df
