import pytest
import torch
import pandas as pd

from src.data.preprocessing import clean_caption, split_dataset
from src.data.vocabulary import Vocabulary
from src.data.dataset import caption_collate_fn

def test_clean_caption():
    raw = "A dog running in the grass!!!"
    cleaned = clean_caption(raw)
    assert cleaned == "a dog running in the grass"

def test_split_dataset_no_leakage():
    data = [
        {"image_id": "img1.jpg", "caption": "cap 1"},
        {"image_id": "img1.jpg", "caption": "cap 2"},
        {"image_id": "img2.jpg", "caption": "cap 3"},
        {"image_id": "img3.jpg", "caption": "cap 4"},
        {"image_id": "img4.jpg", "caption": "cap 5"},
        {"image_id": "img5.jpg", "caption": "cap 6"}
    ]
    df = pd.DataFrame(data)
    train_df, val_df, test_df = split_dataset(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    train_imgs = set(train_df["image_id"])
    val_imgs = set(val_df["image_id"])
    test_imgs = set(test_df["image_id"])

    assert len(train_imgs.intersection(val_imgs)) == 0
    assert len(train_imgs.intersection(test_imgs)) == 0
    assert len(val_imgs.intersection(test_imgs)) == 0

def test_collate_fn():
    vocab = Vocabulary()
    img1 = torch.randn(3, 224, 224)
    img2 = torch.randn(3, 224, 224)
    cap1 = torch.tensor([1, 4, 5, 2])
    cap2 = torch.tensor([1, 4, 5, 6, 7, 2])

    batch = [(img1, cap1, "img1.jpg", "a dog runs"), (img2, cap2, "img2.jpg", "a cat runs fast today")]
    images, captions, lengths, img_ids, raw_caps = caption_collate_fn(batch, pad_idx=vocab.pad_idx)

    assert images.shape == (2, 3, 224, 224)
    assert captions.shape == (2, 6)
    assert lengths.tolist() == [4, 6]
    assert captions[0, 4].item() == vocab.pad_idx
