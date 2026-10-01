import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Callable, Tuple, List, Optional

from .vocabulary import Vocabulary
from .transforms import get_transforms

class FlickrDataset(Dataset):
    """
    PyTorch Dataset for Image Captioning.
    Loads image given image_id and converts caption text into numerical tensor.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        vocab: Vocabulary,
        transform: Optional[Callable] = None,
        max_length: int = 30
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.df.iloc[idx]
        img_id = row["image_id"]
        caption_text = row["caption"]

        img_path = os.path.join(self.image_dir, img_id)
        
        # Load image safely
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback if image path missing or corrupt (e.g. mock blank image for testing)
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform is not None:
            image = self.transform(image)

        tokens = caption_text.split()
        numericalized = self.vocab.numericalize(tokens, add_specials=True)
        
        # Truncate if exceeds max length (preserving <end> token)
        if len(numericalized) > self.max_length:
            numericalized = numericalized[:self.max_length - 1] + [self.vocab.end_idx]

        caption_tensor = torch.tensor(numericalized, dtype=torch.long)
        return image, caption_tensor, img_id, caption_text

def caption_collate_fn(batch, pad_idx: int):
    """
    Collate function to dynamically pad variable-length caption tensors in a batch.
    Returns:
        images: (batch_size, 3, H, W)
        captions: (batch_size, max_seq_len)
        lengths: (batch_size,) sequence lengths before padding
        img_ids: list of image_ids
        raw_captions: list of raw caption strings
    """
    images, captions, img_ids, raw_captions = zip(*batch)

    images = torch.stack(images, dim=0)
    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded_captions = torch.full((len(captions), max_len), fill_value=pad_idx, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return images, padded_captions, lengths_tensor, list(img_ids), list(raw_captions)

def get_dataloader(
    df: pd.DataFrame,
    image_dir: str,
    vocab: Vocabulary,
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    max_length: int = 30
) -> DataLoader:
    transform = get_transforms(split=split)
    dataset = FlickrDataset(
        df=df,
        image_dir=image_dir,
        vocab=vocab,
        transform=transform,
        max_length=max_length
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: caption_collate_fn(b, pad_idx=vocab.pad_idx),
        pin_memory=torch.cuda.is_available()
    )
    return dataloader
