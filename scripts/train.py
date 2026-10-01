import os
import argparse
import yaml
import pandas as pd
import torch
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.utils.device import get_device
from src.data.vocabulary import Vocabulary
from src.data.dataset import get_dataloader
from src.models.cnn_lstm import CNNLSTMCaptioner
from src.models.transformer_decoder import CNNTransformerCaptioner
from src.training.trainer import Trainer
from src.training.checkpoint import load_checkpoint

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--model-type", type=str, default=None, choices=["cnn_lstm", "cnn_transformer"], help="Override model architecture")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.model_type is not None:
        config["model"]["type"] = args.model_type

    seed_everything(config["experiment"]["seed"])
    device = get_device()

    processed_dir = config["dataset"]["processed_dir"]
    train_csv = os.path.join(processed_dir, "train.csv")
    val_csv = os.path.join(processed_dir, "val.csv")
    vocab_file = os.path.join(processed_dir, "vocab.json")

    # If processed files do not exist, run data preparation script
    if not (os.path.exists(train_csv) and os.path.exists(vocab_file)):
        logger.warning("Processed dataset or vocabulary missing. Running data preparation...")
        os.system(f"{sys.executable} scripts/prepare_data.py --config {args.config}")

    # Load vocabulary
    vocab = Vocabulary.load(vocab_file)
    logger.info(f"Loaded vocabulary with {len(vocab)} words.")

    # Load DataFrames
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Prepare DataLoaders
    image_dir = config["dataset"]["image_dir"]
    train_loader = get_dataloader(
        df=train_df,
        image_dir=image_dir,
        vocab=vocab,
        split="train",
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True,
        max_length=config["dataset"]["max_len"]
    )
    val_loader = get_dataloader(
        df=val_df,
        image_dir=image_dir,
        vocab=vocab,
        split="val",
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
        max_length=config["dataset"]["max_len"]
    )

    # Initialize Model
    model_type = config["model"]["type"]
    logger.info(f"Initializing model architecture: {model_type}")

    if model_type == "cnn_lstm":
        model = CNNLSTMCaptioner(
            vocab_size=len(vocab),
            backbone=config["model"]["encoder"]["backbone"],
            embed_size=config["model"]["encoder"]["embed_size"],
            hidden_size=config["model"]["decoder"]["hidden_size"],
            attention_dim=config["model"]["decoder"]["attention_dim"],
            dropout=config["model"]["decoder"]["dropout"],
            freeze_encoder=config["model"]["encoder"]["freeze_encoder"]
        )
    elif model_type == "cnn_transformer":
        model = CNNTransformerCaptioner(
            vocab_size=len(vocab),
            backbone=config["model"]["encoder"]["backbone"],
            embed_size=config["model"]["decoder"]["embed_size"],
            nhead=config["model"]["decoder"]["nhead"],
            num_decoder_layers=config["model"]["decoder"]["num_decoder_layers"],
            dim_feedforward=config["model"]["decoder"]["dim_feedforward"],
            dropout=config["model"]["decoder"]["dropout"],
            freeze_encoder=config["model"]["encoder"]["freeze_encoder"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if args.resume:
        load_checkpoint(args.resume, model=model, device=device)

    # Initialize Trainer and Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pad_idx=vocab.pad_idx,
        config=config,
        device=device
    )

    history = trainer.fit()
    logger.info("Training process completed successfully.")

if __name__ == "__main__":
    main()
