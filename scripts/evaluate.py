import os
import argparse
import yaml
import pandas as pd
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.device import get_device
from src.data.vocabulary import Vocabulary
from src.models.cnn_lstm import CNNLSTMCaptioner
from src.models.transformer_decoder import CNNTransformerCaptioner
from src.training.checkpoint import load_checkpoint
from src.evaluation.evaluate import evaluate_model
from src.evaluation.error_analysis import analyze_errors

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device()

    processed_dir = config["dataset"]["processed_dir"]
    test_csv = os.path.join(processed_dir, "test.csv")
    vocab_file = os.path.join(processed_dir, "vocab.json")

    if not os.path.exists(test_csv) or not os.path.exists(vocab_file):
        logger.error(f"Missing test data or vocabulary in {processed_dir}. Run prepare_data.py first.")
        sys.exit(1)

    vocab = Vocabulary.load(vocab_file)
    test_df = pd.read_csv(test_csv)

    # Instantiate model
    model_type = config["model"]["type"]
    if model_type == "cnn_lstm":
        model = CNNLSTMCaptioner(
            vocab_size=len(vocab),
            backbone=config["model"]["encoder"]["backbone"],
            embed_size=config["model"]["encoder"]["embed_size"],
            hidden_size=config["model"]["decoder"]["hidden_size"],
            attention_dim=config["model"]["decoder"]["attention_dim"],
            dropout=config["model"]["decoder"]["dropout"]
        )
    else:
        model = CNNTransformerCaptioner(
            vocab_size=len(vocab),
            backbone=config["model"]["encoder"]["backbone"],
            embed_size=config["model"]["decoder"]["embed_size"],
            nhead=config["model"]["decoder"]["nhead"],
            num_decoder_layers=config["model"]["decoder"]["num_decoder_layers"],
            dim_feedforward=config["model"]["decoder"]["dim_feedforward"],
            dropout=config["model"]["decoder"]["dropout"]
        )

    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model=model, device=device)
    else:
        logger.warning(f"Checkpoint not found at {args.checkpoint}. Running evaluation on uninitialized model weights.")

    image_dir = config["dataset"]["image_dir"]
    results = evaluate_model(
        model=model,
        vocab=vocab,
        test_df=test_df,
        image_dir=image_dir,
        device=device,
        config=config
    )

    if "sample_predictions" in results:
        analyze_errors(results["sample_predictions"], output_dir=config["experiment"]["output_dir"])

    logger.info("Evaluation finished successfully.")

if __name__ == "__main__":
    main()
