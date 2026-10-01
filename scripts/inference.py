import os
import argparse
import yaml
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.utils.device import get_device
from src.data.vocabulary import Vocabulary
from src.models.cnn_lstm import CNNLSTMCaptioner
from src.training.checkpoint import load_checkpoint
from src.inference.generate import CaptionGenerator

logger = setup_logger()

def main():
    parser = argparse.ArgumentParser(description="Generate Caption for an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--vocab", type=str, default="data/processed/vocab.json", help="Path to vocabulary file")
    parser.add_argument("--method", type=str, default="beam", choices=["greedy", "beam"], help="Decoding strategy")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam search size (k)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        logger.error(f"Image not found at {args.image}")
        sys.exit(1)

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = get_device()

    # Load vocabulary
    vocab = Vocabulary.load(args.vocab)

    # Instantiate model
    model = CNNLSTMCaptioner(
        vocab_size=len(vocab),
        backbone=config["model"]["encoder"]["backbone"],
        embed_size=config["model"]["encoder"]["embed_size"],
        hidden_size=config["model"]["decoder"]["hidden_size"],
        attention_dim=config["model"]["decoder"]["attention_dim"],
        dropout=config["model"]["decoder"]["dropout"]
    )

    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model=model, device=device)
    else:
        logger.warning(f"Checkpoint not found at {args.checkpoint}. Running with initialized weights for demonstration.")

    generator = CaptionGenerator(
        model=model,
        vocab=vocab,
        device=device,
        max_length=config["generation"]["max_length"]
    )

    logger.info(f"Generating caption for image: {args.image} using {args.method} decoding...")
    if args.method == "beam":
        result = generator.generate_beam_search(args.image, beam_size=args.beam_size)
    else:
        result = generator.generate_greedy(args.image)

    print("\n" + "="*50)
    print(f"IMAGE: {args.image}")
    print(f"DECODING: {result['decoding']}")
    print(f"GENERATED CAPTION: {result['caption']}")
    if "score" in result:
        print(f"SCORE: {result['score']:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
