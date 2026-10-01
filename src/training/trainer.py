import os
import time
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from .losses import CaptionLoss
from .checkpoint import save_checkpoint

logger = logging.getLogger("ImageCaptioning")

class Trainer:
    """
    Complete Trainer class managing training, validation, AMP, early stopping, and history tracking.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pad_idx: int,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_idx = pad_idx
        self.config = config
        self.device = device

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.gradient_clip = train_cfg["gradient_clip"]
        self.patience = train_cfg["patience"]
        self.teacher_forcing_ratio = train_cfg.get("teacher_forcing_ratio", 1.0)
        self.use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"

        # Optimizer: Different learning rates for Encoder vs Decoder
        params = []
        if hasattr(model, "encoder") and model.encoder is not None:
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                params.append({"params": encoder_params, "lr": train_cfg["encoder_lr"]})

        decoder_params = [p for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad]
        params.append({"params": decoder_params, "lr": train_cfg["decoder_lr"]})

        self.optimizer = torch.optim.AdamW(
            params,
            weight_decay=train_cfg.get("weight_decay", 1e-5)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        self.criterion = CaptionLoss(
            pad_idx=pad_idx,
            label_smoothing=train_cfg.get("label_smoothing", 0.0)
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": []
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        start_time = time.time()
        for i, (images, captions, lengths, _, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            captions = captions.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                predictions, alphas = self.model(
                    images, captions, lengths, teacher_forcing_ratio=self.teacher_forcing_ratio
                )
                loss = self.criterion(predictions, captions, alphas)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            total_loss += loss.item()

            if (i + 1) % max(1, num_batches // 5) == 0 or (i + 1) == num_batches:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch [{epoch}/{self.epochs}] Batch [{i+1}/{num_batches}] "
                    f"Loss: {loss.item():.4f} ({elapsed:.1f}s)"
                )

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        for images, captions, lengths, _, _ in self.val_loader:
            images = images.to(self.device)
            captions = captions.to(self.device)
            lengths = lengths.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                predictions, alphas = self.model(
                    images, captions, lengths, teacher_forcing_ratio=1.0
                )
                loss = self.criterion(predictions, captions, alphas)

            total_loss += loss.item()

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def fit(self) -> Dict[str, List[float]]:
        best_val_loss = float("inf")
        patience_counter = 0

        checkpoint_dir = self.config["experiment"]["checkpoint_dir"]
        output_dir = self.config["experiment"]["output_dir"]

        logger.info(f"Starting training for {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[-1]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(current_lr)

            logger.info(
                f"--- Epoch [{epoch}/{self.epochs}] Complete --- "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
                "seed": self.config["experiment"]["seed"]
            }
            save_checkpoint(checkpoint_state, is_best, checkpoint_dir=checkpoint_dir)

            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

        # Save training history
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        history_path = os.path.join(output_dir, "logs", "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        return self.history
