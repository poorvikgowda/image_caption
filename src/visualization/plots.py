import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List

sns.set_theme(style="whitegrid")

def plot_eda_summary(df: pd.DataFrame, save_dir: str = "outputs/figures") -> Dict[str, str]:
    """
    Generate publication-quality EDA charts and save to outputs/figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_paths = {}

    # 1. Caption Length Distribution
    lengths = [len(cap.split()) for cap in df["caption"].tolist()]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(lengths, bins=25, kde=True, color="#2b5c8f", ax=ax)
    ax.set_title("Caption Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Words per Caption", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    path1 = os.path.join(save_dir, "caption_length_distribution.png")
    plt.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close()
    file_paths["caption_length"] = path1

    # 2. Top 20 Most Frequent Words
    all_words = [w for cap in df["caption"].tolist() for w in cap.split()]
    counter = Counter(all_words)
    most_common = counter.most_common(20)
    words, counts = zip(*most_common)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), hue=list(words), legend=False, palette="Blues_r", ax=ax)
    ax.set_title("Top 20 Most Frequent Words in Captions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Frequency Count", fontsize=12)
    path2 = os.path.join(save_dir, "top_frequent_words.png")
    plt.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close()
    file_paths["top_words"] = path2

    return file_paths

def plot_training_curves(history: Dict[str, List[float]], save_dir: str = "outputs/figures") -> str:
    """
    Generate Training vs Validation Loss Curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", color="#ff7f0e", label="Val Loss", linewidth=2)
    ax1.set_title("Training & Validation Loss Curves", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.legend(loc="upper right", frameon=True)

    save_path = os.path.join(save_dir, "training_loss_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return save_path
