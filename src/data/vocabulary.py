import json
from collections import Counter
from typing import List, Dict, Union, Optional
import os

class Vocabulary:
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.itos: Dict[int, str] = {}
        self.stoi: Dict[str, int] = {}
        self.freqs: Counter = Counter()

        # Add special tokens
        self.add_token(self.PAD_TOKEN)    # idx 0
        self.add_token(self.START_TOKEN)  # idx 1
        self.add_token(self.END_TOKEN)    # idx 2
        self.add_token(self.UNK_TOKEN)    # idx 3

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.PAD_TOKEN]

    @property
    def start_idx(self) -> int:
        return self.stoi[self.START_TOKEN]

    @property
    def end_idx(self) -> int:
        return self.stoi[self.END_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[self.UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def add_token(self, token: str) -> int:
        if token not in self.stoi:
            idx = len(self.itos)
            self.stoi[token] = idx
            self.itos[idx] = token
            return idx
        return self.stoi[token]

    def build_vocabulary(self, sentence_list: List[List[str]]) -> None:
        """
        Build vocabulary from a list of tokenized sentences based on minimum word frequency.
        """
        for sentence in sentence_list:
            self.freqs.update(sentence)

        for word, count in self.freqs.items():
            if count >= self.min_freq:
                self.add_token(word)

    def numericalize(self, tokens: List[str], add_specials: bool = True) -> List[int]:
        """
        Convert tokens list to integer list. Optionally prepend <start> and append <end>.
        """
        result = []
        if add_specials:
            result.append(self.start_idx)
            
        for token in tokens:
            result.append(self.stoi.get(token, self.unk_idx))

        if add_specials:
            result.append(self.end_idx)

        return result

    def decode(self, indices: List[int], remove_specials: bool = True) -> List[str]:
        """
        Convert integer list back to list of string tokens.
        """
        tokens = []
        for idx in indices:
            token = self.itos.get(idx, self.UNK_TOKEN)
            if remove_specials:
                if token in (self.START_TOKEN, self.PAD_TOKEN):
                    continue
                if token == self.END_TOKEN:
                    break
            tokens.append(token)
        return tokens

    def decode_to_sentence(self, indices: List[int], remove_specials: bool = True) -> str:
        """
        Convert integer list directly to a clean string sentence.
        """
        tokens = self.decode(indices, remove_specials=remove_specials)
        return " ".join(tokens)

    def save(self, filepath: str) -> None:
        """
        Save vocabulary to JSON file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "min_freq": self.min_freq,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "freqs": dict(self.freqs)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """
        Load vocabulary from JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(min_freq=data["min_freq"])
        vocab.stoi = data["stoi"]
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        vocab.freqs = Counter(data["freqs"])
        return vocab
