"""
Standalone Inference Module: Word2Vec + PyTorch Stacked BiLSTM
Lead: Siyam Khan (ID: 2107120)

Loads pre-trained Word2Vec embeddings ('bangla_word2vec.pt'), vocabulary ('word2idx.json'),
and trained BiLSTM checkpoints ('bilstm_{task}.pt') from 'Project files/saved_models/'
to deliver context-aware real-time predictions on raw Bengali text for:
1. Sentiment Analysis (Negative, Neutral, Positive)
2. Sarcasm Detection (Non-Sarcastic, Sarcastic)
3. Hate Speech Detection (Non-Hate, Hate Speech)
"""

import os
import sys
import time
import json
import re
import torch
import torch.nn as nn
import numpy as np

# Ensure UTF-8 output encoding for Windows PowerShell
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add src to path for clean_bangla_text
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import clean_bangla_text

BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_SEQ_LEN = 50
TOKEN_PATTERN = re.compile(r'[\u0980-\u09FFa-zA-Z0-9]+|[!?]')

LABEL_MAPS = {
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
    "sarcasm": {0: "Non-Sarcastic", 1: "Sarcastic"},
    "hate_speech": {0: "Non-Hate", 1: "Hate Speech"}
}


class StackedBiLSTMClassifier(nn.Module):
    def __init__(self, embed_matrix: torch.Tensor, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        vocab_size, embed_dim = embed_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embed_matrix,
            freeze=False,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        out, (h_n, c_n) = self.lstm(embedded)
        forward_h = h_n[-2, :, :]
        backward_h = h_n[-1, :, :]
        bi_hidden = torch.cat((forward_h, backward_h), dim=1)
        bi_hidden = self.dropout(bi_hidden)
        logits = self.fc(bi_hidden)
        return logits


class BanglaTextAnalyzerBiLSTM:
    """
    Production-ready inference engine utilizing pre-trained Word2Vec embeddings
    and serialized PyTorch Stacked BiLSTM checkpoints.
    """
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.device = device
        self._load_artifacts()

    def _load_artifacts(self):
        # 1. Load Vocab & Word2Vec
        vocab_path = os.path.join(self.models_dir, "word2idx.json")
        embed_path = os.path.join(self.models_dir, "bangla_word2vec.pt")

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

        self.embed_matrix = torch.load(embed_path, map_location=self.device)

        # 2. Load Models
        self.models = {}
        for task_name, num_classes, ckpt_name in [
            ("sentiment", 3, "bilstm_sentiment.pt"),
            ("sarcasm", 2, "bilstm_sarcasm.pt"),
            ("hate_speech", 2, "bilstm_hate.pt")
        ]:
            model = StackedBiLSTMClassifier(
                embed_matrix=self.embed_matrix,
                hidden_dim=64,
                num_layers=2,
                num_classes=num_classes,
                dropout=0.0
            ).to(self.device)

            ckpt = torch.load(os.path.join(self.models_dir, ckpt_name), map_location=self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self.models[task_name] = model

    def encode(self, text: str) -> torch.Tensor:
        tokens = TOKEN_PATTERN.findall(text)
        unk_id = self.word2idx.get("<UNK>", 1)
        pad_id = self.word2idx.get("<PAD>", 0)

        indices = [self.word2idx.get(t, unk_id) for t in tokens[:MAX_SEQ_LEN]]
        if len(indices) < MAX_SEQ_LEN:
            indices += [pad_id] * (MAX_SEQ_LEN - len(indices))

        return torch.tensor([indices], dtype=torch.long, device=self.device)

    def analyze(self, raw_text: str) -> dict:
        t_start = time.perf_counter()

        cleaned = clean_bangla_text(raw_text)
        if not cleaned:
            cleaned = raw_text

        x_tensor = self.encode(cleaned)

        results = {
            "raw_text": raw_text,
            "cleaned_text": cleaned
        }

        with torch.no_grad():
            for task_name in ["sentiment", "sarcasm", "hate_speech"]:
                model = self.models[task_name]
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_id = int(np.argmax(probs))
                conf = float(probs[pred_id])
                label = LABEL_MAPS[task_name][pred_id]

                prob_dict = {}
                for idx, cname in LABEL_MAPS[task_name].items():
                    prob_dict[cname] = round(float(probs[idx]) * 100, 2)

                results[task_name] = {
                    "label": label,
                    "label_id": pred_id,
                    "confidence": round(conf * 100, 2),
                    "probabilities": prob_dict
                }

        results["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
        return results


def print_analysis(res: dict):
    print("=" * 65)
    print(f"Text: \"{res['raw_text']}\"")
    print(f"Cleaned: \"{res['cleaned_text']}\"")
    print("-" * 65)
    sent = res['sentiment']
    sarc = res['sarcasm']
    hate = res['hate_speech']
    print(f"  [Sentiment]   : {sent['label']:<15} (Conf: {sent['confidence']}%) | Probs: {sent['probabilities']}")
    print(f"  [Sarcasm]     : {sarc['label']:<15} (Conf: {sarc['confidence']}%) | Probs: {sarc['probabilities']}")
    print(f"  [Hate Speech] : {hate['label']:<15} (Conf: {hate['confidence']}%) | Probs: {hate['probabilities']}")
    print(f"  [Latency]     : {res['latency_ms']} ms (all 3 BiLSTMs combined)")
    print("=" * 65)


if __name__ == "__main__":
    analyzer = BanglaTextAnalyzerBiLSTM()

    samples = [
        "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!",
        "বইটি অসম্ভব সুন্দর, পড়ে অনেক কিছু শিখতে পারলাম। সবাইকে পড়ার অনুরোধ রইলো।",
        "তোদের মতো দেশদ্রোহীদের প্রকাশ্যে ফাঁসি দেওয়া উচিত, তোরা সমাজের কীট!",
        "আজকের আবহাওয়াটা বেশ সাধারণ, খুব গরমও না আবার ঠান্ডাও না।"
    ]

    print("\n--- STANDALONE PYTORCH BILSTM INFERENCE VERIFICATION ---\n")
    for s in samples:
        r = analyzer.analyze(s)
        print_analysis(r)
