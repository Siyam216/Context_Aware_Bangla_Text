"""
Standalone Inference Module: Fine-Tuned BanglaBERT
Lead: Md. Tariful Islam Jony (ID: 2107119)

Loads fine-tuned BanglaBERT sequence classification heads from:
'Project files/saved_models/banglabert_{task}/'
to deliver context aware real-time predictions on raw Bengali text for:
1. Sentiment Analysis (Negative, Neutral, Positive)
2. Sarcasm Detection (Non-Sarcastic, Sarcastic)
3. Hate Speech Detection (Non-Hate, Hate Speech)
"""

import os
import sys
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
MAX_LENGTH = 64

LABEL_MAPS = {
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
    "sarcasm": {0: "Non-Sarcastic", 1: "Sarcastic"},
    "hate_speech": {0: "Non-Hate", 1: "Hate Speech"}
}


class BanglaTextAnalyzerBERT:
    """
    Production-ready inference engine utilizing fine-tuned BanglaBERT checkpoints.
    """
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.device = device
        self._load_artifacts()

    def _load_artifacts(self):
        tok_dir = os.path.join(self.models_dir, "banglabert_sentiment")
        if os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
            self.tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
        self.models = {}

        dir_map = {
            "sentiment": "banglabert_sentiment",
            "sarcasm": "banglabert_sarcasm",
            "hate_speech": "banglabert_hate" if os.path.exists(os.path.join(self.models_dir, "banglabert_hate")) else "banglabert_hate_speech"
        }
        for task_name, dir_name in dir_map.items():
            model_path = os.path.join(self.models_dir, dir_name)
            if os.path.exists(model_path):
                model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(self.device)
                model.eval()
                self.models[task_name] = model
            else:
                print(f"[!] Warning: {model_path} not found.")

    def analyze(self, raw_text: str) -> dict:
        t_start = time.perf_counter()

        cleaned = clean_bangla_text(raw_text)
        if not cleaned:
            cleaned = raw_text

        inputs = self.tokenizer(
            cleaned,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        results = {
            "raw_text": raw_text,
            "cleaned_text": cleaned
        }

        with torch.no_grad():
            for task_name, model in self.models.items():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
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
    for t in ["sentiment", "sarcasm", "hate_speech"]:
        if t in res:
            task_res = res[t]
            print(f"  [{t.title():<12}] : {task_res['label']:<15} (Conf: {task_res['confidence']}%) | Probs: {task_res['probabilities']}")
    print("=" * 65)


if __name__ == "__main__":
    analyzer = BanglaTextAnalyzerBERT()

    samples = [
        "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!",
        "বইটি অসম্ভব সুন্দর, পড়ে অনেক কিছু শিখতে পারলাম। সবাইকে পড়ার অনুরোধ রইলো।",
        "তোদের মতো দেশদ্রোহীদের প্রকাশ্যে ফাঁসি দেওয়া উচিত, তোরা সমাজের কীট!",
        "আজকের আবহাওয়াটা বেশ সাধারণ, খুব গরমও না আবার ঠান্ডাও না।"
    ]

    print("\n--- STANDALONE BANGLABERT INFERENCE VERIFICATION ---\n")
    for s in samples:
        r = analyzer.analyze(s)
        print_analysis(r)
