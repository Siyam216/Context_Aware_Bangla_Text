"""
Standalone Inference Module: TF-IDF + Logistic Regression
Lead: Md. Tariful Islam Jony (ID: 2107119)

Loads pre-trained .joblib models and vectorizers from 'Project files/saved_models/'
to deliver real-time (< 2ms) predictions on raw Bengali text for:
1. Sentiment Analysis (Negative, Neutral, Positive)
2. Sarcasm Detection (Non-Sarcastic, Sarcastic)
3. Hate Speech Detection (Non-Hate, Hate Speech)
"""

import os
import sys
import time
import joblib

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

# Class label mappings
LABEL_MAPS = {
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
    "sarcasm": {0: "Non-Sarcastic", 1: "Sarcastic"},
    "hate_speech": {0: "Non-Hate", 1: "Hate Speech"}
}

class BanglaTextAnalyzerLR:
    """
    Production-ready inference engine utilizing serialized TF-IDF vectorizers
    and Logistic Regression classifiers.
    """
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self._load_artifacts()

    def _load_artifacts(self):
        # Load Vectorizers
        self.vec_sentiment = joblib.load(os.path.join(self.models_dir, "sentiment_tfidf_vectorizer.joblib"))
        self.vec_sarcasm = joblib.load(os.path.join(self.models_dir, "sarcasm_tfidf_vectorizer.joblib"))
        self.vec_hate = joblib.load(os.path.join(self.models_dir, "hate_tfidf_vectorizer.joblib"))

        # Load Models
        self.clf_sentiment = joblib.load(os.path.join(self.models_dir, "sentiment_lr_model.joblib"))
        self.clf_sarcasm = joblib.load(os.path.join(self.models_dir, "sarcasm_lr_model.joblib"))
        self.clf_hate = joblib.load(os.path.join(self.models_dir, "hate_lr_model.joblib"))

    def analyze(self, raw_text: str) -> dict:
        """
        Takes raw Bangla text and returns sentiment, sarcasm, and hate speech predictions
        with confidence scores and latency metrics.
        """
        t_start = time.perf_counter()

        # Step 1: Clean text
        cleaned = clean_bangla_text(raw_text)
        if not cleaned:
            cleaned = raw_text

        # Step 2: Vectorize & Predict Sentiment
        sent_vec = self.vec_sentiment.transform([cleaned])
        sent_pred = int(self.clf_sentiment.predict(sent_vec)[0])
        sent_probs = self.clf_sentiment.predict_proba(sent_vec)[0]
        sent_conf = float(sent_probs[sent_pred])

        # Step 3: Vectorize & Predict Sarcasm
        sarc_vec = self.vec_sarcasm.transform([cleaned])
        sarc_pred = int(self.clf_sarcasm.predict(sarc_vec)[0])
        sarc_probs = self.clf_sarcasm.predict_proba(sarc_vec)[0]
        sarc_conf = float(sarc_probs[sarc_pred])

        # Step 4: Vectorize & Predict Hate Speech
        hate_vec = self.vec_hate.transform([cleaned])
        hate_pred = int(self.clf_hate.predict(hate_vec)[0])
        hate_probs = self.clf_hate.predict_proba(hate_vec)[0]
        hate_conf = float(hate_probs[hate_pred])

        total_latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "sentiment": {
                "label": LABEL_MAPS["sentiment"][sent_pred],
                "label_id": sent_pred,
                "confidence": round(sent_conf * 100, 2),
                "probabilities": {
                    "Negative": round(float(sent_probs[0]) * 100, 2),
                    "Neutral": round(float(sent_probs[1]) * 100, 2),
                    "Positive": round(float(sent_probs[2]) * 100, 2)
                }
            },
            "sarcasm": {
                "label": LABEL_MAPS["sarcasm"][sarc_pred],
                "label_id": sarc_pred,
                "confidence": round(sarc_conf * 100, 2),
                "probabilities": {
                    "Non-Sarcastic": round(float(sarc_probs[0]) * 100, 2),
                    "Sarcastic": round(float(sarc_probs[1]) * 100, 2)
                }
            },
            "hate_speech": {
                "label": LABEL_MAPS["hate_speech"][hate_pred],
                "label_id": hate_pred,
                "confidence": round(hate_conf * 100, 2),
                "probabilities": {
                    "Non-Hate": round(float(hate_probs[0]) * 100, 2),
                    "Hate Speech": round(float(hate_probs[1]) * 100, 2)
                }
            },
            "latency_ms": round(total_latency_ms, 3)
        }


def print_analysis(result: dict):
    print("=" * 65)
    print(f"Text: \"{result['raw_text']}\"")
    print(f"Cleaned: \"{result['cleaned_text']}\"")
    print("-" * 65)
    sent = result['sentiment']
    sarc = result['sarcasm']
    hate = result['hate_speech']
    print(f"  [Sentiment]   : {sent['label']:<15} (Confidence: {sent['confidence']}%) | Probs: {sent['probabilities']}")
    print(f"  [Sarcasm]     : {sarc['label']:<15} (Confidence: {sarc['confidence']}%) | Probs: {sarc['probabilities']}")
    print(f"  [Hate Speech] : {hate['label']:<15} (Confidence: {hate['confidence']}%) | Probs: {hate['probabilities']}")
    print("=" * 65)


if __name__ == "__main__":
    analyzer = BanglaTextAnalyzerLR()
    
    benchmark_samples = [
        "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!",
        "বইটি অসম্ভব সুন্দর, পড়ে অনেক কিছু শিখতে পারলাম। সবাইকে পড়ার অনুরোধ রইলো।",
        "তোদের মতো দেশদ্রোহীদের প্রকাশ্যে ফাঁসি দেওয়া উচিত, তোরা সমাজের কীট!",
        "আজকের আবহাওয়াটা বেশ সাধারণ, খুব গরমও না আবার ঠান্ডাও না।"
    ]
    
    print("\n--- STANDALONE LOGISTIC REGRESSION INFERENCE VERIFICATION ---\n")
    for sample in benchmark_samples:
        res = analyzer.analyze(sample)
        print_analysis(res)
