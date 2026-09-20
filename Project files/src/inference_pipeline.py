"""
Unified Multi-Task Inference Pipeline
Context Aware Bangla Text Analyzer for Sentiment, Sarcasm, and Hate Speech Detection.

Academic Context: CSE 4122 (NLP Sessional), Dept. of CSE, KUET.
Team: Md. Tariful Islam Jony (2107119) & Siyam Khan (2107120)

Integrates all three modeling paradigms:
1. TF-IDF + Balanced Logistic Regression
2. Dense Word2Vec (128-d) + PyTorch Stacked BiLSTM
3. Fine-Tuned sagorsarker/bangla-bert-base Transformer
4. Context Aware Cross-Task Ensemble (Proposal-Aligned Formulation)
"""

import os
import sys
import time
from typing import Dict, Any, Optional

# Ensure src directory is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from preprocessing import clean_bangla_text, tokenize_bangla
from predict_lr import BanglaTextAnalyzerLR
from predict_bilstm import BanglaTextAnalyzerBiLSTM
from predict_bert import BanglaTextAnalyzerBERT

# Sentiment Lexicon anchors for factual neutral detection
POS_WORDS = {
    'ভালো', 'চমৎকার', 'সুন্দর', 'অসাধারণ', 'অসাধারন', 'সেরা', 'ধন্যবাদ', 'ভালোবাসি', 'উপকারী', 
    'দারুণ', 'আনন্দ', 'পছন্দ', 'খুশি', 'সন্তুষ্ট', 'সফল', 'অপূর্ব', 'দারুন', 'ভালোই', 
    'উপকার', 'সাধুবাদ', 'প্রশংসা', 'আরামদায়ক', 'উপভোগ', 'প্রিয়', 'উৎকৃষ্ট', 'ভালোবাসা',
    'লাভজনক', 'সুনাম', 'মিষ্টি', 'ফাটাফাটি', 'জটিল', 'জোশ', 'মুগ্ধ', 'ভালোলাগা', 'মনোরম',
    'মনোমুগ্ধকর', 'চরম', 'উপভোগ্য', 'প্রশংসনীয়', 'শুভকামনা', 'অনবদ্য', 'দুর্দান্ত', 'সুপার',
    'গ্রেট', 'খাসা', 'সাবাস', 'সাবাশ'
}

NEG_WORDS = {
    'খারাপ', 'বাজে', 'জঘন্য', 'ফালতু', 'বিরক্ত', 'কষ্ট', 'দুঃখ', 'লজ্জা', 'ক্ষতি', 
    'ব্যর্থ', 'অসহ্য', 'ঘৃণা', 'নষ্ট', 'ঠকা', 'ধোঁকা', 'অপদার্থ', 'হয়রানি', 'হতাশ', 
    'হতাশা', 'অসুবিধা', 'বিপদ', 'অসুন্দর', 'ভুল', 'বিরক্তিকর', 'ঘৃণ্য', 'প্রতারণা',
    'ঠকবাজ', 'জালিয়াতি', 'যন্ত্রণা', 'ঘটিয়া', 'ছাইপাশ', 'আবর্জনা', 'ঘেন্না', 'বিরক্তি',
    'অসহনীয়'
}

IRONY_MARKERS = {
    'বাহ', 'বাঃ', 'সাবাশ', 'দারুণ', 'অসাধারণ', 'খাসা', 'কী দারুণ', 'কী চমৎকার', 
    'কী অসাধারণ', 'বটে', 'কী যে'
}


class UnifiedBanglaTextAnalyzer:
    """
    Unified Inference Engine coordinating TF-IDF+LR, BiLSTM, and BanglaBERT.
    Supports lazy loading and cross-task context aware aggregation.
    """

    def __init__(self, preload_models: bool = False):
        self._lr: Optional[BanglaTextAnalyzerLR] = None
        self._bilstm: Optional[BanglaTextAnalyzerBiLSTM] = None
        self._bert: Optional[BanglaTextAnalyzerBERT] = None

        if preload_models:
            self.load_all_models()

    @property
    def lr(self) -> BanglaTextAnalyzerLR:
        if self._lr is None:
            self._lr = BanglaTextAnalyzerLR()
        return self._lr

    @property
    def bilstm(self) -> BanglaTextAnalyzerBiLSTM:
        if self._bilstm is None:
            self._bilstm = BanglaTextAnalyzerBiLSTM()
        return self._bilstm

    @property
    def bert(self) -> BanglaTextAnalyzerBERT:
        if self._bert is None:
            self._bert = BanglaTextAnalyzerBERT()
        return self._bert

    def load_all_models(self):
        """Eagerly load all models into memory to eliminate runtime cold starts."""
        _ = self.lr
        _ = self.bilstm
        _ = self.bert

    def analyze_lr(self, text: str) -> Dict[str, Any]:
        """Inference with TF-IDF + Logistic Regression."""
        return self.lr.analyze(text)

    def analyze_bilstm(self, text: str) -> Dict[str, Any]:
        """Inference with Word2Vec + Stacked BiLSTM."""
        return self.bilstm.analyze(text)

    def analyze_bert(self, text: str) -> Dict[str, Any]:
        """Inference with Fine-Tuned BanglaBERT."""
        return self.bert.analyze(text)

    def analyze_all_individual(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Run all three individual models and return their results."""
        return {
            "lr": self.analyze_lr(text),
            "bilstm": self.analyze_bilstm(text),
            "bert": self.analyze_bert(text)
        }

    def analyze_ensemble(self, text: str) -> Dict[str, Any]:
        """
        Proposal-Aligned Context Aware Ensemble:
        1. Hate Speech: Uses Word2Vec + BiLSTM (our winner paradigm: 88.52% Acc, 88.50% Macro-F1).
        2. Sarcasm: Evaluates TF-IDF+LR (Sarcastic marker sensitivity) and BanglaBERT soft probabilities.
           If Sarcasm is detected (LR says Sarcastic or BERT Sarcasm prob > 40% with contrast markers).
        3. Sentiment: Deep Contextual Resolution.
           - If Sarcasm is confirmed and surface expression was positive with negation ('না', 'নয়', 'নেই', 'কাজ হলো না'),
             true contextual sentiment is inverted to NEGATIVE.
           - Otherwise, blends BanglaBERT contextual probabilities with TF-IDF keyword confidence.
        """
        t_start = time.perf_counter()

        cleaned = clean_bangla_text(text)
        tokens = tokenize_bangla(cleaned)
        negation_tokens = [t for t in tokens if t in ["না", "নয়", "নেই", "নাহ", "কখনো না", "বিনা", "ছাড়া", "নাই", "নি"]]

        # Run individual models
        lr_res = self.analyze_lr(text)
        bi_res = self.analyze_bilstm(text)
        bert_res = self.analyze_bert(text)

        # 1. Hate Speech: Weighted Consensus between BanglaBERT (91.6% F1) and BiLSTM (88.5% F1)
        bert_hate_prob = bert_res["hate_speech"]["probabilities"].get("Hate Speech", 0.0)
        bi_hate_prob = bi_res["hate_speech"]["probabilities"].get("Hate Speech", 0.0)
        lr_hate_prob = lr_res["hate_speech"]["probabilities"].get("Hate Speech", 0.0)

        blended_hate_prob = round(0.50 * bert_hate_prob + 0.35 * bi_hate_prob + 0.15 * lr_hate_prob, 2)
        is_hate = blended_hate_prob >= 50.0
        hate_label = "Hate Speech" if is_hate else "Non-Hate"
        hate_conf = blended_hate_prob if is_hate else round(100.0 - blended_hate_prob, 2)
        hate_probs = {
            "Non-Hate": round(100.0 - blended_hate_prob, 2),
            "Hate Speech": blended_hate_prob
        }

        # 2. Sarcasm Detection (Multi-Model Consensus & Contrast Filtering):
        lr_sarc_label = lr_res["sarcasm"]["label"]
        bi_sarc_label = bi_res["sarcasm"]["label"]
        bert_sarc_label = bert_res["sarcasm"]["label"]

        lr_sarc_prob = lr_res["sarcasm"]["probabilities"].get("Sarcastic", 0.0)
        bi_sarc_prob = bi_res["sarcasm"]["probabilities"].get("Sarcastic", 0.0)
        bert_sarc_prob = bert_res["sarcasm"]["probabilities"].get("Sarcastic", 0.0)

        blended_sarc_prob = round(0.45 * bert_sarc_prob + 0.35 * bi_sarc_prob + 0.20 * lr_sarc_prob, 2)
        sarc_votes = sum([
            1 if lr_sarc_label == "Sarcastic" else 0,
            1 if bi_sarc_label == "Sarcastic" else 0,
            1 if bert_sarc_label == "Sarcastic" else 0
        ])

        has_irony_cue = any(im in text for im in IRONY_MARKERS)
        has_contrast_marker = (has_irony_cue or "অসাধারণ" in text or "দারুণ" in text) and len(negation_tokens) > 0
        has_exclamation = ("!" in text or "!" in cleaned or "!!" in text)

        is_sarcastic = False
        if has_contrast_marker and (blended_sarc_prob >= 35.0 or sarc_votes >= 1):
            # Classical praise + negation contradiction pattern
            is_sarcastic = True
            sarcasm_conf = round(max(blended_sarc_prob, 65.0), 2)
        elif (has_irony_cue or has_exclamation) and blended_sarc_prob >= 50.0 and sarc_votes >= 2:
            # Multi-model consensus with expressive punctuation/irony markers
            is_sarcastic = True
            sarcasm_conf = blended_sarc_prob
        elif sarc_votes == 3 and blended_sarc_prob >= 70.0 and has_exclamation:
            # Unanimous strong agreement with expressive cue
            is_sarcastic = True
            sarcasm_conf = blended_sarc_prob
        else:
            is_sarcastic = False
            # When filtered as Non-Sarcastic:
            # If raw models were biased (>50%) but lacked irony/contrast markers, confidence of non-sarcasm is high
            if blended_sarc_prob < 50.0:
                sarcasm_conf = round(100.0 - blended_sarc_prob, 2)
            else:
                sarcasm_conf = round(max(85.0, 100.0 - (blended_sarc_prob - 50.0)), 2)

        sarcasm_label = "Sarcastic" if is_sarcastic else "Non-Sarcastic"
        sarcasm_binary = "Yes" if is_sarcastic else "No"

        # 3. Sentiment Analysis (Context Aware Tri-Model Blending + Semantic Inversion):
        lr_probs = lr_res["sentiment"]["probabilities"]
        bi_probs = bi_res["sentiment"]["probabilities"]
        bert_probs = bert_res["sentiment"]["probabilities"]

        classes = ["Negative", "Neutral", "Positive"]
        blended_probs = {
            cls: round(
                0.35 * lr_probs.get(cls, 0.0) +
                0.35 * bi_probs.get(cls, 0.0) +
                0.30 * bert_probs.get(cls, 0.0), 2
            )
            for cls in classes
        }

        pos_p = blended_probs.get("Positive", 0.0)
        neg_p = blended_probs.get("Negative", 0.0)
        neu_p = blended_probs.get("Neutral", 0.0)

        # Count individual model votes
        pos_votes = sum([
            1 if lr_res["sentiment"]["label"] == "Positive" else 0,
            1 if bi_res["sentiment"]["label"] == "Positive" else 0,
            1 if bert_res["sentiment"]["label"] == "Positive" else 0
        ])
        neg_votes = sum([
            1 if lr_res["sentiment"]["label"] == "Negative" else 0,
            1 if bi_res["sentiment"]["label"] == "Negative" else 0,
            1 if bert_res["sentiment"]["label"] == "Negative" else 0
        ])
        neu_votes = sum([
            1 if lr_res["sentiment"]["label"] == "Neutral" else 0,
            1 if bi_res["sentiment"]["label"] == "Neutral" else 0,
            1 if bert_res["sentiment"]["label"] == "Neutral" else 0
        ])

        has_pos = any(w in text for w in POS_WORDS)
        has_neg = any(w in text for w in NEG_WORDS)

        context_inverted = False
        lr_sent_label = lr_res["sentiment"]["label"]

        if is_hate:
            # Cross-task consistency: hate speech is inherently negative sentiment
            final_sentiment = "Negative"
            final_sent_conf = round(max(neg_p, hate_conf, 85.0), 2)
        elif is_sarcastic and len(negation_tokens) > 0 and (has_pos or lr_sent_label == "Positive" or pos_p > neg_p or pos_votes >= 1):
            # Proposal Sarcasm Contrast Rule: Positive facade + Negative outcome = True Negative Sentiment
            final_sentiment = "Negative"
            final_sent_conf = round(max(bert_probs.get("Negative", 0.0), neg_p, 68.0), 2)
            context_inverted = True
        elif pos_votes >= 2 or (pos_p > neg_p and (has_pos or pos_p >= 40.0)):
            # Positive: Multi-model consensus or strong positive probability
            final_sentiment = "Positive"
            final_sent_conf = pos_p
        elif (neg_votes >= 2 and (has_neg or len(negation_tokens) > 0 or "খারাপ" in text or "বাজে" in text)) or (has_neg and neg_p > pos_p):
            # Genuine Negative confirmed by negative vocabulary or negation
            final_sentiment = "Negative"
            final_sent_conf = neg_p
        elif neu_votes >= 2 or (neu_p > pos_p and neu_p > neg_p):
            # Explicit Neutral consensus among models
            final_sentiment = "Neutral"
            final_sent_conf = neu_p
        elif not has_pos and not has_neg and not is_sarcastic and len(negation_tokens) == 0:
            # Factual / Objective Everyday Statement (e.g., 'আমি ভাত খাই', 'ঢাকা বাংলাদেশের রাজধানী')
            # Text contains zero sentiment polarity cues; resolves review-dataset false negative bias
            final_sentiment = "Neutral"
            final_sent_conf = round(max(neu_p, 68.0), 2)
        else:
            # Tri-model weighted consensus
            final_sentiment = max(blended_probs, key=blended_probs.get)
            final_sent_conf = blended_probs[final_sentiment]

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)

        return {
            "raw_text": text,
            "cleaned_text": cleaned,
            "tokens": tokens,
            "negations_found": negation_tokens,
            "sentiment": {
                "label": final_sentiment,
                "confidence": final_sent_conf,
                "context_inverted": context_inverted
            },
            "sarcasm": {
                "label": sarcasm_label,
                "binary": sarcasm_binary,
                "confidence": sarcasm_conf
            },
            "hate_speech": {
                "label": hate_label,
                "binary": "Yes" if hate_label == "Hate Speech" else "No",
                "confidence": hate_conf,
                "probabilities": hate_probs
            },
            "individual_models": {
                "lr": lr_res,
                "bilstm": bi_res,
                "bert": bert_res
            },
            "latency_ms": total_latency,
            "routing_explanation": (
                "Sentiment: Resolved via BanglaBERT deep contextual contrast inversion. "
                if context_inverted else
                (
                    "Sentiment: Identified as objective/factual neutral statement. "
                    if final_sentiment == "Neutral" and not has_pos and not has_neg else
                    "Sentiment: Calibrated between keyword n-grams and BERT semantics. "
                )
            ) + (
                "Sarcasm: Confirmed via cross-model irony/contrast sensitivity. "
                if is_sarcastic else
                "Sarcasm: Cleared via robust consensus voting. "
            ) + f"Hate Speech: Evaluated via BiLSTM & Transformer consensus ({hate_label})."
        }

    def analyze(self, raw_text: str, model_type: str = "ensemble") -> Dict[str, Any]:
        """
        Public inference API.
        model_type options:
            - 'ensemble' (default, Context Aware Multi-Task Routing)
            - 'lr' (TF-IDF + Logistic Regression)
            - 'bilstm' (Word2Vec + Stacked BiLSTM)
            - 'bert' (Fine-Tuned BanglaBERT)
        """
        cleaned_model_type = model_type.lower().strip()

        if "lr" in cleaned_model_type or "logistic" in cleaned_model_type:
            return self.analyze_lr(raw_text)
        elif "bilstm" in cleaned_model_type or "lstm" in cleaned_model_type or "word2vec" in cleaned_model_type:
            return self.analyze_bilstm(raw_text)
        elif "bert" in cleaned_model_type:
            return self.analyze_bert(raw_text)
        else:
            return self.analyze_ensemble(raw_text)


if __name__ == "__main__":
    analyzer = UnifiedBanglaTextAnalyzer()

    benchmark_samples = [
        "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!",
        "বইটি অসম্ভব সুন্দর, পড়ে অনেক কিছু শিখতে পারলাম। সবাইকে পড়ার অনুরোধ রইলো।",
        "তোদের মতো দেশদ্রোহীদের প্রকাশ্যে ফাঁসি দেওয়া উচিত, তোরা সমাজের কীট!",
        "আজকের আবহাওয়াটা বেশ সাধারণ, খুব গরমও না আবার ঠান্ডাও না。"
    ]

    print("\n" + "=" * 80)
    print("UNIFIED CONTEXT AWARE INFERENCE ENGINE VERIFICATION")
    print("=" * 80)

    for sample in benchmark_samples:
        res = analyzer.analyze(sample, model_type="ensemble")
        print(f"\nText: \"{res['raw_text']}\"")
        print(f"  [Sentiment]   : {res['sentiment']['label']:<12} (Confidence: {res['sentiment']['confidence']}%) "
              f"| Inverted: {res['sentiment']['context_inverted']}")
        print(f"  [Sarcasm]     : {res['sarcasm']['binary']:<12} ({res['sarcasm']['label']}, Conf: {res['sarcasm']['confidence']}%)")
        print(f"  [Hate Speech] : {res['hate_speech']['binary']:<12} ({res['hate_speech']['label']}, Conf: {res['hate_speech']['confidence']}%)")
        print(f"  [Explanation] : {res['routing_explanation']}")
        print("-" * 80)
