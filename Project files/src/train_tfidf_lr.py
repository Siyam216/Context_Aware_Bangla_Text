"""
Phase 3: TF-IDF Feature Extraction & Logistic Regression Classifiers
Lead: Md. Tariful Islam Jony (ID: 2107119)
Academic Mapping: Lab 2 (TF-IDF Vectorization) & Lab 3 (Discriminative Logistic Regression)

This script:
1. Loads cleaned datasets (train, val, test) for:
   - Sentiment (3 classes: Negative=0, Neutral=1, Positive=2)
   - Sarcasm (Binary: Non-Sarcastic=0, Sarcastic=1)
   - Hate Speech (Binary: Non-Hate=0, Hate=1)
2. Fits TfidfVectorizer (unigram + bigram, sublinear_tf=True, custom Bengali token pattern) strictly on train.csv.
3. Transforms train, val, and test splits without data leakage.
4. Trains LogisticRegression with class_weight='balanced' for each task to address class imbalances.
5. Evaluates on validation and test sets (Accuracy, Macro Precision/Recall/F1, Weighted F1, Confusion Matrix).
6. Serializes all vectorizers and models into 'Project files/saved_models/' with joblib.
7. Saves detailed benchmark metrics into 'Project files/results_tfidf_lr.json'.
8. Generates high-res confusion matrix figures in 'Project files/eda_plots/confusion_matrices_lr.png'.
9. Verifies sub-millisecond inference and test benchmarks.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Ensure UTF-8 output encoding for Windows PowerShell
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "eda_plots")
RESULTS_PATH = os.path.join(BASE_DIR, "results_tfidf_lr.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Dataset configurations
TASKS = {
    "sentiment": {
        "dir": os.path.join(CLEANED_DIR, "sentiment"),
        "max_features": 20000,
        "class_names": ["Negative", "Neutral", "Positive"],
        "labels": [0, 1, 2],
        "vec_name": "sentiment_tfidf_vectorizer.joblib",
        "model_name": "sentiment_lr_model.joblib"
    },
    "sarcasm": {
        "dir": os.path.join(CLEANED_DIR, "sarcasm"),
        "max_features": 15000,
        "class_names": ["Non-Sarcastic", "Sarcastic"],
        "labels": [0, 1],
        "vec_name": "sarcasm_tfidf_vectorizer.joblib",
        "model_name": "sarcasm_lr_model.joblib"
    },
    "hate_speech": {
        "dir": os.path.join(CLEANED_DIR, "hate_speech"),
        "max_features": 15000,
        "class_names": ["Non-Hate", "Hate Speech"],
        "labels": [0, 1],
        "vec_name": "hate_tfidf_vectorizer.joblib",
        "model_name": "hate_lr_model.joblib"
    }
}

# Unicode Bengali token pattern capturing words, digits, and expressive punctuation (!, ?)
BANGLA_TOKEN_PATTERN = r'[\u0980-\u09FFa-zA-Z0-9]+|[!?]'


def load_dataset(task_dir: str):
    train_df = pd.read_csv(os.path.join(task_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(task_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(task_dir, "test.csv"))
    
    # Fill any nulls with empty string just in case
    train_df['text'] = train_df['text'].fillna('')
    val_df['text'] = val_df['text'].fillna('')
    test_df['text'] = test_df['text'].fillna('')
    
    return train_df, val_df, test_df


def train_and_evaluate_task(task_name: str, config: dict):
    print(f"\n=======================================================")
    print(f"[*] Training TF-IDF + Logistic Regression: {task_name.upper()}")
    print(f"=======================================================")
    
    # 1. Load Data
    train_df, val_df, test_df = load_dataset(config["dir"])
    print(f"  [+] Loaded splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    
    X_train_raw = train_df['text'].values
    y_train = train_df['label'].values
    
    X_val_raw = val_df['text'].values
    y_val = val_df['label'].values
    
    X_test_raw = test_df['text'].values
    y_test = test_df['label'].values
    
    # 2. Fit TF-IDF Vectorizer strictly on train
    print(f"  [+] Extracting TF-IDF Features (ngram_range=(1,2), max_features={config['max_features']:,}, sublinear_tf=True)...")
    vec_start = time.time()
    vectorizer = TfidfVectorizer(
        token_pattern=BANGLA_TOKEN_PATTERN,
        ngram_range=(1, 2),
        max_features=config["max_features"],
        min_df=2,
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_val_vec = vectorizer.transform(X_val_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    vec_time = time.time() - vec_start
    print(f"  [+] Vectorization completed in {vec_time:.2f}s. Vocabulary size: {len(vectorizer.vocabulary_):,}")
    
    # 3. Train Logistic Regression
    print(f"  [+] Training LogisticRegression (class_weight='balanced', max_iter=1000, solver='lbfgs')...")
    train_start = time.time()
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    lr_model.fit(X_train_vec, y_train)
    train_time = time.time() - train_start
    print(f"  [+] Model trained in {train_time:.2f}s.")
    
    # 4. Evaluate on Validation Set
    val_preds = lr_model.predict(X_val_vec)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec_macro, val_rec_macro, val_f1_macro, _ = precision_recall_fscore_support(
        y_val, val_preds, average='macro', zero_division=0
    )
    print(f"  [+] Validation Performance -> Accuracy: {val_acc*100:.2f}%, Macro F1: {val_f1_macro*100:.2f}%")
    
    # 5. Evaluate on Test Set
    test_preds = lr_model.predict(X_test_vec)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec_macro, test_rec_macro, test_f1_macro, _ = precision_recall_fscore_support(
        y_test, test_preds, average='macro', zero_division=0
    )
    test_prec_wt, test_rec_wt, test_f1_wt, _ = precision_recall_fscore_support(
        y_test, test_preds, average='weighted', zero_division=0
    )
    
    # Class-wise metrics
    p_per_class, r_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, test_preds, labels=config["labels"], average=None, zero_division=0
    )
    
    class_metrics = {}
    for idx, cname in enumerate(config["class_names"]):
        class_metrics[cname] = {
            "precision": float(round(p_per_class[idx], 4)),
            "recall": float(round(r_per_class[idx], 4)),
            "f1_score": float(round(f1_per_class[idx], 4)),
            "support": int(support_per_class[idx])
        }
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds, labels=config["labels"])
    
    print(f"\n  [+] TEST RESULTS for {task_name.upper()}:")
    print(f"      - Test Accuracy:     {test_acc*100:.2f}%")
    print(f"      - Test Macro F1:     {test_f1_macro*100:.2f}%")
    print(f"      - Test Weighted F1:  {test_f1_wt*100:.2f}%")
    print(f"      - Test Macro Recall: {test_rec_macro*100:.2f}%")
    print("\n  [+] Detailed Classification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=config["class_names"], digits=4))
    
    # 6. Save Artifacts (joblib)
    vec_path = os.path.join(MODELS_DIR, config["vec_name"])
    model_path = os.path.join(MODELS_DIR, config["model_name"])
    joblib.dump(vectorizer, vec_path, compress=3)
    joblib.dump(lr_model, model_path, compress=3)
    print(f"  [+] Serialized vectorizer -> {vec_path} ({os.path.getsize(vec_path)/(1024*1024):.2f} MB)")
    print(f"  [+] Serialized model      -> {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
    
    return {
        "task": task_name,
        "vocab_size": len(vectorizer.vocabulary_),
        "vectorization_time_sec": round(vec_time, 2),
        "training_time_sec": round(train_time, 2),
        "validation_metrics": {
            "accuracy": float(round(val_acc, 4)),
            "macro_precision": float(round(val_prec_macro, 4)),
            "macro_recall": float(round(val_rec_macro, 4)),
            "macro_f1": float(round(val_f1_macro, 4))
        },
        "test_metrics": {
            "accuracy": float(round(test_acc, 4)),
            "macro_precision": float(round(test_prec_macro, 4)),
            "macro_recall": float(round(test_rec_macro, 4)),
            "macro_f1": float(round(test_f1_macro, 4)),
            "weighted_f1": float(round(test_f1_wt, 4)),
            "class_wise": class_metrics
        },
        "confusion_matrix": cm.tolist()
    }, cm, vectorizer, lr_model


def plot_all_confusion_matrices(results_dict, cm_dict):
    """
    Plots a 3-panel publication-ready confusion matrix figure for all 3 tasks.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.35)
    
    for idx, (task_name, config) in enumerate(TASKS.items()):
        ax = axes[idx]
        cm = np.array(cm_dict[task_name])
        class_names = config["class_names"]
        
        # Calculate percentages
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Annotation text: Count + Percentage
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"
        
        cmap = "Blues" if task_name == "sentiment" else ("Purples" if task_name == "sarcasm" else "Reds")
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})
        
        test_f1 = results_dict[task_name]["test_metrics"]["macro_f1"] * 100
        test_acc = results_dict[task_name]["test_metrics"]["accuracy"] * 100
        
        title_str = f"{task_name.replace('_', ' ').title()} Matrix\nAcc: {test_acc:.1f}% | Macro-F1: {test_f1:.1f}%"
        ax.set_title(title_str, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=11, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    output_fig_path = os.path.join(PLOTS_DIR, "confusion_matrices_lr.png")
    plt.savefig(output_fig_path, dpi=300)
    plt.close()
    print(f"\n[+] High-resolution Confusion Matrix Plot saved -> {output_fig_path}")


def verify_fast_inference(models: dict):
    """
    Tests live offline inference with serialized artifacts.
    Evaluates execution latency and checks sample Bangla sentences.
    """
    print(f"\n=======================================================")
    print(f"[*] Acceptance Criteria: Fast Inference Verification")
    print(f"=======================================================")
    
    test_cases = [
        "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!",
        "বইটি অসম্ভব সুন্দর, পড়ে অনেক কিছু শিখতে পারলাম। সবাইকে পড়ার অনুরোধ রইলো।",
        "তোদের মতো দেশদ্রোহীদের প্রকাশ্যে ফাঁসি দেওয়া উচিত, তোরা সমাজের কীট!"
    ]
    
    for sentence in test_cases:
        print(f"\n[Input Text]: \"{sentence}\"")
        t0 = time.perf_counter()
        
        predictions = {}
        for task_name in ["sentiment", "sarcasm", "hate_speech"]:
            vec = models[task_name]["vec"]
            clf = models[task_name]["clf"]
            config = TASKS[task_name]
            
            # Vectorize & predict
            x_vec = vec.transform([sentence])
            pred_idx = clf.predict(x_vec)[0]
            pred_label = config["class_names"][pred_idx]
            
            # Probabilities if available
            probs = clf.predict_proba(x_vec)[0]
            confidence = probs[pred_idx] * 100
            predictions[task_name] = f"{pred_label} ({confidence:.1f}%)"
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"  ├─ Sentiment:   {predictions['sentiment']}")
        print(f"  ├─ Sarcasm:     {predictions['sarcasm']}")
        print(f"  ├─ Hate Speech: {predictions['hate_speech']}")
        print(f"  └─ Total Latency (all 3 models): {elapsed_ms:.2f} ms")


def main():
    print("=" * 60)
    print("  PHASE 3: TF-IDF + LOGISTIC REGRESSION TRAINING PIPELINE")
    print("=" * 60)
    total_start = time.time()
    
    all_results = {}
    all_cms = {}
    trained_models = {}
    
    for task_name, config in TASKS.items():
        res, cm, vec, clf = train_and_evaluate_task(task_name, config)
        all_results[task_name] = res
        all_cms[task_name] = cm
        trained_models[task_name] = {"vec": vec, "clf": clf}
    
    # Save results json
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Master Evaluation Results saved -> {RESULTS_PATH}")
    
    # Generate Confusion Matrix Plots
    plot_all_confusion_matrices(all_results, all_cms)
    
    # Run acceptance verification
    verify_fast_inference(trained_models)
    
    total_elapsed = time.time() - total_start
    print(f"\n=======================================================")
    print(f"[✔] PHASE 3 PIPELINE COMPLETED SUCCESSFULLY in {total_elapsed:.1f}s")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
