"""
Exploratory Data Analysis (EDA) & N-Gram Statistics Module for Phase 2.
Implements:
- Class distribution statistics across Train, Val, Test.
- Sentence length distributions (mean, median, 95th percentile).
- Vocabulary size & Type-Token Ratio (TTR).
- Top Unigrams, Bigrams, and Trigrams (Lab 2 Language Modeling).
- Publication-quality EDA charts saved to Project files/eda_plots/.
- Data leakage verification between Train and Test sets.
Lead: Siyam Khan (ID: 2107120)
"""

import os
import sys
import json
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Configure font supporting Bengali
plt.rcParams['font.family'] = 'Nirmala UI'
plt.rcParams['axes.unicode_minus'] = False

# Import tokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import tokenize_bangla, remove_stopwords, get_ngrams

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CLEAN_DATA_DIR = os.path.join(WORKSPACE_ROOT, "Project files", "cleaned_data")
EDA_PLOTS_DIR = os.path.join(WORKSPACE_ROOT, "Project files", "eda_plots")
EDA_JSON_PATH = os.path.join(WORKSPACE_ROOT, "Project files", "eda_summary.json")

os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

TASK_LABELS = {
    "sentiment": {0: "Negative", 1: "Neutral", 2: "Positive"},
    "sarcasm": {0: "Non-Sarcastic", 1: "Sarcastic"},
    "hate_speech": {0: "Non-Hate", 1: "Hate Speech"}
}


def analyze_task(task_name: str) -> dict:
    print(f"\nAnalyzing Task: {task_name.upper()}...")
    task_dir = os.path.join(CLEAN_DATA_DIR, task_name)
    splits = ["train", "val", "test"]
    dfs = {s: pd.read_csv(os.path.join(task_dir, f"{s}.csv")) for s in splits}

    task_summary = {
        "splits": {},
        "train_text_stats": {},
        "vocabulary_stats": {},
        "top_unigrams": [],
        "top_bigrams": [],
        "top_trigrams": [],
        "data_leakage": {}
    }

    # 1. Split sizes and class distributions
    for s in splits:
        df = dfs[s]
        total = len(df)
        dist = df['label'].value_counts().to_dict()
        named_dist = {TASK_LABELS[task_name].get(k, str(k)): int(v) for k, v in dist.items()}
        pct_dist = {k: round((v / total) * 100, 2) for k, v in named_dist.items()}
        task_summary["splits"][s] = {
            "total_rows": total,
            "counts": named_dist,
            "percentages": pct_dist
        }

    # 2. Text & Sequence length stats on Train split
    train_df = dfs["train"]
    tokenized_sentences = [tokenize_bangla(str(t), keep_punct=False) for t in train_df["text"]]
    word_counts = [len(tokens) for tokens in tokenized_sentences]
    char_counts = [len(str(t)) for t in train_df["text"]]

    task_summary["train_text_stats"] = {
        "word_count": {
            "min": int(np.min(word_counts)),
            "max": int(np.max(word_counts)),
            "mean": round(float(np.mean(word_counts)), 2),
            "median": float(np.median(word_counts)),
            "std": round(float(np.std(word_counts)), 2),
            "p95": float(np.percentile(word_counts, 95))
        },
        "char_count": {
            "min": int(np.min(char_counts)),
            "max": int(np.max(char_counts)),
            "mean": round(float(np.mean(char_counts)), 2),
            "median": float(np.median(char_counts))
        }
    }

    # 3. Vocabulary & N-Gram statistics
    unigram_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()
    total_tokens = 0

    for tokens in tokenized_sentences:
        filtered = remove_stopwords(tokens, preserve_negation=True)
        total_tokens += len(tokens)
        unigram_counter.update(filtered)
        if len(filtered) >= 2:
            bigrams = [" ".join(bg) for bg in get_ngrams(filtered, 2)]
            bigram_counter.update(bigrams)
        if len(filtered) >= 3:
            trigrams = [" ".join(tg) for tg in get_ngrams(filtered, 3)]
            trigram_counter.update(trigrams)

    unique_tokens = len(unigram_counter)
    ttr = round(unique_tokens / total_tokens, 4) if total_tokens > 0 else 0

    task_summary["vocabulary_stats"] = {
        "total_tokens": total_tokens,
        "unique_vocabulary_size": unique_tokens,
        "type_token_ratio_ttr": ttr
    }

    task_summary["top_unigrams"] = [{"word": w, "count": int(c)} for w, c in unigram_counter.most_common(20)]
    task_summary["top_bigrams"] = [{"ngram": bg, "count": int(c)} for bg, c in bigram_counter.most_common(20)]
    task_summary["top_trigrams"] = [{"ngram": tg, "count": int(c)} for tg, c in trigram_counter.most_common(10)]

    # 4. Data Leakage Check (Overlap between Train and Test sets)
    train_texts = set(train_df["text"].str.strip())
    test_texts = set(dfs["test"]["text"].str.strip())
    overlap = len(train_texts.intersection(test_texts))
    leakage_pct = round((overlap / len(test_texts)) * 100, 2) if len(test_texts) > 0 else 0
    task_summary["data_leakage"] = {
        "train_unique_texts": len(train_texts),
        "test_unique_texts": len(test_texts),
        "overlap_count": overlap,
        "leakage_percentage": leakage_pct,
        "status": "PASS (Clean Split)" if leakage_pct < 5.0 else "WARNING"
    }

    return task_summary, dfs, word_counts, unigram_counter, bigram_counter


def plot_class_distributions(all_summaries: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    fig.patch.set_facecolor('#FFFFFF')
    colors = {
        "sentiment": ['#EF4444', '#94A3B8', '#10B981'],
        "sarcasm": ['#3B82F6', '#8B5CF6'],
        "hate_speech": ['#0D9488', '#E11D48']
    }

    for idx, (task, summary) in enumerate(all_summaries.items()):
        ax = axes[idx]
        train_counts = summary["splits"]["train"]["counts"]
        labels = list(train_counts.keys())
        counts = list(train_counts.values())
        task_colors = colors.get(task, ['#64748B'] * len(labels))

        bars = ax.bar(labels, counts, color=task_colors, edgecolor='#1E293B', linewidth=1.2, width=0.55)
        ax.set_title(f"{task.replace('_', ' ').title()} Class Distribution (Train)", fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel("Number of Samples", fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='semibold')

    plt.tight_layout()
    plot_path = os.path.join(EDA_PLOTS_DIR, "class_distributions.png")
    plt.savefig(plot_path, facecolor='#FFFFFF')
    plt.close()
    print(f"Saved Plot: {plot_path}")


def plot_sentence_lengths(word_counts_dict: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)
    fig.patch.set_facecolor('#FFFFFF')
    colors = ['#3B82F6', '#8B5CF6', '#E11D48']

    for idx, (task, counts) in enumerate(word_counts_dict.items()):
        ax = axes[idx]
        # Filter 99th percentile for clean display
        cap = np.percentile(counts, 99)
        filtered = [c for c in counts if c <= cap]

        ax.hist(filtered, bins=30, color=colors[idx], edgecolor='#1E293B', alpha=0.75, density=True)
        median_val = np.median(counts)
        mean_val = np.mean(counts)
        ax.axvline(median_val, color='#DC2626', linestyle='--', linewidth=1.5, label=f'Median: {median_val:.1f}')
        ax.axvline(mean_val, color='#047857', linestyle='-', linewidth=1.5, label=f'Mean: {mean_val:.1f}')

        ax.set_title(f"{task.replace('_', ' ').title()} Word Length Distribution", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Words per Sentence", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc='upper right', frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(EDA_PLOTS_DIR, "sentence_lengths.png")
    plt.savefig(plot_path, facecolor='#FFFFFF')
    plt.close()
    print(f"Saved Plot: {plot_path}")


def plot_top_ngrams(ngrams_dict: dict):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), dpi=200)
    fig.patch.set_facecolor('#FFFFFF')

    for row_idx, (task, (ug_counter, bg_counter)) in enumerate(ngrams_dict.items()):
        # Unigrams (Left)
        ax_ug = axes[row_idx, 0]
        top_ug = ug_counter.most_common(12)[::-1]
        words = [x[0] for x in top_ug]
        counts = [x[1] for x in top_ug]
        ax_ug.barh(words, counts, color='#3B82F6', edgecolor='#1E293B', height=0.6)
        ax_ug.set_title(f"{task.replace('_', ' ').title()}: Top Unigrams", fontsize=11, fontweight='bold')
        ax_ug.grid(axis='x', linestyle='--', alpha=0.5)

        # Bigrams (Right)
        ax_bg = axes[row_idx, 1]
        top_bg = bg_counter.most_common(12)[::-1]
        bgs = [x[0] for x in top_bg]
        bg_counts = [x[1] for x in top_bg]
        ax_bg.barh(bgs, bg_counts, color='#8B5CF6', edgecolor='#1E293B', height=0.6)
        ax_bg.set_title(f"{task.replace('_', ' ').title()}: Top Bigrams (N-Gram LM)", fontsize=11, fontweight='bold')
        ax_bg.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(EDA_PLOTS_DIR, "top_ngrams.png")
    plt.savefig(plot_path, facecolor='#FFFFFF')
    plt.close()
    print(f"Saved Plot: {plot_path}")


def main():
    print("="*60)
    print(">>> PHASE 2: EXPLORATORY DATA ANALYSIS & CORPUS METRICS")
    print("="*60)

    tasks = ["sentiment", "sarcasm", "hate_speech"]
    all_summaries = {}
    word_counts_dict = {}
    ngrams_dict = {}

    for task in tasks:
        summary, dfs, word_counts, ug_counter, bg_counter = analyze_task(task)
        all_summaries[task] = summary
        word_counts_dict[task] = word_counts
        ngrams_dict[task] = (ug_counter, bg_counter)

    # Save summary JSON
    with open(EDA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved Master EDA Summary JSON: {EDA_JSON_PATH}")

    # Generate Publication-Quality Plots
    print("\nGenerating Visualizations...")
    plot_class_distributions(all_summaries)
    plot_sentence_lengths(word_counts_dict)
    plot_top_ngrams(ngrams_dict)

    # Print Formatted Verification Summary Table
    print("\n" + "="*60)
    print(">>> PHASE 2 VERIFICATION SUMMARY TABLE")
    print("="*60)

    records = []
    for task in tasks:
        s = all_summaries[task]
        records.append({
            "Task": task.upper(),
            "Train Rows": f"{s['splits']['train']['total_rows']:,}",
            "Val Rows": f"{s['splits']['val']['total_rows']:,}",
            "Test Rows": f"{s['splits']['test']['total_rows']:,}",
            "Vocab Size": f"{s['vocabulary_stats']['unique_vocabulary_size']:,}",
            "Avg Words/Sent": f"{s['train_text_stats']['word_count']['mean']}",
            "P95 Words": f"{s['train_text_stats']['word_count']['p95']}",
            "Leakage": f"{s['data_leakage']['overlap_count']} ({s['data_leakage']['leakage_percentage']}%)"
        })

    summary_df = pd.DataFrame(records)
    print(summary_df.to_string(index=False))
    print("\n[OK] Phase 2 Execution Completed Successfully!")


if __name__ == "__main__":
    main()
