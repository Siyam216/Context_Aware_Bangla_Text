"""
Phase 4: Word2Vec Dense Embeddings Generator (PPMI + SVD Dense Skip-Gram Factorization)
Lead: Siyam Khan (ID: 2107120)
Academic Mapping: Lab 3 (Word2Vec Embeddings) & Lab 4 (PyTorch Pretrained Embeddings)

This module:
1. Collects a unified corpus across all 3 training splits (Sentiment, Sarcasm, Hate Speech).
2. Builds a comprehensive vocabulary of the top 20,000 words (+ <PAD>=0, <UNK>=1).
3. Constructs a symmetric word co-occurrence matrix with sliding context window (W=3).
4. Computes Positive Pointwise Mutual Information (PPMI) - equivalent to Skip-Gram with Negative Sampling.
5. Projects into 128-dimensional dense continuous vector space via Truncated SVD.
6. Serializes the embedding tensor ('bangla_word2vec.pt') and vocabulary ('word2idx.json') into 'Project files/saved_models/'.
7. Validates semantic embeddings with cosine similarity checks.
"""

import os
import sys
import time
import json
import re
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Ensure UTF-8 output encoding for Windows PowerShell
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

VOCAB_SIZE = 20000
EMBED_DIM = 128
WINDOW_SIZE = 3
TOKEN_PATTERN = re.compile(r'[\u0980-\u09FFa-zA-Z0-9]+|[!?]')


def train_word2vec_embeddings():
    print("=" * 65)
    print("  PHASE 4.1: BANGLA WORD2VEC DENSE EMBEDDINGS (PPMI-SVD)")
    print("=" * 65)
    t_start = time.time()

    # 1. Load all training texts
    print("[+] Ingesting training splits from all 3 tasks...")
    datasets = [
        os.path.join(CLEANED_DIR, "sentiment", "train.csv"),
        os.path.join(CLEANED_DIR, "sarcasm", "train.csv"),
        os.path.join(CLEANED_DIR, "hate_speech", "train.csv")
    ]
    
    all_texts = []
    for path in datasets:
        df = pd.read_csv(path)
        all_texts.extend(df['text'].dropna().astype(str).tolist())
    
    print(f"  [+] Total training sentences: {len(all_texts):,}")

    # 2. Tokenize and build vocabulary
    print("[+] Tokenizing sentences and building vocabulary...")
    word_freq = Counter()
    tokenized_corpus = []
    for text in all_texts:
        tokens = TOKEN_PATTERN.findall(text)
        if tokens:
            tokenized_corpus.append(tokens)
            word_freq.update(tokens)

    most_common = word_freq.most_common(VOCAB_SIZE)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    
    for idx, (word, count) in enumerate(most_common, start=2):
        word2idx[word] = idx
        idx2word[idx] = word

    total_vocab_size = len(word2idx)
    print(f"  [+] Vocabulary created: {total_vocab_size:,} tokens (including <PAD>=0, <UNK>=1)")
    print(f"  [+] Top 5 words by frequency: {[w for w, c in most_common[:5]]}")

    # 3. Construct Co-occurrence Matrix
    print(f"[+] Building co-occurrence matrix (window_size={WINDOW_SIZE})...")
    t_co = time.time()
    
    # Store co-occurrences efficiently
    row_indices = []
    col_indices = []
    data_counts = defaultdict(int)

    for tokens in tokenized_corpus:
        # Convert tokens to indices (ignoring <PAD> and <UNK> for co-occurrence)
        indices = [word2idx[t] for t in tokens if t in word2idx and word2idx[t] > 1]
        n_tokens = len(indices)
        for i in range(n_tokens):
            w_idx = indices[i]
            start = max(0, i - WINDOW_SIZE)
            end = min(n_tokens, i + WINDOW_SIZE + 1)
            for j in range(start, end):
                if i != j:
                    c_idx = indices[j]
                    # Map to 0-indexed vocabulary (w_idx - 2)
                    data_counts[(w_idx - 2, c_idx - 2)] += 1

    num_words = VOCAB_SIZE
    rows = []
    cols = []
    vals = []
    for (r, c), cnt in data_counts.items():
        rows.append(r)
        cols.append(c)
        vals.append(cnt)

    co_matrix = csr_matrix((vals, (rows, cols)), shape=(num_words, num_words), dtype=np.float32)
    print(f"  [+] Co-occurrence matrix built in {time.time() - t_co:.2f}s with {len(vals):,} non-zero cells.")

    # 4. Compute PPMI Matrix
    print("[+] Computing Positive Pointwise Mutual Information (PPMI)...")
    t_ppmi = time.time()
    total_cooc = co_matrix.sum()
    row_sums = np.array(co_matrix.sum(axis=1)).flatten()
    col_sums = np.array(co_matrix.sum(axis=0)).flatten()

    # Avoid zero division
    row_sums[row_sums == 0] = 1.0
    col_sums[col_sums == 0] = 1.0

    # Element-wise PPMI: max(0, log( (C_ij * total) / (C_i * C_j) ))
    co_matrix = co_matrix.tocoo()
    expected = (row_sums[co_matrix.row] * col_sums[co_matrix.col]) / total_cooc
    pmi_vals = np.log((co_matrix.data + 1e-8) / (expected + 1e-8))
    ppmi_vals = np.maximum(pmi_vals, 0)

    # Filter strictly positive values
    pos_mask = ppmi_vals > 0
    ppmi_matrix = csr_matrix(
        (ppmi_vals[pos_mask], (co_matrix.row[pos_mask], co_matrix.col[pos_mask])),
        shape=(num_words, num_words),
        dtype=np.float32
    )
    print(f"  [+] PPMI matrix computed in {time.time() - t_ppmi:.2f}s.")

    # 5. SVD Dimensionality Reduction (128 Dimensions)
    print(f"[+] Applying Truncated SVD to extract {EMBED_DIM}-dim dense word vectors...")
    t_svd = time.time()
    svd = TruncatedSVD(n_components=EMBED_DIM, random_state=42, algorithm="randomized", n_iter=7)
    dense_embeddings = svd.fit_transform(ppmi_matrix)
    print(f"  [+] Truncated SVD completed in {time.time() - t_svd:.2f}s. Explained variance: {svd.explained_variance_ratio_.sum()*100:.2f}%")

    # Normalize vectors to unit length
    norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_embeddings = dense_embeddings / norms

    # Construct full PyTorch embedding matrix of shape (total_vocab_size, EMBED_DIM)
    full_embed_matrix = np.zeros((total_vocab_size, EMBED_DIM), dtype=np.float32)
    # <PAD> at 0 remains all zeros
    # <UNK> at 1 gets small random values
    np.random.seed(42)
    full_embed_matrix[1] = np.random.normal(scale=0.01, size=EMBED_DIM)
    # Words 2..N get normalized SVD embeddings
    full_embed_matrix[2:] = normalized_embeddings

    # Convert to PyTorch Tensor
    embed_tensor = torch.tensor(full_embed_matrix, dtype=torch.float32)

    # 6. Save Artifacts
    embed_path = os.path.join(MODELS_DIR, "bangla_word2vec.pt")
    vocab_path = os.path.join(MODELS_DIR, "word2idx.json")
    idx_path = os.path.join(MODELS_DIR, "idx2word.json")

    torch.save(embed_tensor, embed_path)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx2word, f, ensure_ascii=False)

    print(f"\n[+] Serialized Embeddings -> {embed_path} ({os.path.getsize(embed_path)/(1024*1024):.2f} MB)")
    print(f"[+] Serialized Vocab Mapping -> {vocab_path} ({len(word2idx):,} words)")

    # 7. Semantic Cosine Similarity Verification
    def find_top_similar(target_word: str, top_k: int = 5):
        if target_word not in word2idx:
            return []
        t_idx = word2idx[target_word]
        t_vec = embed_tensor[t_idx].numpy()
        # Cosine similarity against all words
        scores = np.dot(normalized_embeddings, t_vec)
        top_indices = np.argsort(scores)[::-1][1:top_k+1]
        return [(idx2word[i + 2], round(float(scores[i]), 4)) for i in top_indices]

    print("\n[+] Semantic Word Similarity Checks:")
    test_words = ["ভালো", "খারাপ", "সুন্দর", "বই"]
    for tw in test_words:
        sims = find_top_similar(tw)
        sim_str = ", ".join([f"{w} ({s})" for w, s in sims])
        print(f"  - Similar to '{tw}': {sim_str}")

    print(f"\n[✔] Word2Vec Embeddings Pipeline Finished in {time.time() - t_start:.2f}s")
    return embed_tensor, word2idx, idx2word


if __name__ == "__main__":
    train_word2vec_embeddings()
