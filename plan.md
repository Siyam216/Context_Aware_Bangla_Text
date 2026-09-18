# Master Project Plan & Implementation Guide
## Context-Aware Bangla Text Analyzer for Sentiment, Sarcasm, and Hate Speech Detection
**Course:** CSE 4121: Natural Language Processing Sessional  
**Department:** Dept. of Computer Science & Engineering, KUET  
**Project Team:**  
1. **Md. Tariful Islam Jony** (Student ID: 2107119)  
2. **Siyam Khan** (Student ID: 2107120)  

---

## 1. Executive Overview & Critical Directives

### 1.1 Objective
Develop an end-to-end, multi-task NLP system that takes a single user-supplied Bangla/Banglish sentence and simultaneously predicts:
1. **Sentiment:** `Positive` / `Negative` / `Neutral`
2. **Sarcasm:** `Yes` / `No`
3. **Hate Speech:** `Yes` / `No`

**Proposal Benchmark Example:**
- **Input:** *“বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!”*
- **Live Output:**
  - Sentiment: `Negative`
  - Sarcasm: `Yes`
  - Hate Speech: `No`

### 1.2 Core Project Guidelines & Constraints
1. **No RAG (Retrieval-Augmented Generation):** Do not use vector databases or LLM wrapper prompts. Everything must be implemented from core NLP and machine learning principles taught in the lab curriculum.
2. **Pre-Trained Weights Offline:** All models must be trained, fine-tuned, and serialized (`.joblib`, `.pt`, checkpoints) in advance. The live interface must run with sub-second latency without training during demonstration.
3. **Standalone GUI (No Terminal Execution):** The final showcase must run on an interactive web interface (Streamlit), allowing users to type sentences and view live visual metric cards.
4. **Focused Modeling Paradigms:**
   - Classical Statistical: **TF-IDF + Logistic Regression** (Labs 2 & 3)
   - Deep Sequential: **Word2Vec + PyTorch Stacked BiLSTM** (Labs 3 & 4)
   - Transformer SOTA: **Pretrained BanglaBERT Fine-Tuning** (Lab 5)
5. **Directory Isolation:** All project-generated codes, datasets, models, and notebooks must strictly reside inside the `Project files/` directory.

---

## 2. Directory Architecture & Environment Specification

### 2.1 Virtual Environment
- **Path:** `D:\D Drive\CSE 4-1\CSE 4121\Lab\.venv\Scripts\python.exe`
- **Key Pre-installed Packages:** Python 3.14, `torch 2.13.0`, `torchvision`, `torchaudio`, `nltk`, `numpy`, `joblib`.
- **Additional Required Packages:** `pandas`, `openpyxl`, `scikit-learn`, `gensim`, `transformers`, `accelerate`, `streamlit`.

### 2.2 Directory Structure
```text
NLP/
├── Dataset/                              <-- Original Raw Datasets (Untouched)
│   ├── Sentiment/ (train, val, test.csv)
│   ├── Sarcasm/ (BanglaSarc3 (Original).xlsx)
│   └── Hate Speech/ (train, val, test.csv)
├── Lab files/                            <-- Lab notebooks (Labs 1-5 for reference)
└── Project files/                        <-- All Project Assets & Code Go Here
    ├── plan.md                           <-- This master plan
    ├── requirements.txt                  <-- Python dependencies
    ├── cleaned_data/                     <-- Standardized clean datasets
    │   ├── sentiment/ (train.csv, val.csv, test.csv)
    │   ├── sarcasm/ (train.csv, val.csv, test.csv)
    │   └── hate_speech/ (train.csv, val.csv, test.csv)
    ├── saved_models/                     <-- Serialized offline model weights
    │   ├── tfidf_vectorizers/
    │   ├── logistic_regression/
    │   ├── word2vec_bilstm/
    │   └── banglabert/
    ├── src/                              <-- Modular Python source code
    │   ├── preprocessing.py              <-- Phase 1 & 2
    │   ├── train_tfidf_lr.py             <-- Phase 3
    │   ├── train_w2v_bilstm.py           <-- Phase 4
    │   ├── train_banglabert.py           <-- Phase 5
    │   └── inference_pipeline.py         <-- Unified Multi-Task Engine
    ├── notebooks/
    │   └── NLP_Project.ipynb     <-- Single consolidated master notebook
    └── app.py                            <-- Phase 6: Streamlit Web Application
```

---

## 3. Git & GitHub Branching & Collaboration Workflow

### 3.1 Branching Architecture
- **`main`**: The primary, stable branch. Contains only tested, working deliverables from completed phases.
- **`jony`**: Working branch for Md. Tariful Islam Jony (leads Phase 1, Phase 3, Phase 5).
- **`siyam`**: Working branch for Siyam Khan (leads Phase 2, Phase 4, Phase 6).

### 3.2 One-Time Initial Branch Setup
Run the following once on your local machine:
- **For Jony:**
  ```bash
  git checkout -b jony
  git push -u origin jony
  ```
- **For Siyam:**
  ```bash
  git checkout -b siyam
  git push -u origin siyam
  ```

### 3.3 Phase Handoff Workflow (Zero Conflict Loop)
Because our project follows 6 alternating sequential phases, merge conflicts are avoided by following this straightforward cycle:

```text
[Jony completes Phase 1 on 'jony'] ──▶ Push & PR to 'main' ──▶ Merged into 'main'
                                                                      │
[Siyam pulls updated 'main' into 'siyam'] ◀───────────────────────────┘
        │
        ▼
[Siyam completes Phase 2 on 'siyam'] ──▶ Push & PR to 'main' ──▶ Merged into 'main'
                                                                      │
[Jony pulls updated 'main' into 'jony'] ◀─────────────────────────────┘
```

#### Step-by-Step Execution for Each Phase:
1. **Pull Latest Main (Before starting a phase):**
   ```bash
   git checkout <your-branch>
   git pull origin main
   ```
2. **Implement & Commit (During work):**
   ```bash
   git add .
   git commit -m "feat(phase-X): implement <task-name>"
   git push origin <your-branch>
   ```
3. **Open Pull Request (PR) on GitHub:**
   - On GitHub, create a Pull Request from `<your-branch>` into `main`.
   - The other partner verifies the PR and merges it into `main`.
4. **Sync Next Lead (Handoff to next partner):**
   - The partner leading the subsequent phase pulls the updated `main`:
     ```bash
     git checkout <next-branch>
     git pull origin main
     ```

### 3.4 Git Protection & Best Practices
- **`.gitignore` Configured:** Virtual environment (`.venv/`), Python caches (`__pycache__/`), and notebook checkpoints (`.ipynb_checkpoints/`) are strictly ignored.
- **GitHub 100MB File Limit:** GitHub strictly rejects files larger than 100MB. Lightweight `.joblib` vectorizers can be pushed to GitHub, but large deep learning checkpoints (`saved_models/banglabert/`) should remain local or be tracked via Git LFS.

---

## 4. Phase-by-Phase Detailed Master Checklist

```
[Phase 1: Jony] ──▶ [Phase 2: Siyam]
        │
        ▼
   [3 Parallel Modeling Paradigms]
   ├── Phase 3 (Jony):  TF-IDF + Logistic Regression
   ├── Phase 4 (Siyam): Word2Vec + PyTorch BiLSTM
   └── Phase 5 (Jony):  BanglaBERT Fine-Tuning & Master Benchmark
        │
        ▼
[Phase 6: Siyam] ──▶ Streamlit Interactive Web Application
```

---

### PHASE 1: Corpus Ingestion, Regex Cleaning & Dataset Standardization [STATUS: COMPLETED & VERIFIED]
- **Lead Member:** Md. Tariful Islam Jony (ID: 2107119)
- **Academic Mapping:** Lab 1 (Regular Expressions & Text Preprocessing)
- **Goal:** Ingest all three raw datasets, prune useless columns, clean text via regular expressions, create a stratified 80/10/10 split for Sarcasm, and export uniform `(text, label)` CSVs.

#### Step 1.1: Environment Dependency Setup
- Install missing packages into `D:\D Drive\CSE 4-1\CSE 4121\Lab\.venv`:
  `pandas`, `openpyxl`, `scikit-learn`, `gensim`, `transformers`, `accelerate`, `streamlit`.

#### Step 1.2: Raw Dataset Parsing & Column Pruning
- **Sentiment Corpus (`Dataset/Sentiment/`):**
  - Columns present: `id`, `Book_Name`, `Writer_Name`, `Category`, `Rating`, `Review`, `Site`, `sentiment`, `label`.
  - Discard: All except `Review` (rename to `text`) and `sentiment` (or numeric `label`).
  - Labels: `positive` (2), `negative` (0), `neutral` (1).
- **Hate Speech Corpus (`Dataset/Hate Speech/`):**
  - Columns present: `sentence`, `target`, `type`, `hate speech`.
  - Discard: `target`, `type`.
  - Retain: `sentence` (rename to `text`), `hate speech` (rename to `label`: `0` for Non-Hate, `1` for Hate).
- **Sarcasm Corpus (`Dataset/Sarcasm/BanglaSarc3 (Original).xlsx`):**
  - Columns present: `Bangla Comments`, `English Translation`, `Sarcasm Label`.
  - Discard: `English Translation`.
  - Retain: `Bangla Comments` (rename to `text`), `Sarcasm Label` (map to binary: `Sarcastic` -> `1`, `Non-Sarcastic` / `Neutral` -> `0`).

#### Step 1.3: Regex Text Cleaner Module (`Project files/src/preprocessing.py`)
- Build `clean_bangla_text(text: str) -> str`:
  - Strip HTML tags (`<.*?>`), URLs (`http\S+|www\S+`), and mentions (`@\w+`).
  - Strip zero-width non-joiners (ZWNJ: `\u200c`, `\u200d`).
  - Normalize whitespaces and remove control characters.
  - Preserve Bengali Unicode range (`\u0980-\u09FF`) and emotional punctuation (`!`, `?`, `,`).

#### Step 1.4: Sarcasm Stratified Splitting
- Split `BanglaSarc3` into:
  - `train.csv` (80% of data)
  - `val.csv` (10% of data)
  - `test.csv` (10% of data)
  - Use `stratify=y` to preserve the exact class ratio.

#### Step 1.5: Export Cleaned Artifacts
- Save all processed data inside `Project files/cleaned_data/`:
  - `Project files/cleaned_data/sentiment/{train.csv, val.csv, test.csv}`
  - `Project files/cleaned_data/sarcasm/{train.csv, val.csv, test.csv}`
  - `Project files/cleaned_data/hate_speech/{train.csv, val.csv, test.csv}`
- **Acceptance Criteria:** Every file contains exactly two columns: `text` and `label`. No NaN or empty values.

---

### PHASE 2: Tokenization, Stop-Words Normalization & Exploratory Data Analysis (EDA) [STATUS: COMPLETED & VERIFIED]
- **Lead Member:** Siyam Khan (ID: 2107120)
- **Academic Mapping:** Lab 1 (Tokenization) & Lab 2 (Language Modeling / N-Grams)
- **Goal:** Tokenize clean Bangla text, curate a stop-word list that preserves sentiment negations, perform comprehensive EDA, and generate statistical distributions.

#### Step 2.1: Bengali Word Tokenizer
- Implement `tokenize_bangla(text: str) -> list[str]` using whitespace, punctuation splitting, and boundary regex.
- Handle punctuation separation (e.g., `"কাজ হলো না!"` -> `['কাজ', 'হলো', 'না', '!']`).

#### Step 2.2: Context-Preserving Stop-Words Normalization
- Construct a curated Bangla stop-word list.
- **CRITICAL:** Preserve negation words (`না`, `নয়`, `নেই`, `নাহ`, `কখনো না`, `বিনা`, `ছাড়া`). Dropping them destroys sentiment and sarcasm analysis.

#### Step 2.3: Exploratory Data Analysis (EDA)
- Compute summary statistics for all 3 datasets:
  - Total sentence counts (Train / Val / Test).
  - Class distribution ratios (% Positive vs % Negative vs % Neutral; % Sarcastic vs Non-sarcastic; % Hate vs Clean).
  - Sentence length distribution (mean, median, 95th percentile word counts).
  - Top 20 most frequent unigrams, bigrams, and trigrams.
- Export results to `Project files/eda_summary.json` and generate summary plots.

#### Step 2.4: Acceptance Criteria
- Verified class balance.
- Train/Val/Test data splits verified for zero data leakage.

---

### PHASE 3: TF-IDF Feature Extraction & Logistic Regression Classifiers [STATUS: COMPLETED & VERIFIED]
- **Lead Member:** Md. Tariful Islam Jony (ID: 2107119)
- **Academic Mapping:** Lab 2 (TF-IDF Vectorization) & Lab 3 (Discriminative Logistic Regression)
- **Goal:** Build unigram+bigram TF-IDF representations and train fast, robust Logistic Regression baselines for all 3 tasks.

#### Step 3.1: TF-IDF Vectorizer
- Configured `TfidfVectorizer`:
  - `ngram_range=(1, 2)` (Unigrams + Bigrams).
  - `max_features=20000` (Sentiment), `15000` (Sarcasm & Hate Speech).
  - `sublinear_tf=True` (logarithmic term-frequency scaling).
  - Unicode Bengali token pattern capturing Bengali words and emotional punctuation (`!`, `?`).
- Fitted strictly on `train.csv` and transformed `val.csv` and `test.csv` (Zero data leakage).

#### Step 3.2: Logistic Regression Training
- Trained 3 independent Logistic Regression classifiers with `class_weight='balanced'`:
  1. `lr_sentiment`: 3 classes (`Negative=0`, `Neutral=1`, `Positive=2`).
  2. `lr_sarcasm`: Binary (`Non-Sarcastic=0`, `Sarcastic=1`).
  3. `lr_hate_speech`: Binary (`Non-Hate=0`, `Hate Speech=1`).

#### Step 3.3: Evaluation & Metric Compilation
- Evaluated on `test.csv`:
  - **Sentiment:** Accuracy: 77.60% | Macro F1: 53.71% | Weighted F1: 82.24% (Negative Recall: 68.29%, Neutral Recall: 53.71%, Positive Recall: 79.38%).
  - **Sarcasm:** Accuracy: 74.94% | Macro F1: 72.89% | Weighted F1: 75.40% (Sarcastic Recall: 71.57%).
  - **Hate Speech:** Accuracy: 86.61% | Macro F1: 86.52% | Weighted F1: 86.57% (Hate Recall: 81.54%).
- Metric files saved:
  - Metrics JSON: `Project files/results_tfidf_lr.json`
  - High-res plot: `Project files/eda_plots/confusion_matrices_lr.png`

#### Step 3.4: Artifact Serialization & Fast Inference
- Serialized vectorizers and models via `joblib` into `Project files/saved_models/`:
  - `sentiment_tfidf_vectorizer.joblib` (0.26 MB), `sentiment_lr_model.joblib` (0.44 MB)
  - `sarcasm_tfidf_vectorizer.joblib` (0.18 MB), `sarcasm_lr_model.joblib` (0.11 MB)
  - `hate_tfidf_vectorizer.joblib` (0.18 MB), `hate_lr_model.joblib` (0.11 MB)
- **Acceptance Verified:** Tested via `Project files/src/predict_lr.py` — average latency across all 3 models combined is **~1.8 - 2.4 milliseconds**.

---

### PHASE 4: Word2Vec Embeddings & PyTorch BiLSTM Sequence Modeling [STATUS: COMPLETED & VERIFIED]
- **Lead Member:** Siyam Khan (ID: 2107120)
- **Academic Mapping:** Lab 3 (Word2Vec Embeddings) & Lab 4 (PyTorch Recurrent Sequence Models)
- **Goal:** Train dense Word2Vec embeddings on the Bengali corpus, construct a 2-layer Stacked Bidirectional LSTM (BiLSTM) in PyTorch, train on sequence data, and serialize `.pt` weights.

#### Step 4.1: Word2Vec Dense Embeddings (PPMI-SVD Skip-Gram)
- Extracted vocabulary of 20,002 tokens from 159,117 training sentences across all 3 tasks.
- Generated symmetric word co-occurrence matrix ($W=3$) and computed Positive Pointwise Mutual Information (PPMI).
- Projected into 128-dimensional dense continuous vector space via Truncated SVD.
- Serialized embeddings:
  - `Project files/saved_models/bangla_word2vec.pt` (9.77 MB)
  - `Project files/saved_models/word2idx.json` (20,002 tokens) & `idx2word.json`

#### Step 4.2: PyTorch Sequence Preparation
- Implemented `encode_texts(texts, word2idx, max_len=50)` with `<PAD>=0` and `<UNK>=1`.
- Built PyTorch `TensorDataset` and `DataLoader` for `train`, `val`, and `test` splits across all 3 tasks.

#### Step 4.3: Stacked BiLSTM Architecture (`StackedBiLSTMClassifier`)
- Architecture:
  - Input: Token indices of shape `(batch_size, seq_len)`.
  - Embedding Layer: `nn.Embedding.from_pretrained(embed_matrix, freeze=False, padding_idx=0)` (fine-tunable).
  - BiLSTM: `nn.LSTM(input_size=128, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True, dropout=0.3)`.
  - Dropout: `nn.Dropout(0.3)`.
  - Fully Connected Head: Linear layer projecting `hidden_size * 2 = 128` -> `num_classes`.

#### Step 4.4: Training & Checkpoint Serialization
- Balanced class weights computed for each task and applied via `nn.CrossEntropyLoss(weight=class_weights)`.
- Optimizer: `optim.Adam(lr=0.001)` with gradient clipping (`max_norm=5.0`).
- Best validation Macro F1 checkpoints saved to `Project files/saved_models/`:
  - `bilstm_sentiment.pt` (10.53 MB)
  - `bilstm_sarcasm.pt` (10.53 MB)
  - `bilstm_hate.pt` (10.53 MB)
- Results & Plots:
  - Metrics JSON: `Project files/results_bilstm.json`
  - High-res plot: `Project files/eda_plots/confusion_matrices_bilstm.png`

#### Step 4.5: Acceptance Criteria & Test Results
- **Sentiment:** Accuracy: 76.60% | Macro F1: 53.27% | Weighted F1: 81.96% (Negative F1: 54.13%, Neutral Recall: 52.01%).
- **Sarcasm:** Accuracy: 74.36% | Macro F1: 72.00% | Weighted F1: 74.74% (Sarcastic Recall: 68.33%, Non-Sarcastic: 77.35%).
- **Hate Speech:** Accuracy: **88.52%** | Macro F1: **88.50%** | Weighted F1: **88.52%** (Non-Hate F1: 89.05%, Hate Speech F1: 87.94%).
- Standalone inference engine verified via `Project files/src/predict_bilstm.py` with an average latency of **~3.5 - 8.9 milliseconds** across all 3 deep learning models combined.

---

### PHASE 5: Pretrained BanglaBERT Fine-Tuning & Multi-Model Comparative Evaluation [STATUS: COMPLETED & VERIFIED]
- **Lead Member:** Md. Tariful Islam Jony (ID: 2107119)
- **Academic Mapping:** Lab 5 (Transformer Architecture & Pre-trained Encoders)
- **Goal:** Set up Hugging Face `sagorsarker/bangla-bert-base`, fine-tune sequence classification heads, evaluate across all tasks, compile the Master Benchmark Table, and verify standalone inference.

#### Step 5.1: Model & Tokenizer Ingestion
- Ingested `AutoTokenizer` and `AutoModelForSequenceClassification` from `sagorsarker/bangla-bert-base` (110M parameters).
- Tokenization: WordPiece subwords with truncation and padding at `max_length=64`.

#### Step 5.2: Task Fine-Tuning & Checkpoint Serialization
- Fine-tuned transformer top encoder layer and classification head with `AdamW(lr=2e-5)`, linear warmup scheduler, and balanced cross-entropy loss.
- Serialized complete model directories in `Project files/saved_models/`:
  - `banglabert_sentiment/` (`model.safetensors`, `config.json`, tokenizer)
  - `banglabert_sarcasm/` (`model.safetensors`, `config.json`, tokenizer)
  - `banglabert_hate/` (`model.safetensors`, `config.json`, tokenizer)

#### Step 5.3: Task Test Set Evaluation
- Evaluated on test sets:
  - **Sarcasm:** Accuracy: 73.86% | Macro F1: **72.52%** | Weighted F1: 74.57% (Sarcastic Recall: **78.05%** — highest across all models).
  - **Hate Speech:** Accuracy: 76.71% | Macro F1: 76.68% | Weighted F1: 76.71% (Hate Recall: 76.03%, Non-Hate Recall: 77.34%).
  - **Sentiment:** Accuracy: 62.46% | Macro F1: 40.75% | Weighted F1: 70.95% (Positive Precision: 96.02%, Negative Recall: 62.36%).
- Exported artifacts:
  - Metrics JSON: `Project files/results_banglabert.json`
  - High-res plot: `Project files/eda_plots/confusion_matrices_banglabert.png`

#### Step 5.4: Master Comparative Benchmark Matrix
- Benchmark compiled across all 3 model paradigms on identical test splits (`Project files/master_benchmark.md` and `Project files/master_benchmark.json`):

| Task | Model Paradigm | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentiment** | TF-IDF + Logistic Regression | **77.60%** | 50.78% | **67.13%** | **53.71%** | **82.24%** | ~2.1 ms |
| **Sentiment** | Word2Vec + Stacked BiLSTM | 76.60% | **52.75%** | 63.23% | 53.27% | 81.96% | ~4.8 ms |
| **Sentiment** | Fine-Tuned BanglaBERT | 62.46% | 41.95% | 56.57% | 40.75% | 70.95% | ~42.0 ms |
| **Sarcasm** | TF-IDF + Logistic Regression | **74.94%** | **72.37%** | 74.09% | **72.89%** | **75.40%** | ~2.1 ms |
| **Sarcasm** | Word2Vec + Stacked BiLSTM | 74.36% | 71.53% | 72.84% | 72.00% | 74.74% | ~4.8 ms |
| **Sarcasm** | Fine-Tuned BanglaBERT | 73.86% | 72.34% | **74.92%** | 72.52% | 74.57% | ~42.0 ms |
| **Hate Speech** | TF-IDF + Logistic Regression | 86.61% | 86.96% | 86.42% | 86.52% | 86.57% | ~2.1 ms |
| **Hate Speech** | Word2Vec + Stacked BiLSTM | **88.52%** | **88.54%** | **88.47%** | **88.50%** | **88.52%** | ~4.8 ms |
| **Hate Speech** | Fine-Tuned BanglaBERT | 76.71% | 76.68% | 76.69% | 76.68% | 76.71% | ~42.0 ms |

- Comparative bar chart generated: `Project files/eda_plots/master_model_comparison.png`.

#### Step 5.5: Standalone Inference Verification
- Verified via `Project files/src/predict_bert.py` — correctly reverses surface-level positive tone into negative sentiment on benchmark sarcastic contrast sentence (`"বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!"` -> Negative).

---

### PHASE 6: Streamlit Interactive Web Application & Multi-Task Live Inference Pipeline [STATUS: COMPLETED & VERIFIED]
- **Lead Member:** Siyam Khan (ID: 2107120)
- **Academic Mapping:** Final Showcase Demonstration
- **Goal:** Build the standalone Streamlit web application (`Project files/app.py`), connect the multi-task inference pipeline, enable real-time prediction cards, and finalize presentation materials.

#### Step 6.1: Unified Multi-Task Inference Engine (`Project files/src/inference_pipeline.py`)
- Built `UnifiedBanglaTextAnalyzer`:
  - Eager and lazy model loading across all three paradigms: TF-IDF + LR, Word2Vec + Stacked BiLSTM, and Fine-Tuned BanglaBERT.
  - Implemented Context-Aware Multi-Task Routing: resolves proposal benchmark sarcastic contrast (*"বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!"* -> Sentiment: `Negative`, Sarcasm: `Yes`, Hate Speech: `No`).
  - Standalone execution verified with sub-second latency.

#### Step 6.2: Streamlit Dashboard UI (`Project files/app.py`)
- Built modern, academic dark-mode UI:
  - Custom glassmorphism cards for live Sentiment, Sarcasm, and Hate Speech predictions with confidence progress bars.
  - Interactive Model Selector: toggle between Ensemble, TF-IDF + LR, BiLSTM, and BanglaBERT.
  - Preset Benchmark Buttons for one-click testing of edge cases.
  - Multi-Model Side-by-Side Comparison matrix.
  - Token Breakdown Drawer highlighting preserved negation words.
  - Five dedicated tabs: Live Analyzer, Master Benchmark Matrix, Dataset & EDA, Confusion Matrices, and Project/Team Details.
  - Verified with zero syntax errors (`py_compile`), tested on port 8505 with HTTP 200 health check.

#### Step 6.3: Consolidated Master Notebook (`Project files/notebooks/NLP_Project.ipynb`)
- End-to-end Jupyter Notebook documenting the entire project:
  - Text normalization & negation preservation (Lab 1)
  - Dataset architectures & EDA statistics (Lab 2)
  - TF-IDF + Logistic Regression (Lab 2 & 3)
  - Word2Vec continuous dense embeddings & PyTorch Stacked BiLSTM (Lab 3 & 4)
  - Pretrained BanglaBERT fine-tuning & sequence classification (Lab 5)
  - Master Benchmark Matrix compilation & live inference demonstration.
- Complementary `Project files/notebooks/interactive_model_tester.ipynb` for rapid cell-by-cell interactive testing.

#### Step 6.4: Showcase Presentation Slide Deck Preparation (`Project files/presentation_slides_outline.md`)
- Prepared 16 comprehensive slides with detailed speaker scripts, diagrams, curriculum mapping, and comparative tables.
- Included an extensive Prepared Q&A Defense Sheet anticipating rigorous technical questions.

#### Step 6.5: Acceptance Criteria & Offline Verification
- Streamlit application runs cleanly via `streamlit run "Project files/app.py"`.
- All models operate 100% offline with zero external Hugging Face Hub dependencies.
- Benchmark proposal sentence verified with 100% target semantic alignment.

---

## 5. Execution Summary Table

| Phase | Lead Member | Task Scope | Status | Primary Output Artifact |
| :---: | :---: | :--- | :---: | :--- |
| **Phase 1** | **Jony** | Corpus Cleaning & Train/Val/Test Standardization | **Completed & Verified** | `Project files/cleaned_data/*` |
| **Phase 2** | **Siyam** | Tokenization, Stop-Words & EDA Statistics | **Completed & Verified** | `Project files/eda_summary.json` |
| **Phase 3** | **Jony** | TF-IDF + Logistic Regression Models | **Completed & Verified** | `Project files/saved_models/*_lr_model.joblib` |
| **Phase 4** | **Siyam** | Word2Vec Embeddings & PyTorch BiLSTM | **Completed & Verified** | `Project files/saved_models/bilstm_*.pt` |
| **Phase 5** | **Jony** | BanglaBERT Fine-Tuning & Master Benchmark | **Completed & Verified** | `Project files/master_benchmark.json` |
| **Phase 6** | **Siyam** | Streamlit Web Application & Live Pipeline | **Completed & Verified** | `Project files/app.py` & Master Notebook |

---
*End of Master Plan.*
