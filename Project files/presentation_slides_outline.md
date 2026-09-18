# 🇧🇩 Presentation Slide Deck Outline
## Context-Aware Bangla Text Analyzer for Sentiment, Sarcasm, and Hate Speech Detection

- **Course:** CSE 4121: Natural Language Processing Sessional
- **Academic Term:** 4th Year, 1st Term (CSE 4-1)
- **Department:** Department of Computer Science & Engineering, KUET
- **Project Team:**
  - **Md. Tariful Islam Jony** (Student ID: 2107119)
  - **Siyam Khan** (Student ID: 2107120)

---

## Slide 1: Title & Academic Header
- **Slide Title:** Context-Aware Bangla Text Analyzer: Multi-Task Detection of Sentiment, Sarcasm, and Hate Speech Across Three Progressive NLP Paradigms
- **Visuals:** KUET Logo, Project Title Banner, Team Member Names & Student IDs.
- **Key Talking Points:**
  - Good morning / afternoon respected audience and peers.
  - Today, we present our end-to-end NLP sessional project addressing the fundamental challenges of understanding informal, colloquial, and sarcastic Bengali social media text.
  - We systematically compare three distinct paradigms from the CSE 4121 syllabus: Statistical (TF-IDF + LR), Sequential Recurrent (Word2Vec + Stacked BiLSTM), and Modern Transformer (BanglaBERT), integrated into a live context-aware inference dashboard.

---

## Slide 2: The Core Linguistic Challenge & Motivation
- **Slide Title:** Why is Bengali Social Media Text Uniquely Difficult?
- **Key Content:**
  1. **Surface Polarity vs. True Intent (Sarcastic Inversion):**
     - Typical sentiment models rely on surface keywords.
     - *Example:* "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!"
     - Surface words like *“অসাধারণ”* deceive traditional lexicons into predicting **Positive**, ignoring the negative outcome (*“কাজ হলো না”*).
  2. **Morphological Complexity & Spelling Variations:**
     - Colloquial Bengali features inflections, colloquial contractions, and lack of capitalization.
  3. **High Co-occurrence of Harmful Content:**
     - Sarcasm, toxicity, and sentiment are deeply intertwined in online discourse; evaluating them in isolation yields incomplete understanding.

---

## Slide 3: Project Objectives & Proposal Benchmark
- **Slide Title:** Objectives & Benchmark Goal
- **Key Content:**
  - **Multi-Task Objective:** Given any raw Bengali or Banglish sentence, simultaneously predict:
    1. **Sentiment Polarity:** Positive, Negative, or Neutral.
    2. **Sarcasm Presence:** Yes (Sarcastic) or No (Non-Sarcastic).
    3. **Hate Speech / Toxicity:** Yes (Hate Speech) or No (Non-Hate).
  - **Proposal Target Benchmark Verification:**
    - **Input:** *“বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!”*
    - **Expected & Achieved Output:**
      - **Sentiment:** `Negative` *(Contextually Resolved)*
      - **Sarcasm:** `Yes` *(Sarcastic)*
      - **Hate Speech:** `No` *(Non-Hate)*
  - **Strict Constraints Satisfied:**
    - No external LLM prompt wrappers or RAG databases; strictly built upon core NLP curriculum.
    - Zero online dependencies during evaluation; 100% offline local weights.
    - Live sub-second inference via an interactive Streamlit dashboard.

---

## Slide 4: Academic Integration with CSE 4121 Syllabus
- **Slide Title:** Seamless Mapping to NLP Lab Curriculum
- **Table / Visual Mapping:**
  | Lab Session | Core Topic Taught in Lab | Direct Implementation in Project |
  | :--- | :--- | :--- |
  | **Lab 1** | Text Normalization & Regex | Bengali Unicode regex cleaner, zero-width stripper, **negation preservation pipeline** |
  | **Lab 2** | N-Gram Modeling & TF-IDF | Sublinear Unigram + Bigram TF-IDF vectorizer (20,000 features, sublinear TF) |
  | **Lab 3** | Embeddings & Logistic Regression | 128-d dense Word2Vec continuous embeddings + Balanced Class-Weighted LR |
  | **Lab 4** | Recurrent Sequence Models | PyTorch 2-layer Stacked Bidirectional LSTM with dropout and gradient clipping |
  | **Lab 5** | Pretrained Transformers | Fine-tuned `sagorsarker/bangla-bert-base` (110M parameters) with sequence classification heads |

---

## Slide 5: Dataset Architecture & Normalization Pipeline
- **Slide Title:** Corpus Standardization & Linguistic Preprocessing
- **Key Content:**
  - **Three Curated Datasets:**
    - **Sentiment:** 312,128 sentences (Balanced across 90% positive skew using balanced loss weighting).
    - **Sarcasm:** 12,096 sentences from Bengali social media (BanglaSarc3).
    - **Hate Speech:** 50,309 sentences from political, religious, and social commentary.
  - **Unified Split Protocol:** Strict 80% Train, 10% Validation, 10% Test across all tasks.
  - **Negation-Preserving Preprocessor:**
    - Normalizes Bengali characters, strips noisy URLs, mentions, HTML tags, and non-printable bytes.
    - **Critical Innovation:** Explicitly protects negation markers (`না`, `নয়`, `নেই`, `নাহ`, `কখনো না`, `বিনা`, `ছাড়া`). Removing these tokens as standard "stopwords" destroys sarcasm and sentiment analysis!

---

## Slide 6: Modeling Paradigm 1 — Statistical Baseline (TF-IDF + LR)
- **Slide Title:** Paradigm 1: Sublinear TF-IDF + Balanced Logistic Regression
- **Speaker:** Md. Tariful Islam Jony (ID: 2107119)
- **Architecture & Highlights:**
  - Feature Space: Word N-Grams $(1, 2)$, Sublinear TF scaling ($1 + \log(TF)$), max 20,000 features.
  - Loss Formulation: Multinomial Logistic Regression with inverse class weighting to penalize minority class errors.
  - **Strengths:** Highly efficient inference, highly sensitive to explicit sarcastic marker words (*“বাহ”*, *“অসাধারণ”*).
  - **Limitations:** Bag-of-words assumption ignores long-range sequential negation and word order.

---

## Slide 7: Modeling Paradigm 2 — Deep Sequential (Word2Vec + Stacked BiLSTM)
- **Slide Title:** Paradigm 2: 128-d Word2Vec + PyTorch 2-Layer Stacked BiLSTM
- **Speaker:** Siyam Khan (ID: 2107120)
- **Architecture & Highlights:**
  - **Custom Dense Embeddings:** 128-dimensional dense vector space trained on 159,117 sentences.
  - **Network Architecture:**
    - Embedding Layer (128-d, fine-tunable during backpropagation)
    - 2-layer Stacked Bidirectional LSTM (Hidden size = 128 per direction, total 256)
    - Dropout ($p=0.30$) for regularization against overfitting
    - Fully Connected classification head with Softmax
  - **Key Result:** **Champion model for Hate Speech Detection**, achieving **88.52% Test Accuracy** and **88.50% Macro F1**.
  - **Why it wins:** Hate speech in Bengali relies heavily on sequential syntactic patterns and aggressive verb compounds captured effectively by bidirectional gates.

---

## Slide 8: Modeling Paradigm 3 — Transformer SOTA (Fine-Tuned BanglaBERT)
- **Slide Title:** Paradigm 3: Fine-Tuned `sagorsarker/bangla-bert-base`
- **Speaker:** Md. Tariful Islam Jony (ID: 2107119)
- **Architecture & Highlights:**
  - 110M parameter pretrained bidirectional transformer encoder.
  - WordPiece subword tokenization (vocabulary: 102,025 tokens), max sequence length $L=64$.
  - Sequence Classification heads fine-tuned with AdamW ($\text{lr} = 2 \times 10^{-5}$), linear warmup scheduler, and balanced cross-entropy.
  - **Key Result:** **Highest Sarcastic Class Recall (78.05%)** across all paradigms. Captures subtle contrast between praise prefix and disappointing suffix.

---

## Slide 9: Master Comparative Benchmark Matrix
- **Slide Title:** Comprehensive Experimental Results Across All Tasks
- **Master Comparison Table:**
  | Task | Model Paradigm | Test Accuracy | Macro F1 | Weighted F1 |
  | :--- | :--- | :---: | :---: | :---: |
  | **Sentiment** | **TF-IDF + Logistic Regression** | **77.60%** | **53.71%** | **82.24%** |
  | Sentiment | Word2Vec + Stacked BiLSTM | 76.60% | 53.27% | 81.96% |
  | Sentiment | Fine-Tuned BanglaBERT | 62.46% | 40.75% | 70.95% |
  | **Sarcasm** | **TF-IDF + Logistic Regression** | **74.94%** | **72.89%** | **75.40%** |
  | Sarcasm | Word2Vec + Stacked BiLSTM | 74.36% | 72.00% | 74.74% |
  | Sarcasm | Fine-Tuned BanglaBERT | 73.86% | 72.52% | 74.57% *(Recall: 78.1%)* |
  | **Hate Speech**| TF-IDF + Logistic Regression | 86.61% | 86.52% | 86.57% |
  | **Hate Speech**| **Word2Vec + Stacked BiLSTM** | **88.52%** | **88.50%** | **88.52%** |
  | Hate Speech| Fine-Tuned BanglaBERT | 76.71% | 76.68% | 76.71% |

---

## Slide 10: Crucial Academic Insights & Trade-Off Analysis
- **Slide Title:** Critical Evaluation & "No Free Lunch" Findings
- **Key Points:**
  1. **No Single Architecture Wins Everything:**
     - BiLSTM is optimal for syntactically bounded aggressive expressions (Hate Speech).
     - TF-IDF + LR remains extremely robust for high-skew keyword vocabularies.
     - BanglaBERT is essential for semantic contrast resolution in sarcastic rhetoric.
  2. **Architectural Complexity Trade-off:**
     - Linear models provide high interpretability and light computational footprint.
     - Recurrent networks capture sequential grammar and aggressive compounds.
     - Transformers model contextual nuances and semantic inversion.
  3. **Context-Aware Fusion:**
     - Combining the strengths of each paradigm into an intelligent router achieves the highest real-world semantic fidelity.

---

## Slide 11: Unified Context-Aware Multi-Task Pipeline
- **Slide Title:** Context-Aware Ensemble & Contrast Inversion
- **Pipeline Workflow:**
  ```
  Raw Text Input
        │
        ▼
  Negation-Preserving Normalizer (Preserves 'না', 'নয়', etc.)
        │
  ┌─────┴─────────────────────────┐
  │                               │
  ▼                               ▼
Sarcasm Cue Detection         Sequential Hate Speech
(TF-IDF + BanglaBERT)         (BiLSTM Classifier - 88.5% Acc)
  │                               │
  ▼                               ▼
Is Sarcasm Present?           Clean vs Hate Speech Label
  │
  ├─► YES: Invert surface positive words into NEGATIVE Sentiment
  │
  └─► NO:  Calibrate Sentiment via consensus voting
  ```
- **Result on Proposal Example:** Correctly outputs **Sentiment: Negative**, **Sarcasm: Yes**, **Hate Speech: No**.

---

## Slide 12: Live System Architecture & Streamlit Dashboard
- **Slide Title:** Standalone Interactive GUI Demonstration
- **Speaker:** Siyam Khan (ID: 2107120)
- **Features of the Web Application (`app.py`):**
  - **Modern UI:** Custom dark-mode glassmorphism cards, responsive typography, live progress bars.
  - **Model Selector:** Live toggle between Ensemble, TF-IDF+LR, BiLSTM, and BanglaBERT.
  - **Preset Benchmark Buttons:** Single-click testing of proposal edge cases.
  - **Side-by-Side Comparison:** Instant comparative matrix showing predictions from all three models simultaneously.
  - **Linguistic Inspector:** Tokenization drawer highlighting preserved negation words.
  - **EDA & Diagnostic Tabs:** Integrated confusion matrices and dataset distributions.

---

## Slide 13: Live Demonstration
- **Slide Title:** Live Demonstration & Edge Case Testing
- **Actions During Demo:**
  1. Launch Streamlit app (`streamlit run "Project files/app.py"`).
  2. Click **Proposal Sarcasm** preset (*"বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!"*).
     - Show: Sentiment: Negative (Context Inverted), Sarcasm: Sarcastic (Yes), Hate Speech: Clean (No).
  3. Click **Positive Review** preset (*"বইটা অসম্ভব সুন্দর এবং অনুপ্রেরণামূলক!"*).
     - Show: Sentiment: Positive, Sarcasm: No, Hate: No.
  4. Click **Hate Speech** preset.
     - Show: Hate Speech: Yes (Red Badge), Sentiment: Negative.
  5. Show **Confusion Matrices Tab** and **Side-by-Side Comparison** table.

---

## Slide 14: Division of Responsibilities & Team Contributions
- **Slide Title:** Team Collaboration & Work Distribution
- **Table of Contributions:**
  | Member | Student ID | Core Responsibilities & Contributions |
  | :--- | :---: | :--- |
  | **Md. Tariful Islam Jony** | **2107119** | **Phase 1:** Corpus cleaning, label encoding & 80/10/10 split standardization.<br>**Phase 3:** TF-IDF feature extraction & balanced Logistic Regression models.<br>**Phase 5:** Hugging Face BanglaBERT fine-tuning & evaluation metrics. |
  | **Siyam Khan** | **2107120** | **Phase 2:** Negation-preserving tokenization, stop-words & EDA visual analysis.<br>**Phase 4:** 128-d Word2Vec embeddings & PyTorch Stacked BiLSTM modeling.<br>**Phase 6:** Context-aware inference engine, Streamlit GUI & Master Notebook. |

---

## Slide 15: Conclusion & Future Work
- **Slide Title:** Summary of Achievements & Future Horizons
- **Summary:**
  - Built an end-to-end context-aware text analyzer for Bengali social media text without external LLMs.
  - Benchmarked 3 model families across 3 diverse tasks on 374,000+ total sentences.
  - Solved the sarcastic contrast dilemma highlighted in the project proposal.
  - Delivered a production-grade, offline Streamlit web dashboard and consolidated Jupyter notebooks.
- **Future Enhancements:**
  - Multi-task joint loss training (shared encoder backbone).
  - Code-mixed Banglish phonetic mapping (converting Romanized Bangla to Unicode).
  - Quantization to INT8 (reducing model size from 657 MB to < 170 MB for mobile deployment).

---

## Slide 16: Acknowledgements & Q&A
- **Slide Title:** Questions & Discussion
- **Content:**
  - We express our sincere gratitude to the Department of Computer Science & Engineering, KUET, for continuous support throughout CSE 4121.
  - **Open for Questions & Discussion from the Audience.**

---

# 🧠 Prepared Q&A Defense Sheet (For Jony & Siyam)

### Q1: Why did BiLSTM outperform BanglaBERT on Hate Speech detection?
**Answer:** The Hate Speech dataset exhibits strong sequential and syntactic patterns (e.g., abusive idioms, imperative threatening verbs). The 2-layer BiLSTM with custom domain-trained Word2Vec embeddings effectively modeled these sequential structures without being diluted by the broad generic masked language modeling pretraining of BanglaBERT. Furthermore, with 50,000 sentences, BiLSTM avoided the subtle overfitting risk of fine-tuning 110M parameters.

### Q2: Why did TF-IDF + Logistic Regression achieve the highest accuracy in Sentiment Analysis?
**Answer:** The Sentiment dataset has a severe 90% positive class imbalance. TF-IDF paired with class-balanced sample weighting explicitly penalizes errors on the minority neutral and negative classes, maintaining high precision on key sentiment words (*“অসাধারণ”*, *“বাজে”*). Deep transformers, when trained without extensive hyperparameter search on heavily skewed datasets, tend to suffer from majority-class bias.

### Q3: How exactly does the Context-Aware Ensemble resolve sarcasm contrast?
**Answer:** In sarcastic contrast sentences, there is a conflict between positive surface descriptors (e.g., *“অসাধারণ”*) and negative event outcomes (e.g., *“কাজ হলো না”*). Our engine detects sarcastic markers via TF-IDF/BERT and scans for preserved negation tokens. When a sentence is flagged as sarcastic and contains negation tokens, the engine overrides the naive positive surface classification and maps the true sentiment to **Negative**, aligning with human pragmatic intent.

### Q4: Why was negation preservation critical during preprocessing?
**Answer:** Standard NLTK or generic stop-word lists treat *“না”* (no/not), *“নয়”* (is not), and *“নেই”* (absent) as frequent stop-words and remove them. If you strip *“না”* from *“কাজ হলো না”*, the sentence becomes *“কাজ হলো”* (work was done), completely reversing the true meaning. Preserving negation is essential for all sentiment and sarcasm modeling.
