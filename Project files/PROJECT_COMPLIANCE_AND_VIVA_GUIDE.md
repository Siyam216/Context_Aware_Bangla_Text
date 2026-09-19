# 🇧🇩 Context-Aware Bengali Text Multi-Task Analyzer
## Project Compliance, Theoretical Mechanisms & Comprehensive Showcase/Viva Defense Guide

- **Course:** CSE 4122 (Natural Language Processing Sessional)
- **Academic Term:** 4th Year, 1st Term (CSE 4-1)
- **Department:** Department of Computer Science & Engineering, KUET
- **Authors:**
  - **Md. Tariful Islam Jony** (Roll: 2107119)
  - **Siyam Khan** (Roll: 2107120)

---

# 📑 Table of Contents
1. [Executive Compliance Matrix (Every Instruction Checked & Verified)](#1-executive-compliance-matrix)
2. [Detailed Requirement Verification (Where & How in Code)](#2-detailed-requirement-verification)
3. [Core Technical Mechanisms (With Math, Diagrams & Numerical Examples)](#3-core-technical-mechanisms)
   - [Mechanism 1: Word Embeddings (Lookup, Vector Math & Cosine Similarity)](#mechanism-1-word-embeddings)
   - [Mechanism 2: Generative vs. Discriminative Classifiers (MNB vs. Logistic Regression)](#mechanism-2-generative-vs-discriminative-classifiers)
   - [Mechanism 3: Sublinear TF-IDF Vectorization](#mechanism-3-sublinear-tf-idf-vectorization)
   - [Mechanism 4: Stacked Bidirectional LSTM (BiLSTM)](#mechanism-4-stacked-bidirectional-lstm-bilstm)
   - [Mechanism 5: Transformer Self-Attention & BanglaBERT Fine-Tuning](#mechanism-5-transformer-self-attention--banglabert)
   - [Mechanism 6: Dynamic Context-Aware Semantic Inversion Engine](#mechanism-6-dynamic-context-aware-semantic-inversion-engine)
4. [Top 20 Probable Viva / Showcase Questions & Bulletproof Model Answers](#4-top-20-probable-viva--showcase-questions)

---

# 1. Executive Compliance Matrix

Here is how every single instruction from Shawon Sir and the showcase guidelines has been **100% fulfilled** in our project:

| # | Teacher's / Showcase Instruction | Compliance Status | Where in Codebase? | Concrete Implementation Details |
| :--- | :--- | :---: | :--- | :--- |
| **1** | **Pretrained vs. Custom Embeddings**<br>*(Either use pretrained or train from corpus. If trained, explain embedding process. Balanced comparison is best.)* | **100% Fulfilled**<br>*(Both Implemented)* | • `src/train_bilstm.py` (Lines 8, 48–75)<br>• `saved_models/bangla_word2vec.pt`<br>• `src/predict_bert.py` (Lines 30–65) | • **Custom Embedding:** Trained 128-dimensional dense continuous Word2Vec embeddings from our Bengali corpus.<br>• **Pretrained Embedding:** Used subword WordPiece embeddings from `sagorsarker/bangla-bert-base` (110M params).<br>• Fully compared in Master Benchmark! |
| **2** | **Corpus Size**<br>*(Moderate/reasonably sized corpus to justify final decisions, not just a toy corpus.)* | **100% Fulfilled**<br>*(218k+ Sentences)* | • `cleaned_data/`<br>• `eda_summary.json`<br>• `eda_plots/class_distributions.png` | Total **218,371 annotated Bengali sentences**:<br>• **Sentiment:** 156,010 samples<br>• **Sarcasm:** 12,089 samples<br>• **Hate Speech:** 50,272 samples<br>Divided into strict **70:10:20** (Train/Val/Test) splits. |
| **3** | **Labeled Data & Supervised vs. Unsupervised Scope**<br>*(Primarily supervised, but unsupervised exploration adds value.)* | **100% Fulfilled** | • `src/train_bilstm.py`<br>• `src/train_tfidf_lr.py`<br>• `notebooks/NLP_Project.ipynb` | • **Supervised:** Multi-task 3-way polarity, sarcasm, and hate classification.<br>• **Unsupervised Exploration:** Word2Vec continuous embedding learning (learning semantics from raw co-occurrences without human labels) + Unsupervised N-gram frequency distribution analysis. |
| **4** | **Interface Requirement**<br>*(No terminal code; clear GUI/input-output system where user gives input and observes output results.)* | **100% Fulfilled**<br>*(Interactive Web App)* | • `app.py`<br>• `run_app.bat`<br>• URL: `http://localhost:8501` | Full **Streamlit Interactive Web Application**:<br>• Text area for typing/pasting raw Bengali sentences.<br>• 1-Click "Analyze Context" button.<br>• Live Multi-Task Output Cards, probability confidence meters, benchmark viewer, and confusion matrix visualizers. |
| **5** | **Generative vs. Discriminative Classifiers**<br>*(Compare at least one generative and one discriminative approach.)* | **100% Fulfilled** | • `src/train_tfidf_lr.py`<br>• `master_benchmark.md`<br>• Section IV of Academic Paper | • **Generative Model:** Multinomial Naive Bayes (MNB) modeling Joint Probability $P(X, Y) = P(Y)P(X\|Y)$.<br>• **Discriminative Models:** Logistic Regression, Linear SVM, BiLSTM, and BanglaBERT modeling Conditional Probability $P(Y\|X)$ directly. |
| **6** | **Data Collection & Preprocessing Documentation**<br>*(Describe sources, collection methods, and preprocessing steps.)* | **100% Fulfilled** | • `Project files/report/main.tex`<br>• `presentation_slides_outline.md`<br>• `src/data_preprocessing.py` | • Comprehensive documentation of social media, reviews, and comment corpus sources.<br>• **Negation Preservation Algorithm:** Protects crucial negation tokens (`না`, `নয়`, `নেই`, `নাহ`) from destructive stopword stripping! |
| **7** | **Strict Ban on RAG / LLM Wrappers**<br>*(No RAG, no prompt wrappers; everything built from scratch.)* | **100% Fulfilled**<br>*(Pure Scratch NLP)* | • `src/`<br>• `saved_models/`<br>• Entire project root | • **Zero RAG, Zero LangChain, Zero OpenAI API, Zero Prompts.**<br>• 100% built on PyTorch tensors, Scikit-Learn pipelines, and local HuggingFace Transformer weights running offline on local CPU/GPU. |
| **8** | **Pre-trained Models Setup in Advance**<br>*(All weights pre-trained and saved before showcase.)* | **100% Fulfilled**<br>*(Instant Offline Inference)* | • `saved_models/banglabert_sentiment/`<br>• `saved_models/banglabert_sarcasm/`<br>• `saved_models/banglabert_hate/`<br>• `saved_models/bilstm_*.pt`<br>• `saved_models/tfidf_*.joblib` | All models were pre-trained (including GPU fine-tuning on Google Colab T4) and saved locally into `saved_models/`. The app launches in 2 seconds and evaluates in real time with **zero network dependency**! |

---

# 2. Detailed Requirement Verification

### 2.1 Pretrained vs. Custom Embeddings
* **Where in code:**
  - `Project files/src/train_bilstm.py`: Implements `StackedBiLSTMClassifier` which takes `bangla_word2vec.pt`.
  - `Project files/src/predict_bert.py`: Implements `BertClassifierWrapper` loading `AutoTokenizer` and `AutoModelForSequenceClassification` from `saved_models/banglabert_{task}`.
* **Explanation:**
  We implemented **both** embedding strategies:
  1. **Custom Bengali Word2Vec (128-d):** We trained continuous vector representations directly on our cleaned Bengali corpus. Each word in our vocabulary $V$ is mapped to a dense real-valued vector $\mathbf{v} \in \mathbb{R}^{128}$.
  2. **Pretrained Subword Transformer Embeddings:** We utilized `sagorsarker/bangla-bert-base` (pre-trained on 27.5 GB of Bengali text). It uses WordPiece subword tokenization with 768-dimensional contextual hidden states, dynamically adjusting word vectors based on surrounding syntax.

### 2.2 Dataset Dimensions & Splitting Protocol
* **Where in code:** `Project files/cleaned_data/`, `Project files/eda_summary.json`.
* **Breakdown:**
  - **Sentiment Analysis:** 156,010 sentences (109,229 Train, 15,569 Val, 31,212 Test).
  - **Sarcasm Detection:** 12,089 sentences (8,462 Train, 1,209 Val, 2,418 Test).
  - **Hate Speech Detection:** 50,272 sentences (35,190 Train, 5,027 Val, 10,055 Test).
  - **Total:** 218,371 sentences!
* **Splits:** We strictly maintained 70% Train, 10% Validation, and 20% Test. The vectorizers and tokenizers were fitted **strictly on the training split**, preventing any data leakage into validation or test sets.

### 2.3 Non-Terminal User Interface (GUI)
* **Where in code:** `Project files/app.py`, launched via `run_app.bat`.
* **Execution:**
  Users do not touch the command line. They open the web dashboard in their browser at `http://localhost:8501`, enter any Bengali sentence, and immediately view:
  1. Context-aware multi-task classification cards (Sentiment, Sarcasm, Hate Speech).
  2. Softmax probability confidence bars.
  3. Sarcasm inversion explanation (explaining why a sentence was re-routed).
  4. Tabs for Master Benchmark Matrix, Dataset EDA Distributions, and Confusion Matrices.

---

# 3. Core Technical Mechanisms

Sir will ask **how things work under the hood**. Here is the complete mathematical and conceptual explanation with concrete numerical examples.

---

## Mechanism 1: Word Embeddings
*(How words become numbers, lookup tables, and cosine similarity)*

### 1. The Core Concept
Computers cannot multiply Bengali strings. A word embedding is a parameterized function mapping words from a discrete vocabulary $\mathcal{V}$ to continuous vectors in $\mathbb{R}^d$:
$$\mathbf{e}_w = \mathbf{E}(w), \quad \mathbf{e}_w \in \mathbb{R}^d$$
In PyTorch, this is implemented as an `nn.Embedding(num_embeddings, embedding_dim)` layer, which is fundamentally a weight matrix $\mathbf{W}_{\text{embed}} \in \mathbb{R}^{|\mathcal{V}| \times d}$.

```
One-Hot Vector x_w          Embedding Matrix W (|V| x d)            Dense Vector e_w
[ 0,  1,  0,  0 ]   ×   [ [ 0.12, -0.45,  0.88 ],       =     [ -0.34, 0.72, 0.15 ]
                              [ -0.34,  0.72,  0.15 ],   (row 1)
                              [  0.91,  0.05, -0.22 ],
                              [ -0.18, -0.61,  0.40 ] ]
```

### 2. Concrete Numerical Example

Suppose our toy vocabulary has 4 Bengali words ($|\mathcal{V}| = 4$) with $d = 3$ dimensions:
- Index 0: **আমি** (I)
- Index 1: **বাংলা** (Bengali)
- Index 2: **ভালোবাসি** (Love)
- Index 3: **নিন্দা** (Hate/Scorn)

Our embedding matrix $\mathbf{W}_{\text{embed}}$ is:
$$\mathbf{W}_{\text{embed}} = \begin{bmatrix}
0.20 & 0.80 & 0.10 \\
0.15 & 0.75 & 0.20 \\
0.90 & 0.10 & 0.40 \\
-0.85 & 0.05 & -0.30
\end{bmatrix}$$

**Query:** How similar are the embeddings of **ভালোবাসি** ($\mathbf{e}_2$) and **নিন্দা** ($\mathbf{e}_3$)?

We calculate the **Cosine Similarity**:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

1. Vector $\mathbf{u} = \mathbf{e}_2 = [0.90, 0.10, 0.40]$
   - Length $\|\mathbf{u}\| = \sqrt{0.90^2 + 0.10^2 + 0.40^2} = \sqrt{0.81 + 0.01 + 0.16} = \sqrt{0.98} \approx 0.9899$
2. Vector $\mathbf{v} = \mathbf{e}_3 = [-0.85, 0.05, -0.30]$
   - Length $\|\mathbf{v}\| = \sqrt{(-0.85)^2 + 0.05^2 + (-0.30)^2} = \sqrt{0.7225 + 0.0025 + 0.09} = \sqrt{0.815} \approx 0.9028$
3. Dot Product $\mathbf{u} \cdot \mathbf{v}$:
   $$\mathbf{u} \cdot \mathbf{v} = (0.90 \times -0.85) + (0.10 \times 0.05) + (0.40 \times -0.30)$$
   $$\mathbf{u} \cdot \mathbf{v} = -0.765 + 0.005 - 0.120 = -0.880$$
4. Cosine Similarity:
   $$\text{Sim}(\mathbf{u}, \mathbf{v}) = \frac{-0.880}{0.9899 \times 0.9028} = \frac{-0.880}{0.8937} \approx \mathbf{-0.9847}$$

**Interpretation:** The cosine similarity is near $-1.0$, demonstrating that the vector space correctly understands that **ভালোবাসি** (Love) and **নিন্দা** (Hate) point in almost opposite semantic directions!

---

## Mechanism 2: Generative vs. Discriminative Classifiers
*(Multinomial Naive Bayes vs. Logistic Regression)*

Shawon Sir explicitly asked for a comparison between Generative and Discriminative paradigms.

### 1. The Theoretical Contrast
* **Generative Models (e.g., Naive Bayes):**
  - Learn the **joint probability distribution** $P(X, Y) = P(Y) \cdot P(X|Y)$.
  - They model how the data was generated for each class, asking: *"Given that this sentence is Sarcastic, how likely is it to generate these words?"*
  - Use Bayes' Theorem to invert the conditional:
    $$P(Y=c|X) = \frac{P(Y=c) \prod_{i=1}^n P(w_i|Y=c)}{P(X)}$$
* **Discriminative Models (e.g., Logistic Regression, SVM, BiLSTM, BERT):**
  - Learn the **conditional probability distribution** $P(Y|X)$ directly.
  - They do not care how the words were generated; they only care about finding the **decision boundary** that separates classes:
    $$P(Y=1|X) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

```
    GENERATIVE (Naive Bayes)                  DISCRIMINATIVE (Logistic Regression)
Models the distribution of each class:       Models the decision boundary directly:
         Class 0          Class 1                       Class 0   |   Class 1
          *  *              #  #                         *   *    |    #   #
        *  *  *            #  #  #                      *  *  *   |   #  #  #
          *  *              #  #                         *   *    |    #   #
    P(X|Y=0)P(Y=0)     P(X|Y=1)P(Y=1)                        w^T x + b = 0
```

### 2. Concrete Numerical Walkthrough

Let us classify the sentence: **"ভালো না"** (Not good) into **Positive ($+$)** or **Negative ($-$)**.

#### A. Generative Approach (Multinomial Naive Bayes):
Suppose our training corpus has:
- Total Sentences: $10$ ($6$ Positive, $4$ Negative) $\implies P(+) = 0.6, P(-) = 0.4$.
- Vocabulary: $V = \{\text{ভালো}, \text{না}, \text{খারাপ}\} \implies |V| = 3$.
- Word counts in Positive sentences:
  - $\text{count}(\text{"ভালো"}, +) = 8$
  - $\text{count}(\text{"না"}, +) = 1$
  - Total words in Positive class: $8 + 1 + 1 = 10$.
- Word counts in Negative sentences:
  - $\text{count}(\text{"ভালো"}, -) = 2$
  - $\text{count}(\text{"না"}, -) = 7$
  - Total words in Negative class: $2 + 7 + 1 = 10$.

Using **Laplace Smoothing** ($+1$):
$$P(w|c) = \frac{\text{count}(w, c) + 1}{\sum_{w'} \text{count}(w', c) + |V|}$$

1. **For Positive Class ($+$):**
   - $P(\text{"ভালো"}|+) = \frac{8 + 1}{10 + 3} = \frac{9}{13} \approx 0.6923$
   - $P(\text{"না"}|+) = \frac{1 + 1}{10 + 3} = \frac{2}{13} \approx 0.1538$
   - Joint Probability Score:
     $$S(+) = P(+) \times P(\text{"ভালো"}|+) \times P(\text{"না"}|+) = 0.6 \times 0.6923 \times 0.1538 = \mathbf{0.0639}$$

2. **For Negative Class ($-$):**
   - $P(\text{"ভালো"}|-) = \frac{2 + 1}{10 + 3} = \frac{3}{13} \approx 0.2308$
   - $P(\text{"না"}|-) = \frac{7 + 1}{10 + 3} = \frac{8}{13} \approx 0.6154$
   - Joint Probability Score:
     $$S(-) = P(-) \times P(\text{"ভালো"}|-) \times P(\text{"না"}|-) = 0.4 \times 0.2308 \times 0.6154 = \mathbf{0.0568}$$

Normalized Posterior:
$$P(+|\text{"ভালো না"}) = \frac{0.0639}{0.0639 + 0.0568} = \mathbf{52.9\%}$$
$$P(-|\text{"ভালো না"}) = \frac{0.0568}{0.0639 + 0.0568} = \mathbf{47.1\%}$$

**Failure of Naive Bayes:** Because Naive Bayes assumes words are conditionally independent ($P(\text{ভালো, না}) = P(\text{ভালো}) \times P(\text{না})$), the overwhelming frequency of "ভালো" in positive training data caused Naive Bayes to predict **Positive (52.9%)**, failing to understand that "না" inverted "ভালো"!

#### B. Discriminative Approach (Logistic Regression with Bigrams):
Logistic regression extracts the bigram feature: `"ভালো_না"`.
- Weight vector: $w_{\text{"ভালো"}} = +0.8$, $w_{\text{"না"}} = -0.5$, $w_{\text{"ভালো\_না"}} = \mathbf{-2.4}$, Bias $b = -0.1$.
- Linear Logit $z$:
  $$z = w_{\text{"ভালো"}} + w_{\text{"না"}} + w_{\text{"ভালো\_না"}} + b = 0.8 - 0.5 - 2.4 - 0.1 = \mathbf{-2.2}$$
- Probability via Sigmoid $\sigma(z)$:
  $$P(Y=+ | X) = \frac{1}{1 + e^{-(-2.2)}} = \frac{1}{1 + e^{2.2}} = \frac{1}{1 + 9.025} = \mathbf{0.0997 \approx 10.0\%}$$
  $$P(Y=- | X) = 1 - 0.100 = \mathbf{90.0\%} \implies \text{\textbf{Negative!}}$$

**Why Discriminative is Superior:** Logistic Regression directly optimizes class separation and easily learns negative weights for composite phrases and bigrams, successfully capturing syntactic inversion!

---

## Mechanism 3: Sublinear TF-IDF Vectorization
*(Why term frequency scaling matters in social media)*

Standard Term Frequency (TF) counts raw occurrences. But if a toxic commenter writes *"খারাপ"* 20 times, is the comment 20 times more toxic than one writing it once? **No.**

We applied **Sublinear TF Scaling**:
$$\text{sublinear\_tf}(t, d) = 1 + \log(tf_{t, d}) \quad \text{for } tf_{t, d} > 0$$

### Numerical Comparison:
- If $tf = 1 \implies 1 + \log(1) = 1.0$
- If $tf = 2 \implies 1 + \log(2) = 1 + 0.693 = 1.693$
- If $tf = 20 \implies 1 + \log(20) = 1 + 2.996 = \mathbf{3.996}$ (instead of $20.0$!)

Combined with Inverse Document Frequency:
$$w_{t, d} = (1 + \log(tf_{t, d})) \times \left( \log\left(\frac{1 + N}{1 + df_t}\right) + 1 \right)$$
This prevents repetitive social media rants from exploding vector norms and biasing linear classifiers.

---

## Mechanism 4: Stacked Bidirectional LSTM (BiLSTM)
*(Capturing past and future context)*

In Bengali sentences, the verb or negation token often appears at the very end of the sentence (Subject-Object-Verb / SOV structure). A standard unidirectional LSTM reads left-to-right, meaning early words have no idea if a negation exists at the end.

### 1. BiLSTM Architecture

```
Forward Pass:      (w1) ---> [LSTM_Fwd] ---> [LSTM_Fwd] ---> [LSTM_Fwd] ---> h_T_fwd
                               ^                 ^                 ^
Input Tokens:               "অসাধারণ"           "service"        "হলো না"
                               v                 v                 v
Backward Pass:     h_1_bwd <--- [LSTM_Bwd] <--- [LSTM_Bwd] <--- [LSTM_Bwd] <--- (wT)

Concatenation:     h_final = [ h_T_fwd  ||  h_1_bwd ]  ∈ R^(2 * hidden_dim) = R^(128)
```

### 2. The Internal LSTM Cell Gates (Math to Explain to Sir)
For each time step $t$, the LSTM cell regulates information flow through 4 gates:
1. **Forget Gate:** Decides what past information to throw away:
   $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
2. **Input Gate:** Decides which new values to update:
   $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
3. **Candidate Memory Cell:** Generates candidate values:
   $$\tilde{C}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)$$
4. **New Cell State:** Combines old memory and new candidate:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
5. **Output Gate & Hidden State:**
   $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o), \quad h_t = o_t \odot \tanh(C_t)$$

By stacking **2 layers** with **Spatial Dropout ($p = 0.3$)**, the model prevents feature co-adaptation and captures complex syntactic hierarchies.

---

## Mechanism 5: Transformer Self-Attention & BanglaBERT
*(State-of-the-Art Contextual Representations)*

`sagorsarker/bangla-bert-base` is an ELECTRA-discriminator based model with:
- 12 Transformer Encoder Layers
- 12 Attention Heads per layer
- 768 Hidden Dimension ($d_{\text{model}} = 768$)
- 110 Million Parameters

### 1. The Scaled Dot-Product Attention Equation
$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

- **$Q$ (Query):** *"What am I looking for?"*
- **$K$ (Key):** *"What content do I have?"*
- **$V$ (Value):** *"What representation should I pass forward?"*

### 2. Why Divide by $\sqrt{d_k}$? (Sir love asking this!)
For $d_k = 64$, $\sqrt{d_k} = 8$. If we don't divide by $\sqrt{d_k}$, the dot product $Q K^T$ grows large in magnitude:
$$\text{Var}(Q \cdot K) = d_k$$
Large dot products push the Softmax function into regions with **extremely small gradients** (the vanishing gradient problem). Dividing by $\sqrt{d_k}$ stabilizes the variance to $1.0$, keeping gradients healthy during backpropagation!

---

## Mechanism 6: Dynamic Context-Aware Semantic Inversion Engine
*(Resolving Sarcastic Contradictions)*

### 1. The Real-World Linguistic Problem
Input: **“বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!”**
- Traditional Sentiment Model reads: *“অসাধারণ”*, *“বাহ”* $\implies$ Predicts **Positive**!
- Sarcasm Model reads the contradiction between praise (*“অসাধারণ”*) and failure (*“কাজ হলো না”*) $\implies$ Predicts **Sarcastic (Yes)**!
- Hate Speech Model evaluates toxic intent $\implies$ Predicts **Non-Hate (No)**.

### 2. The Decision Logic (Inference Rule)

```
                           Input Bengali Text
                                   |
                +------------------+------------------+
                |                                     |
         Sarcasm Model                          Sentiment Model
                |                                     |
      P(Sarcastic) >= 0.65?                          Raw Polarity
                |                                     |
        +-------+-------+                             |
        | YES           | NO                          |
        v               v                             v
Is Sentiment Pos?     Keep Sentiment <-------------+  |
        |                                             |
   +----+----+                                        |
   | YES     | NO                                     |
   v         +----------------------------------------+
INVERT POLARITY:
Change Sentiment -> NEGATIVE
Add Explanation: "Sarcastic Inversion Detected"
```

In `src/inference_pipeline.py` and `app.py`, this logic reconciles conflicting predictions into a single, contextually coherent output.

---

# 4. Top 20 Probable Viva / Showcase Questions

### General & Setup Questions

#### Q1: "Why didn't you use ChatGPT, GPT-4, or LangChain/RAG for this project?"
> **Answer:** "Sir, Shawon Sir explicitly instructed us not to use RAG or commercial LLM prompt wrappers because feeding a dataset to an external API obscures foundational NLP principles. We wanted to build everything from first principles—covering text preprocessing, vector space modeling, recurrent neural networks (BiLSTM), and fine-tuning open-source pre-trained transformers (`BanglaBERT`) on local GPU hardware."

#### Q2: "What is the difference between Word2Vec, GloVe, and BERT embeddings?"
> **Answer:**
> - **Word2Vec (Mikolov et al.):** Predicts context words using neural skip-grams or CBOW over local sliding windows. Static: each word gets exactly one fixed vector regardless of context.
> - **GloVe (Pennington et al.):** Uses global matrix factorization of the whole corpus co-occurrence matrix. Still static.
> - **BERT/BanglaBERT (Devlin et al.):** Contextual subword embeddings using self-attention. The word *“কাজ”* gets a completely different 768-d vector depending on whether it appears in *“কাজের মানুষ”* (hardworking) or *“কাজ হলো না”* (failure).

#### Q3: "Is your app running in the terminal or on a GUI?"
> **Answer:** "Sir, we have built a complete, interactive Web GUI using Streamlit (`app.py`), running locally at `localhost:8501`. Anyone can type any Bengali sentence and observe instant multi-task classifications, confidence gauges, and explanations without touching any terminal code."

---

### Machine Learning & Preprocessing Questions

#### Q4: "Why did you create a custom Negation Preservation Algorithm?"
> **Answer:** "In standard NLP, stopwords (frequent words) are removed. However, in Bengali, words like *“না”* (not), *“নয়”* (is not), and *“নেই”* (absent) are grammatical stopwords. If a standard stopword filter strips *“না”* from *“ভালো না”*, the sentence becomes *“ভালো”*—completely inverting the sentiment from Negative to Positive! Our custom preprocessor explicitly protects a whitelist of 12 Bengali negation operators while discarding generic noise."

#### Q5: "Why did Logistic Regression with TF-IDF perform slightly better on Sentiment Accuracy (77.60%) than BanglaBERT (76.39%)?"
> **Answer:** "The Sentiment dataset has an extreme 90% positive class imbalance. Logistic Regression was trained with inverse class weighting (`class_weight='balanced'`), which explicitly penalizes minority-class errors. BanglaBERT, when fine-tuned over 109k samples under standard cross-entropy without heavy focal loss, slightly optimizes for majority-class probability calibration. Furthermore, lexical sentiment triggers (*“অসাধারণ”*, *“বাজে”*) are easily separable linearly."

#### Q6: "Why is Macro F1 much lower (~51.5% - 53.7%) than Accuracy (~76.4% - 77.6%) on Sentiment Analysis?"
> **Answer:** "In an imbalanced dataset where 90% of samples are Positive, a naive dummy model predicting 'Positive' for everything would get 90% accuracy but 0% recall on Negative and Neutral, giving a terrible Macro F1. Macro F1 computes the unweighted average of F1 across all classes:
> $$\text{Macro F1} = \frac{F1_{\text{Neg}} + F1_{\text{Neu}} + F1_{\text{Pos}}}{3}$$
> It treats minority classes with equal importance, preventing majority-class bias. Our high Weighted F1 (>81%) confirms high overall fidelity, but Macro F1 provides an honest, rigorous metric."

#### Q7: "Why did you use Sublinear TF scaling instead of raw Term Frequency?"
> **Answer:** "In social media, people often repeat words 10 or 20 times (e.g., *'অনেক অনেক অনেক ভালো'*). If we use raw count, the vector norm is dominated by that one word. Sublinear TF uses $1 + \log(TF)$, mapping 20 occurrences to ~3.99, preventing single-token domination."

---

### Deep Learning & Architecture Questions

#### Q8: "How does your BiLSTM process a sentence, and why is Bidirectional needed?"
> **Answer:** "Bengali is an SOV (Subject-Object-Verb) language where negation often appears at the end. In a forward-only LSTM, when processing the first word *“অসাধারণ”*, the hidden state has no knowledge of the upcoming *“হলো না”*. The backward LSTM reads from right to left, passing future context backward. Concatenating both directions $[\overrightarrow{h}_T ; \overleftarrow{h}_1]$ provides a full 256-dimensional contextual summary."

#### Q9: "What is Spatial Dropout in PyTorch, and why use it over standard Dropout?"
> **Answer:** "Standard dropout drops random individual neurons across embedding dimensions. Spatial Dropout (or `Dropout1d`) drops entire word channels across the sequence. In NLP, this prevents adjacent words from co-depending on specific embedding dimensions, forcing the BiLSTM to learn robust linguistic patterns."

#### Q10: "What optimizer and learning rate did you use for fine-tuning BanglaBERT?"
> **Answer:** "We used **AdamW** (Adam with decoupled weight decay) with a learning rate of $\eta = 2 \times 10^{-5}$, linear warmup for the first 10% of steps, and linear decay. A small learning rate is mandatory when fine-tuning 110M parameters to prevent catastrophic forgetting of pre-trained Bengali language knowledge."

#### Q11: "What is the function of the `[CLS]` token in BERT?"
> **Answer:** "The `[CLS]` (Classification) token is prepended to every input sequence. Since self-attention allows all tokens to attend to all other tokens across all 12 layers, the final hidden state of `[CLS]` aggregates the semantic representation of the entire sequence. We attach a linear classification head directly to $\mathbf{h}_{[\text{CLS}]}$ for sequence prediction."

---

### Dataset & Evaluation Questions

#### Q12: "How did you ensure there is NO data leakage between train and test sets?"
> **Answer:**
> 1. We split the datasets into train (70%), val (10%), and test (20%) before any vectorization.
> 2. The TF-IDF vectorizer called `fit()` strictly on `train.csv`, and only `transform()` on validation and test sets.
> 3. Out-of-vocabulary words in the test set were safely handled using the unk-token protocol without modifying vocabulary frequencies."

#### Q13: "How many total samples do you have, and what are the 95th percentile lengths?"
> **Answer:** "We have **218,371 total sentences**. Sentiment has 156,010; Hate Speech has 50,272; Sarcasm has 12,089. The 95th percentile word length across all corpora is **45 words**, which justifies setting our maximum sequence length to 50 for BiLSTM and 64/128 for BanglaBERT."

#### Q14: "Why did BanglaBERT achieve such a massive improvement on Hate Speech (91.59% Acc vs 86.61% for TF-IDF)?"
> **Answer:** "Hate speech and cyberbullying in Bengali rely on slang, metaphorical insults, and indirect threats. TF-IDF only looks at exact word tokens and fails when words are misspelled or veiled. BanglaBERT's WordPiece subword tokenization breaks unknown slang into morphological sub-roots, and its 12 self-attention layers capture toxic intent even when explicit slurs are avoided."

---

### Ensemble & Showcase Demo Questions

#### Q15: "How does the Context-Aware Sarcastic Inversion actually execute in your demo?"
> **Answer:** "When the user clicks 'Analyze Context':
> 1. All 3 models predict simultaneously.
> 2. The Sarcasm model outputs a probability $P(\text{Sarcastic})$.
> 3. If $P(\text{Sarcastic}) \ge 0.65$ and the raw sentiment prediction was 'Positive', the engine triggers **Semantic Inversion**, flipping the output sentiment to **Negative** and highlighting the detected sarcastic contrast.
> 4. Meanwhile, the Hate Speech model independently verifies whether the sarcastic comment contains abusive or harassing content."

#### Q16: "What happens if an input sentence is completely out-of-vocabulary (OOV)?"
> **Answer:** "In TF-IDF, unseen n-grams simply map to zeros, and the intercept (prior probability) determines the base prediction. In BiLSTM, unknown tokens are mapped to `<UNK>` (index 1). In BanglaBERT, WordPiece tokenization fragments unseen words into smaller known character n-grams (e.g., `##কর`), virtually eliminating OOV errors."

#### Q17: "Is there any internet requirement to run your project?"
> **Answer:** "None whatsoever. All fine-tuned models, tokenizer vocabularies, Word2Vec weights, and TF-IDF pickles are saved locally in `saved_models/`. The application runs 100% offline."

#### Q18: "What are the limitations of your project?"
> **Answer:**
> 1. Code-mixed Banglish (Bengali written in English letters, e.g., *'valo na'*) is only partially supported.
> 2. Open-domain Bengali sentiment corpora are heavily skewed toward positive reviews.
> 3. Future work involves multi-task hard parameter sharing where a single transformer trunk branches into 3 classification heads to reduce total parameters."

#### Q19: "Can you show me where the models are saved on disk?"
> **Answer:** "Yes, sir! In the project root under `Project files/saved_models/`:
> - `banglabert_sentiment/` (model weights, tokenizer, config)
> - `banglabert_sarcasm/`
> - `banglabert_hate/`
> - `bilstm_*.pt` (PyTorch checkpoints)
> - `tfidf_*.joblib` & `lr_*.joblib` (Scikit-learn serialized models)."

#### Q20: "If you had 1 more month, what would you add?"
> **Answer:** "We would implement a **Joint Multi-Task Transformer (MT-BERT)** with a shared 12-layer trunk and 3 parallel classification heads trained with uncertainty-weighted multi-task loss. We would also add a phoneme-based transliteration module to seamlessly normalize Banglish social media comments into native Bengali script."

---

### 🎯 Quick Summary for Viva / Showcase Day:
- **Project Name:** Context-Aware Bengali Text Multi-Task Analyzer
- **Course:** CSE 4122 (Natural Language Processing Sessional)
- **Top Metrics:** Hate Speech: **91.59% Acc**, Sarcasm: **76.76% Acc**, Sentiment: **76.39% Acc / 81.29% Weighted F1**
- **To Launch Demo:** Double-click `run_app.bat` or run `streamlit run "Project files/app.py"`
