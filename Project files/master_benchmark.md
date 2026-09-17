# Master Comparative Evaluation Matrix (CSE 4121 NLP Project)

Comprehensive benchmark across all three modeling paradigms on identical official test splits:

| Task | Model Paradigm | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1 (%) | Inference Time (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentiment** | TF-IDF + Logistic Regression | 77.60% | 50.78% | 67.13% | **53.71%** | 82.24% | ~2.1 ms |
| **Sentiment** | Word2Vec + Stacked BiLSTM | 76.60% | 52.75% | 63.23% | **53.27%** | 81.96% | ~4.8 ms |
| **Sentiment** | Fine-Tuned BanglaBERT | 62.46% | 41.95% | 56.57% | **40.75%** | 70.95% | ~42.0 ms |
| **Sarcasm** | TF-IDF + Logistic Regression | 74.94% | 72.37% | 74.09% | **72.89%** | 75.40% | ~2.1 ms |
| **Sarcasm** | Word2Vec + Stacked BiLSTM | 74.36% | 71.53% | 72.84% | **72.00%** | 74.74% | ~4.8 ms |
| **Sarcasm** | Fine-Tuned BanglaBERT | 73.86% | 72.34% | 74.92% | **72.52%** | 74.57% | ~42.0 ms |
| **Hate Speech** | TF-IDF + Logistic Regression | 86.61% | 86.96% | 86.42% | **86.52%** | 86.57% | ~2.1 ms |
| **Hate Speech** | Word2Vec + Stacked BiLSTM | 88.52% | 88.54% | 88.47% | **88.50%** | 88.52% | ~4.8 ms |
| **Hate Speech** | Fine-Tuned BanglaBERT | 76.71% | 76.68% | 76.69% | **76.68%** | 76.71% | ~42.0 ms |
