# Master Comparative Evaluation Matrix (CSE 4121 NLP Project)

Comprehensive benchmark across all three modeling paradigms on identical official test splits:

| Task | Model Paradigm | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sentiment** | TF-IDF + Logistic Regression | 77.60% | 50.78% | 67.13% | **53.71%** | 82.24% |
| **Sentiment** | Word2Vec + Stacked BiLSTM | 76.60% | 52.75% | 63.23% | **53.27%** | 81.96% |
| **Sentiment** | Fine-Tuned BanglaBERT | 62.46% | 41.95% | 56.57% | **40.75%** | 70.95% |
| **Sarcasm** | TF-IDF + Logistic Regression | 74.94% | 72.37% | 74.09% | **72.89%** | 75.40% |
| **Sarcasm** | Word2Vec + Stacked BiLSTM | 74.36% | 71.53% | 72.84% | **72.00%** | 74.74% |
| **Sarcasm** | Fine-Tuned BanglaBERT | 73.86% | 72.34% | 74.92% | **72.52%** | 74.57% |
| **Hate Speech** | TF-IDF + Logistic Regression | 86.61% | 86.96% | 86.42% | **86.52%** | 86.57% |
| **Hate Speech** | Word2Vec + Stacked BiLSTM | 88.52% | 88.54% | 88.47% | **88.50%** | 88.52% |
| **Hate Speech** | Fine-Tuned BanglaBERT | 76.71% | 76.68% | 76.69% | **76.68%** | 76.71% |
