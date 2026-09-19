# Master Comparative Evaluation Matrix (CSE 4122 NLP Project)

Comprehensive benchmark across all three modeling paradigms on identical official test splits:

| Task | Model Paradigm | Test Accuracy (%) | Macro Precision (%) | Macro Recall (%) | Macro F1-Score (%) | Weighted F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sentiment** | TF-IDF + Logistic Regression | 77.60% | 50.78% | 67.13% | **53.71%** | 82.24% |
| **Sentiment** | Word2Vec + Stacked BiLSTM | 76.60% | 52.75% | 63.23% | **53.27%** | 81.96% |
| **Sentiment** | Fine-Tuned BanglaBERT | 76.39% | 48.81% | 64.57% | **51.51%** | 81.29% |
| **Sarcasm** | TF-IDF + Logistic Regression | 74.94% | 72.37% | 74.09% | **72.89%** | 75.40% |
| **Sarcasm** | Word2Vec + Stacked BiLSTM | 74.36% | 71.53% | 72.84% | **72.00%** | 74.74% |
| **Sarcasm** | Fine-Tuned BanglaBERT | 76.76% | 74.30% | 76.21% | **74.89%** | 77.19% |
| **Hate Speech** | TF-IDF + Logistic Regression | 86.61% | 86.96% | 86.42% | **86.52%** | 86.57% |
| **Hate Speech** | Word2Vec + Stacked BiLSTM | 88.52% | 88.54% | 88.47% | **88.50%** | 88.52% |
| **Hate Speech** | Fine-Tuned BanglaBERT | 91.59% | 91.56% | 91.60% | **91.58%** | 91.59% |
