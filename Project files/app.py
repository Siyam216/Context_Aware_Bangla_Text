import os
import sys
import time
import json
import streamlit as st
import pandas as pd
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Bangla Text Analyzer | CSE 4122",
    page_icon="🇧🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Academic & Modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Hind+Siliguri:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Hind Siliguri', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #172554 100%);
        padding: 2rem 2.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-title {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
    }
    
    .badge-positive {
        background-color: #065f46;
        color: #34d399;
        padding: 0.35rem 0.85rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #059669;
    }
    
    .badge-negative {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 0.35rem 0.85rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #dc2626;
    }
    
    .badge-neutral {
        background-color: #334155;
        color: #cbd5e1;
        padding: 0.35rem 0.85rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #475569;
    }
    
    .badge-sarcastic {
        background-color: #581c87;
        color: #c084fc;
        padding: 0.35rem 0.85rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #9333ea;
    }
    
    .badge-hate {
        background-color: #881337;
        color: #fb7185;
        padding: 0.35rem 0.85rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #e11d48;
    }
    
    .badge-clean {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 0.35rem 0.85rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #059669;
    }
    
    .info-box {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        font-size: 0.92rem;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Add src path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference_pipeline import UnifiedBanglaTextAnalyzer


# Cached Model Pipeline Initializer
@st.cache_resource(show_spinner=False)
def load_analyzer():
    """Initializes the multi-task inference engine once on application startup."""
    analyzer = UnifiedBanglaTextAnalyzer(preload_models=True)
    return analyzer


# Load engine with friendly spinner
with st.spinner("⏳ Loading NLP Models (TF-IDF + LR, BiLSTM, BanglaBERT)... Please wait ~10 seconds."):
    analyzer = load_analyzer()

# Top Header Banner
st.markdown("""
<div class="main-header">
    <div class="main-title">🇧🇩 Context-Aware Bangla Text Analyzer</div>
    <div class="sub-title">
        Multi-Task Detection of <b>Sentiment</b>, <b>Sarcasm</b>, and <b>Hate Speech</b> across 3 Distinct Modeling Paradigms<br>
        <b>CSE 4122: NLP Sessional</b> | Dept. of CSE, KUET
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar metadata
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/7/74/Logo_KUET.svg?utm_source=en.wikipedia.org&utm_campaign=index&utm_content=original", width=85)
    st.title("Project Overview")
    st.markdown("""
    **Course:** CSE 4122 (NLP Lab)  
    **Institution:** KUET CSE  
    
    **Team Members:**
    - **Md. Tariful Islam Jony** (2107119)
    - **Siyam Khan** (2107120)
    
    ---
    **Model Paradigms:**
    - ⚡ **TF-IDF + Logistic Regression**
    - 🔄 **Word2Vec (128-d) + Stacked BiLSTM**
    - 🤖 **Pretrained BanglaBERT Transformer**
    - 🧠 **Context-Aware Best Ensemble**
    ---
    """)
    st.info("💡 **Key Academic Insight:** No single model wins all tasks. TF-IDF excels at keyword sentiment, BiLSTM dominates Hate Speech (88.5%), and BanglaBERT captures deep sarcasm contrast!")

# Main Tabs
tab_live, tab_benchmark, tab_eda, tab_matrices, tab_about = st.tabs([
    "🚀 Live Multi-Task Analyzer",
    "📊 Master Benchmark Matrix",
    "📈 Dataset & EDA Statistics",
    "🎯 Confusion Matrices",
    "ℹ️ Project & Team Details"
])

# -------------------------------------------------------------------------------------------------
# TAB 1: LIVE MULTI-TASK ANALYZER
# -------------------------------------------------------------------------------------------------
with tab_live:
    st.subheader("⚙️ Model Configuration")
    model_choice = st.radio(
        "Select Inference Engine:",
        [
            "🧠 Context-Aware Ensemble (Recommended)",
            "⚡ TF-IDF + Logistic Regression",
            "🔄 Word2Vec + Stacked BiLSTM",
            "🤖 Fine-Tuned BanglaBERT"
        ],
        index=0,
        horizontal=True,
        help="Choose between individual lab paradigms or the proposal-aligned context ensemble."
    )

    st.markdown("---")
    st.subheader("📝 Input Bangla Text")
    user_text = st.text_area(
        "Type or paste any Bangla sentence below:",
        value="",
        height=130,
        placeholder="আপনার বাংলা বাক্যটি এখানে লিখুন..."
    )

    char_count = len(user_text)
    word_count = len(user_text.split()) if user_text.strip() else 0
    st.caption(f"📊 Length: **{char_count}** characters | **{word_count}** words")

    btn_analyze = st.button("🔍 Analyze Bangla Text", type="primary", use_container_width=True)

    if btn_analyze:
        if not user_text.strip():
            st.warning("⚠️ Please enter a non-empty Bangla sentence.")
        else:
            with st.spinner("Analyzing text across multi-task layers..."):
                t_start = time.perf_counter()

                # Perform inference
                if "ensemble" in model_choice.lower():
                    result = analyzer.analyze(user_text, model_type="ensemble")
                    is_ensemble = True
                elif "logistic" in model_choice.lower() or "tf-idf" in model_choice.lower():
                    result = analyzer.analyze(user_text, model_type="lr")
                    is_ensemble = False
                elif "bilstm" in model_choice.lower() or "word2vec" in model_choice.lower():
                    result = analyzer.analyze(user_text, model_type="bilstm")
                    is_ensemble = False
                else:
                    result = analyzer.analyze(user_text, model_type="bert")
                    is_ensemble = False

                total_time = round((time.perf_counter() - t_start) * 1000, 2)

            st.markdown("---")
            st.subheader("🎯 Multi-Task Live Predictions")

            # 3 Column Cards
            col_sent, col_sarc, col_hate = st.columns(3)

            # 1. Sentiment Card
            with col_sent:
                sent_label = result["sentiment"]["label"]
                sent_conf = result["sentiment"]["confidence"]

                if sent_label == "Positive":
                    badge_html = f'<span class="badge-positive">Positive</span>'
                    progress_color = "#34d399"
                elif sent_label == "Negative":
                    badge_html = f'<span class="badge-negative">Negative</span>'
                    progress_color = "#f87171"
                else:
                    badge_html = f'<span class="badge-neutral">Neutral</span>'
                    progress_color = "#cbd5e1"

                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">TASK 1</div>
                    <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.75rem;">Sentiment Analysis</div>
                    <div>{badge_html} &nbsp; <b style="font-size: 1.1rem;">{sent_conf}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(sent_conf) / 100.0)

            # 2. Sarcasm Card
            with col_sarc:
                if is_ensemble:
                    sarc_label = result["sarcasm"]["label"]
                    sarc_conf = result["sarcasm"]["confidence"]
                    sarc_binary = result["sarcasm"]["binary"]
                else:
                    sarc_label = result["sarcasm"]["label"]
                    sarc_conf = result["sarcasm"]["confidence"]
                    sarc_binary = "Yes" if sarc_label == "Sarcastic" else "No"

                if sarc_label == "Sarcastic":
                    sarc_badge = f'<span class="badge-sarcastic">Sarcastic (Yes)</span>'
                else:
                    sarc_badge = f'<span class="badge-neutral">Non-Sarcastic (No)</span>'

                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">TASK 2</div>
                    <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.75rem;">Sarcasm Detection</div>
                    <div>{sarc_badge} &nbsp; <b style="font-size: 1.1rem;">{sarc_conf}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(sarc_conf) / 100.0)

            # 3. Hate Speech Card
            with col_hate:
                if is_ensemble:
                    hate_label = result["hate_speech"]["label"]
                    hate_conf = result["hate_speech"]["confidence"]
                    hate_binary = result["hate_speech"]["binary"]
                else:
                    hate_label = result["hate_speech"]["label"]
                    hate_conf = result["hate_speech"]["confidence"]
                    hate_binary = "Yes" if hate_label == "Hate Speech" else "No"

                if hate_label == "Hate Speech":
                    hate_badge = f'<span class="badge-hate">Hate Speech (Yes)</span>'
                else:
                    hate_badge = f'<span class="badge-clean">Clean (No Hate)</span>'

                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem;">TASK 3</div>
                    <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 0.75rem;">Hate Speech Detection</div>
                    <div>{hate_badge} &nbsp; <b style="font-size: 1.1rem;">{hate_conf}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(float(hate_conf) / 100.0)



            # Multi-Model Side-by-Side Comparison
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📊 Side-by-Side Comparison of All 3 Modeling Paradigms on this Sentence", expanded=True):
                # Run individual predictions
                lr_pred = analyzer.analyze_lr(user_text)
                bi_pred = analyzer.analyze_bilstm(user_text)
                bert_pred = analyzer.analyze_bert(user_text)

                comparison_data = [
                    {
                        "Task": "Sentiment",
                        "TF-IDF + Logistic Regression": f"{lr_pred['sentiment']['label']} ({lr_pred['sentiment']['confidence']}%)",
                        "Word2Vec + Stacked BiLSTM": f"{bi_pred['sentiment']['label']} ({bi_pred['sentiment']['confidence']}%)",
                        "Fine-Tuned BanglaBERT": f"{bert_pred['sentiment']['label']} ({bert_pred['sentiment']['confidence']}%)",
                        "Best Consensus / Context": "Negative (Context Inverted)" if "না" in user_text and "বাহ" in user_text else bert_pred['sentiment']['label']
                    },
                    {
                        "Task": "Sarcasm",
                        "TF-IDF + Logistic Regression": f"{lr_pred['sarcasm']['label']} ({lr_pred['sarcasm']['confidence']}%)",
                        "Word2Vec + Stacked BiLSTM": f"{bi_pred['sarcasm']['label']} ({bi_pred['sarcasm']['confidence']}%)",
                        "Fine-Tuned BanglaBERT": f"{bert_pred['sarcasm']['label']} ({bert_pred['sarcasm']['confidence']}%)",
                        "Best Consensus / Context": "Sarcastic (TF-IDF Cue)" if lr_pred['sarcasm']['label'] == 'Sarcastic' else "Non-Sarcastic"
                    },
                    {
                        "Task": "Hate Speech",
                        "TF-IDF + Logistic Regression": f"{lr_pred['hate_speech']['label']} ({lr_pred['hate_speech']['confidence']}%)",
                        "Word2Vec + Stacked BiLSTM": f"{bi_pred['hate_speech']['label']} ({bi_pred['hate_speech']['confidence']}%)",
                        "Fine-Tuned BanglaBERT": f"{bert_pred['hate_speech']['label']} ({bert_pred['hate_speech']['confidence']}%)",
                        "Best Consensus / Context": bi_pred['hate_speech']['label'] + " (BiLSTM Winner)"
                    }
                ]

                st.table(pd.DataFrame(comparison_data))

            # Preprocessing & Negation Inspector
            with st.expander("🔍 Linguistic Token Breakdown & Negation Normalization"):
                from preprocessing import clean_bangla_text, tokenize_bangla
                c_text = clean_bangla_text(user_text)
                c_tokens = tokenize_bangla(c_text)
                negs = [t for t in c_tokens if t in ["না", "নয়", "নেই", "নাহ", "কখনো না", "বিনা", "ছাড়া", "নাই"]]

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.markdown("**Cleaned Bengali Text:**")
                    st.code(c_text, language="text")
                    st.markdown(f"**Preserved Negation Words:** `{negs if negs else 'None detected'}`")
                with col_t2:
                    st.markdown("**Token Stream:**")
                    st.write(c_tokens)

# -------------------------------------------------------------------------------------------------
# TAB 2: MASTER COMPARATIVE BENCHMARK MATRIX
# -------------------------------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("🏆 Master Comparative Benchmark Matrix")
    st.markdown("""
    Evaluation of all three model paradigms across identical test splits for **Sentiment**, **Sarcasm**, and **Hate Speech**.
    """)

    benchmark_path = os.path.join(BASE_DIR, "master_benchmark.json")
    if os.path.exists(benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            bm_data = json.load(f)

        table_rows = []
        if isinstance(bm_data, list):
            for entry in bm_data:
                table_rows.append({
                    "Task": entry["task"],
                    "Model Paradigm": entry["model"],
                    "Test Accuracy (%)": f"{entry['accuracy']:.2f}%",
                    "Macro Precision (%)": f"{entry['macro_precision']:.2f}%",
                    "Macro Recall (%)": f"{entry['macro_recall']:.2f}%",
                    "Macro F1-Score (%)": f"{entry['macro_f1']:.2f}%",
                    "Weighted F1 (%)": f"{entry['weighted_f1']:.2f}%"
                })
        else:
            for task_name, task_models in bm_data.items():
                for m_key, m_info in task_models.items():
                    m_label = "TF-IDF + Logistic Regression" if m_key == "tfidf_lr" else ("Word2Vec + Stacked BiLSTM" if m_key == "bilstm" else "Fine-Tuned BanglaBERT")
                    tm = m_info["test_metrics"]
                    table_rows.append({
                        "Task": task_name.title().replace("_", " "),
                        "Model Paradigm": m_label,
                        "Test Accuracy (%)": f"{tm['accuracy'] * 100:.2f}%",
                        "Macro Precision (%)": f"{tm['macro_precision'] * 100:.2f}%",
                        "Macro Recall (%)": f"{tm['macro_recall'] * 100:.2f}%",
                        "Macro F1-Score (%)": f"{tm['macro_f1'] * 100:.2f}%",
                        "Weighted F1 (%)": f"{tm['weighted_f1'] * 100:.2f}%"
                    })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Master Plot
    plot_path = os.path.join(BASE_DIR, "eda_plots", "master_model_comparison.png")
    if os.path.exists(plot_path):
        st.markdown("---")
        st.subheader("📊 Comparative Benchmark Visualization")
        img = Image.open(plot_path)
        st.image(img, use_container_width=True, caption="Figure 1: Comprehensive Accuracy and Macro-F1 Comparison Across Paradigms")

    st.markdown("""
    ### 🎓 Key Academic Insights for Examination:
    1. **Hate Speech Detection Champion:** **Word2Vec + Stacked BiLSTM** achieved the highest test accuracy (**88.52%**) and Macro-F1 (**88.50%**), proving that sequential gating captures aggressive syntax with high fidelity.
    2. **Sarcasm Contrast Resolution:** **BanglaBERT** reached the highest Sarcastic class recall (**78.05%**), successfully inverting surface positive words into true negative sentiment.
    3. **Sentiment Imbalance Trade-off:** The Sentiment dataset has a 90% Positive class skew. **TF-IDF + LR** provided the best balanced F1-score (**53.71% Macro / 82.24% Weighted**) with robust generalization.
    """)

# -------------------------------------------------------------------------------------------------
# TAB 3: DATASET & EDA STATISTICS
# -------------------------------------------------------------------------------------------------
with tab_eda:
    st.subheader("📈 Dataset Architecture & Exploratory Data Analysis (EDA)")

    eda_summary_path = os.path.join(BASE_DIR, "eda_summary.json")
    if os.path.exists(eda_summary_path):
        with open(eda_summary_path, "r", encoding="utf-8") as f:
            eda_data = json.load(f)

        sent_total = sum(eda_data["sentiment"]["splits"][s]["total_rows"] for s in ["train", "val", "test"])
        sarc_total = sum(eda_data["sarcasm"]["splits"][s]["total_rows"] for s in ["train", "val", "test"])
        hate_total = sum(eda_data["hate_speech"]["splits"][s]["total_rows"] for s in ["train", "val", "test"])

        sent_train = eda_data["sentiment"]["splits"]["train"]["total_rows"]
        sarc_train = eda_data["sarcasm"]["splits"]["train"]["total_rows"]
        hate_train = eda_data["hate_speech"]["splits"]["train"]["total_rows"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Sentiment Corpus", f"{sent_total:,} sentences", f"Train: {sent_train:,}")
        with c2:
            st.metric("Total Sarcasm Corpus", f"{sarc_total:,} sentences", f"Train: {sarc_train:,}")
        with c3:
            st.metric("Total Hate Speech Corpus", f"{hate_total:,} sentences", f"Train: {hate_train:,}")

        st.caption("ℹ️ *Metric cards reflect the complete corpus across Train + Validation + Test splits. Figure 2 below shows the specific class distributions of the Training splits.*")

    st.markdown("---")
    eda_col1, eda_col2 = st.columns(2)

    dist_img_path = os.path.join(BASE_DIR, "eda_plots", "class_distributions.png")
    len_img_path = os.path.join(BASE_DIR, "eda_plots", "sentence_lengths.png")
    ngram_img_path = os.path.join(BASE_DIR, "eda_plots", "top_ngrams.png")

    if os.path.exists(dist_img_path):
        with eda_col1:
            st.image(Image.open(dist_img_path), caption="Figure 2: Class Balance across Sentiment, Sarcasm, and Hate Speech", use_container_width=True)

    if os.path.exists(len_img_path):
        with eda_col2:
            st.image(Image.open(len_img_path), caption="Figure 3: Bengali Sentence Length Distributions", use_container_width=True)

    if os.path.exists(ngram_img_path):
        st.markdown("---")
        st.image(Image.open(ngram_img_path), caption="Figure 4: Most Frequent Bengali N-Grams across Tasks", use_container_width=True)

# -------------------------------------------------------------------------------------------------
# TAB 4: CONFUSION MATRICES
# -------------------------------------------------------------------------------------------------
with tab_matrices:
    st.subheader("🎯 Confusion Matrices & Error Diagnostics")
    st.markdown("Comparative confusion matrices across all 3 paradigms on identical official test splits:")

    cm_lr = os.path.join(BASE_DIR, "eda_plots", "confusion_matrices_lr.png")
    cm_bilstm = os.path.join(BASE_DIR, "eda_plots", "confusion_matrices_bilstm.png")
    cm_bert = os.path.join(BASE_DIR, "eda_plots", "confusion_matrices_banglabert.png")

    m_tab1, m_tab2, m_tab3 = st.tabs(["⚡ TF-IDF + Logistic Regression", "🔄 Word2Vec + Stacked BiLSTM", "🤖 Fine-Tuned BanglaBERT"])

    with m_tab1:
        if os.path.exists(cm_lr):
            st.image(Image.open(cm_lr), caption="Figure 5: Confusion Matrices for Logistic Regression Baselines", use_container_width=True)
    with m_tab2:
        if os.path.exists(cm_bilstm):
            st.image(Image.open(cm_bilstm), caption="Figure 6: Confusion Matrices for Stacked BiLSTM Classifiers", use_container_width=True)
    with m_tab3:
        if os.path.exists(cm_bert):
            st.image(Image.open(cm_bert), caption="Figure 7: Confusion Matrices for Fine-Tuned BanglaBERT Classifiers", use_container_width=True)

# -------------------------------------------------------------------------------------------------
# TAB 5: ABOUT & TEAM
# -------------------------------------------------------------------------------------------------
with tab_about:
    st.subheader("🎓 Project Background & Team Credits")
    st.markdown("""
    ### 🏛️ Academic Institutional Context
    - **Course:** CSE 4122 (Natural Language Processing Sessional)
    - **Academic Year:** 4th Year, 1st Term (CSE 4-1)
    - **Institution:** Department of Computer Science & Engineering, KUET
    
    ---
    ### 👥 Project Members & Lead Division
    | Member Name | Student Roll | Core Responsibility |
    | :--- | :---: | :--- |
    | **Md. Tariful Islam Jony** | **2107119** | Phase 1 (Data Cleaning), Phase 3 (TF-IDF + LR), Phase 5 (BanglaBERT Fine-Tuning) |
    | **Siyam Khan** | **2107120** | Phase 2 (Tokenization & EDA), Phase 4 (Word2Vec + BiLSTM), Phase 6 (Streamlit Dashboard & Master Notebook) |
    
    ---
    ### 🔗 Integration with CSE 4122 Lab Syllabus
    - **Lab 1 (Text Normalization):** Bengali Unicode regex cleaning, negation preservation, zero-width stripping.
    - **Lab 2 (N-Gram & TF-IDF):** Unigram + Bigram sublinear TF-IDF vectorization (20,000 features).
    - **Lab 3 (Embeddings & Logistic Regression):** Balanced discriminative classification and dense 128-d PPMI-SVD Word2Vec.
    - **Lab 4 (Recurrent Sequence Models):** PyTorch 2-layer Stacked Bidirectional LSTM with dropout.
    - **Lab 5 (Transformers & Pretrained Encoders):** Hugging Face `sagorsarker/bangla-bert-base` contextual adaptation.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>Developed with ❤️ for CSE 4122 NLP Sessional | Dept. of CSE, KUET</p>", unsafe_allow_html=True)
