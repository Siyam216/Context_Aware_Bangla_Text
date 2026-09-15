"""
Batch Dataset Ingestion, Cleaning & Standardization Pipeline for Phase 1.
Implements:
1. Column pruning (keeping only text & label).
2. Regex cleaning via clean_bangla_text.
3. Sarcasm dataset parsing & 80/10/10 stratified split.
4. Exporting uniform (text, label) CSVs to Project files/cleaned_data/.
Lead: Md. Tariful Islam Jony (ID: 2107119)
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import clean_bangla_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import clean_bangla_text

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DATA_DIR = os.path.join(WORKSPACE_ROOT, "Dataset")
CLEAN_DATA_DIR = os.path.join(WORKSPACE_ROOT, "Project files", "cleaned_data")


def process_sentiment():
    print("\n" + "="*50)
    print(">>> [1/3] Processing Sentiment Dataset...")
    raw_sentiment_dir = os.path.join(RAW_DATA_DIR, "Sentiment")
    out_dir = os.path.join(CLEAN_DATA_DIR, "sentiment")
    os.makedirs(out_dir, exist_ok=True)

    file_mapping = {
        "train.csv": "train.csv",
        "validation.csv": "val.csv",
        "test.csv": "test.csv"
    }

    # Label mapping: 'negative' -> 0, 'neutral' -> 1, 'positive' -> 2
    label_str_map = {"negative": 0, "neutral": 1, "positive": 2}

    for raw_name, clean_name in file_mapping.items():
        raw_path = os.path.join(raw_sentiment_dir, raw_name)
        out_path = os.path.join(out_dir, clean_name)
        print(f"Reading {raw_path}...")
        df = pd.read_csv(raw_path, encoding='utf-8', on_bad_lines='skip')

        # Check column names (handle possible BOM in column name)
        df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]

        # Extract text
        text_col = 'Review' if 'Review' in df.columns else df.columns[5]
        label_col = 'label' if 'label' in df.columns else 'sentiment'

        print(f"Extracting columns: Text='{text_col}', Label='{label_col}'")
        texts = df[text_col].astype(str).apply(clean_bangla_text)
        
        # Handle label mapping
        raw_labels = df[label_col]
        if raw_labels.dtype == object:
            # Map string to int if needed
            labels = raw_labels.str.strip().str.lower().map(
                lambda x: label_str_map.get(x, int(x) if str(x).isdigit() else None)
            )
        else:
            labels = raw_labels.astype(int)

        clean_df = pd.DataFrame({"text": texts, "label": labels})
        
        # Drop empty texts or null labels
        clean_df = clean_df[clean_df["text"].str.strip().str.len() > 1]
        clean_df = clean_df.dropna(subset=["label"])
        clean_df["label"] = clean_df["label"].astype(int)

        clean_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"Saved: {out_path} ({len(clean_df):,} rows)")
        print("Class distribution:", dict(clean_df['label'].value_counts()))


def process_hate_speech():
    print("\n" + "="*50)
    print(">>> [2/3] Processing Hate Speech Dataset...")
    raw_hate_dir = os.path.join(RAW_DATA_DIR, "Hate Speech")
    out_dir = os.path.join(CLEAN_DATA_DIR, "hate_speech")
    os.makedirs(out_dir, exist_ok=True)

    file_mapping = {
        "train.csv": "train.csv",
        "val.csv": "val.csv",
        "test.csv": "test.csv"
    }

    for raw_name, clean_name in file_mapping.items():
        raw_path = os.path.join(raw_hate_dir, raw_name)
        out_path = os.path.join(out_dir, clean_name)
        print(f"Reading {raw_path}...")
        df = pd.read_csv(raw_path, encoding='utf-8', on_bad_lines='skip')
        df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]

        text_col = 'sentence'
        label_col = 'hate speech'

        print(f"Extracting columns: Text='{text_col}', Label='{label_col}'")
        texts = df[text_col].astype(str).apply(clean_bangla_text)
        labels = pd.to_numeric(df[label_col], errors='coerce')

        clean_df = pd.DataFrame({"text": texts, "label": labels})
        clean_df = clean_df[clean_df["text"].str.strip().str.len() > 1]
        clean_df = clean_df.dropna(subset=["label"])
        clean_df["label"] = clean_df["label"].astype(int)

        clean_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"Saved: {out_path} ({len(clean_df):,} rows)")
        print("Class distribution:", dict(clean_df['label'].value_counts()))


def process_sarcasm():
    print("\n" + "="*50)
    print(">>> [3/3] Processing Sarcasm Dataset & Stratified Split...")
    raw_sarcasm_path = os.path.join(RAW_DATA_DIR, "Sarcasm", "BanglaSarc3 (Original).xlsx")
    out_dir = os.path.join(CLEAN_DATA_DIR, "sarcasm")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading Excel: {raw_sarcasm_path}...")
    df = pd.read_excel(raw_sarcasm_path)
    df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]

    text_col = 'Bangla Comments'
    label_col = 'Sarcasm Label'

    print(f"Extracting columns: Text='{text_col}', Label='{label_col}'")
    texts = df[text_col].astype(str).apply(clean_bangla_text)
    
    # Binary mapping: 'Sarcastic' -> 1, 'Non-Sarcastic' / 'Neutral' -> 0
    def map_sarcasm(label):
        val = str(label).strip()
        if val == 'Sarcastic':
            return 1
        elif val in ['Non-Sarcastic', 'Neutral']:
            return 0
        return None

    labels = df[label_col].apply(map_sarcasm)

    clean_df = pd.DataFrame({"text": texts, "label": labels})
    clean_df = clean_df[clean_df["text"].str.strip().str.len() > 1]
    clean_df = clean_df.dropna(subset=["label"])
    clean_df["label"] = clean_df["label"].astype(int)

    print(f"Total valid cleaned samples: {len(clean_df):,}")
    print("Overall Class distribution (0=Non-Sarcastic, 1=Sarcastic):", dict(clean_df['label'].value_counts()))

    # Stratified Split: 80% Train, 10% Val, 10% Test
    train_df, temp_df = train_test_split(
        clean_df, test_size=0.20, random_state=42, stratify=clean_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )

    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')

    print(f"Saved: {train_path} ({len(train_df):,} rows)")
    print(f"Saved: {val_path} ({len(val_df):,} rows)")
    print(f"Saved: {test_path} ({len(test_df):,} rows)")


def verify_all_datasets():
    print("\n" + "="*50)
    print(">>> VERIFICATION REPORT: Cleaned Data Integrity Checks")
    print("="*50)

    tasks = ["sentiment", "sarcasm", "hate_speech"]
    splits = ["train.csv", "val.csv", "test.csv"]

    all_passed = True
    summary_records = []

    for task in tasks:
        for split in splits:
            fpath = os.path.join(CLEAN_DATA_DIR, task, split)
            if not os.path.exists(fpath):
                print(f"❌ MISSING FILE: {fpath}")
                all_passed = False
                continue

            df = pd.read_csv(fpath)
            # Checks
            cols = list(df.columns)
            cols_ok = (cols == ["text", "label"])
            nan_count = df.isna().sum().sum()
            row_count = len(df)
            class_dist = dict(df['label'].value_counts())

            status = "✅ PASS" if cols_ok and nan_count == 0 and row_count > 0 else "❌ FAIL"
            if status == "❌ FAIL":
                all_passed = False

            summary_records.append({
                "Task": task,
                "Split": split,
                "Rows": f"{row_count:,}",
                "Columns": str(cols),
                "NaN Count": nan_count,
                "Class Distribution": str(class_dist),
                "Status": status
            })

    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

    if all_passed:
        print("\n🎉 ALL PHASE 1 GOALS SUCCESSFULLY VERIFIED & ACHIEVED!")
    else:
        print("\n⚠️ SOME CHECKS FAILED. PLEASE REVIEW ABOVE LOGS.")


if __name__ == "__main__":
    process_sentiment()
    process_hate_speech()
    process_sarcasm()
    verify_all_datasets()
