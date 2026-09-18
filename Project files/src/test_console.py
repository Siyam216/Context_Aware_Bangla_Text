"""
Interactive Console Testing Tool
Compare TF-IDF+LR, Word2Vec+BiLSTM, and BanglaBERT side-by-side on any custom Bangla text.

Usage:
    # 1. Interactive terminal prompt:
    python "Project files/src/test_console.py"

    # 2. Direct sentence via command-line:
    python "Project files/src/test_console.py" "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!"
"""

import sys
import os
import time

# Ensure src path is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Auto-detect and switch to .venv interpreter if running under external/global python
venv_candidates = [
    os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".venv", "Scripts", "python.exe")),
    os.path.abspath(os.path.join(current_dir, "..", "..", ".venv", "Scripts", "python.exe")),
    r"D:\D Drive\CSE 4-1\CSE 4121\Lab\.venv\Scripts\python.exe"
]
for venv_python in venv_candidates:
    if os.path.exists(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
        import subprocess
        result = subprocess.run([venv_python] + sys.argv)
        sys.exit(result.returncode)

from predict_lr import BanglaTextAnalyzerLR
from predict_bilstm import BanglaTextAnalyzerBiLSTM
from predict_bert import BanglaTextAnalyzerBERT


def print_comparison(raw_text: str, lr_res: dict, bilstm_res: dict, bert_res: dict):
    print("\n" + "=" * 80)
    print(f"INPUT TEXT: \"{raw_text}\"")
    print("=" * 80)
    
    header = f"{'Task':<14} | {'TF-IDF + LR':<22} | {'BiLSTM (Word2Vec)':<22} | {'Fine-Tuned BanglaBERT':<22}"
    print(header)
    print("-" * 80)
    
    for task_key, task_name in [("sentiment", "Sentiment"), ("sarcasm", "Sarcasm"), ("hate_speech", "Hate Speech")]:
        lr_val = f"{lr_res[task_key]['label']} ({lr_res[task_key]['confidence']}%)"
        bilstm_val = f"{bilstm_res[task_key]['label']} ({bilstm_res[task_key]['confidence']}%)"
        bert_val = f"{bert_res[task_key]['label']} ({bert_res[task_key]['confidence']}%)"
        print(f"{task_name:<14} | {lr_val:<22} | {bilstm_val:<22} | {bert_val:<22}")
        
    print("=" * 80 + "\n")


def main():
    print("\n[INFO] Initializing and loading all 3 models into memory... Please wait.")
    t0 = time.time()
    lr_analyzer = BanglaTextAnalyzerLR()
    bilstm_analyzer = BanglaTextAnalyzerBiLSTM()
    bert_analyzer = BanglaTextAnalyzerBERT()
    print(f"[SUCCESS] All 3 models loaded in {time.time() - t0:.2f} seconds!\n")

    # If sentence provided as CLI arguments:
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:]).strip()
        if text:
            print(f"Analyzing provided text: \"{text}\"...")
            lr_r = lr_analyzer.analyze(text)
            bi_r = bilstm_analyzer.analyze(text)
            bert_r = bert_analyzer.analyze(text)
            print_comparison(text, lr_r, bi_r, bert_r)
            return

    # Interactive loop:
    print("=" * 80)
    print(" CONTEXT-AWARE BANGLA TEXT ANALYZER - INTERACTIVE CONSOLE")
    print(" Type any Bangla sentence to analyze, or type 'exit' or 'q' to quit.")
    print("=" * 80)

    # Run default benchmark samples first
    print("\n--- Running 1 Demo Sentence Automatically ---")
    demo_sample = "বাহ! কী অসাধারণ service, তিন ঘণ্টা অপেক্ষা করেও কাজ হলো না!"
    lr_r = lr_analyzer.analyze(demo_sample)
    bi_r = bilstm_analyzer.analyze(demo_sample)
    bert_r = bert_analyzer.analyze(demo_sample)
    print_comparison(demo_sample, lr_r, bi_r, bert_r)

    while True:
        try:
            print("Enter Bangla sentence (or 'q' to exit):")
            user_input = input(">> ").strip()
            if not user_input or user_input.lower() in ["exit", "q", "quit"]:
                print("Exiting interactive test. Have a great day!")
                break
            
            lr_r = lr_analyzer.analyze(user_input)
            bi_r = bilstm_analyzer.analyze(user_input)
            bert_r = bert_analyzer.analyze(user_input)
            print_comparison(user_input, lr_r, bi_r, bert_r)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive test. Have a great day!")
            break


if __name__ == "__main__":
    main()
