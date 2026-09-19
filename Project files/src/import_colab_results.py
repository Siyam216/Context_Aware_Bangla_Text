"""
Helper script to import trained BanglaBERT weights and results from Google Colab.
Usage:
1. Download 'banglabert_colab_results.zip' from Google Colab into your Downloads folder or project root.
2. Run: python "Project files/src/import_colab_results.py"
"""

import os
import sys
import zipfile
import shutil
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

possible_zip_locations = [
    os.path.join(WORKSPACE_ROOT, "banglabert_colab_results.zip"),
    os.path.join(BASE_DIR, "banglabert_colab_results.zip"),
    os.path.expanduser("~/Downloads/banglabert_colab_results.zip")
]

zip_path = None
for loc in possible_zip_locations:
    if os.path.exists(loc):
        zip_path = loc
        break

if not zip_path:
    print("[*] 'banglabert_colab_results.zip' not found (already extracted or deleted).")
    print("    Checking if Project files/results_banglabert.json exists...")
    if not os.path.exists(os.path.join(BASE_DIR, "results_banglabert.json")):
        print("[-] results_banglabert.json not found. Exiting.")
        sys.exit(1)
    print("[+] Found results_banglabert.json. Skipping extraction and proceeding to plot updates.")
else:
    print(f"[+] Found Colab package: {zip_path}")
    print("[+] Extracting new fine-tuned models...")

    extract_dir = os.path.join(BASE_DIR, "temp_colab_extract")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    # 1. Copy saved_models
    src_models = os.path.join(extract_dir, "saved_models")
    dest_models = os.path.join(BASE_DIR, "saved_models")

    if os.path.exists(src_models):
        for item in os.listdir(src_models):
            s = os.path.join(src_models, item)
            d = os.path.join(dest_models, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
                print(f"    -> Updated model directory: {item}")

    # 2. Copy results_banglabert.json
    src_results = os.path.join(extract_dir, "results_banglabert.json")
    dest_results = os.path.join(BASE_DIR, "results_banglabert.json")
    if os.path.exists(src_results):
        shutil.copy(src_results, dest_results)
        print("    -> Updated Project files/results_banglabert.json")

    # Clean temp dir
    shutil.rmtree(extract_dir, ignore_errors=True)
    print("[+] Successfully replaced local BanglaBERT weights with GPU-trained models!")



# 3. Automatically regenerate Master Benchmark
print("[+] Updating Master Benchmark Matrix...")
python_exe = sys.executable
bench_script = os.path.join(BASE_DIR, "src", "evaluate_master_benchmark.py")
subprocess.run([python_exe, bench_script], check=True)

# 4. Automatically regenerate BanglaBERT Confusion Matrix Plot
print("[+] Regenerating BanglaBERT Confusion Matrix Heatmaps...")
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json

    results_path = os.path.join(BASE_DIR, "results_banglabert.json")
    plots_dir = os.path.join(BASE_DIR, "eda_plots")
    output_fig_path = os.path.join(plots_dir, "confusion_matrices_banglabert.png")

    with open(results_path, "r", encoding="utf-8") as f:
        results_dict = json.load(f)

    tasks = {
        "sarcasm": {"class_names": ["Non-Sarcastic", "Sarcastic"]},
        "hate_speech": {"class_names": ["Non-Hate", "Hate Speech"]},
        "sentiment": {"class_names": ["Negative", "Neutral", "Positive"]}
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.35)

    for idx, (task_name, config) in enumerate(tasks.items()):
        ax = axes[idx]
        cm = np.array(results_dict[task_name]["confusion_matrix"])
        class_names = config["class_names"]

        cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

        cmap = "Purples" if task_name == "sarcasm" else ("Reds" if task_name == "hate_speech" else "Blues")
        sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})

        test_f1 = results_dict[task_name]["test_metrics"]["macro_f1"] * 100
        test_acc = results_dict[task_name]["test_metrics"]["accuracy"] * 100

        task_title = "Hate Speech" if task_name == "hate_speech" else task_name.title()
        title_str = f"BanglaBERT {task_title} Matrix\nAcc: {test_acc:.1f}% | Macro-F1: {test_f1:.1f}%"
        ax.set_title(title_str, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=11, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300)
    plt.close()
    print(f"[+] Updated BanglaBERT Confusion Matrix Plot saved -> {output_fig_path}")
except Exception as e:
    print(f"[!] Warning: Could not regenerate confusion matrix plot: {e}")

print("\n" + "="*70)
print("SUCCESS: Master Benchmark, Streamlit App & Confusion Matrices are now updated!")
print("Refresh your Streamlit browser tab to see the new benchmark scores.")
print("="*70)

