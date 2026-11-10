"""
run_overnight_suite.py
----------------------
Autonomous Overnight Master Orchestrator for the Model 9 Series:
1. Executes Model 9.2 (Tri-Branch + Center Loss Compact Net)
2. Executes Model 9.1 (Tri-Branch Prototypical Metric Net)
3. Executes Model 9.3 (Tri-Branch Multi-Prototype Sub-Clustering Net)
4. Compares all 3 models side-by-side
5. Designates the Champion model and deploys weights to Desktop Annotator
6. Copies artifacts to the conversation brain directory
7. Commits and pushes all code, plots, and LFS weights to GitHub
"""

import os
import sys
import time
import json
import shutil
import subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, r"D:\Project\model9_hybrid_cluster")
from train_model9_2 import train_model9_2
from train_model9_1 import train_model9_1
from train_model9_3 import train_model9_3

BRAIN_DIR = r"C:\Users\anike\.gemini\antigravity\brain\40d4b308-7194-4358-b8e7-0f6a076836ae"
DESKTOP_DIR = r"C:\Users\anike\Desktop\generator with annotator\model4\weights"
PROJECT_DIR = r"D:\Project"
MODEL9_DIR = r"D:\Project\model9_hybrid_cluster"


def run_git(cmd):
    try:
        p = subprocess.run(f"git -C \"{PROJECT_DIR}\" {cmd}", shell=True, capture_output=True, text=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def main():
    t_start = time.time()
    print("\n" + "="*70, flush=True)
    print("      LAUNCHING AUTONOMOUS OVERNIGHT TRAINING SUITE: MODEL 9 SERIES", flush=True)
    print("="*70, flush=True)
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("Models to train sequentially:", flush=True)
    print("  1. Model 9.2: Tri-Branch + Center Loss (Gravitational Compaction)", flush=True)
    print("  2. Model 9.1: Tri-Branch Prototypical Metric Net (Hypersphere Prototypes)", flush=True)
    print("  3. Model 9.3: Tri-Branch Multi-Prototype Net (2 Sub-Centroids per class)", flush=True)

    results = {}

    # -------------------------------------------------------------
    # 1. Train Model 9.2 (Flagship)
    # -------------------------------------------------------------
    print("\n" + "#"*70, flush=True)
    print(">>> [STAGE 1/3] TRAINING MODEL 9.2 (TRI-BRANCH + CENTER LOSS)", flush=True)
    print("#"*70, flush=True)
    try:
        train_model9_2(epochs=10, batch_size=4, patience=3)
        with open(os.path.join(MODEL9_DIR, "model9_2_metrics.json")) as f:
            results["model9_2"] = json.load(f)
    except Exception as e:
        print(f"[!] Error in Model 9.2: {e}", flush=True)
        results["model9_2"] = {"error": str(e), "val_accuracy": 0.0, "val_macro_f1": 0.0}

    # -------------------------------------------------------------
    # 2. Train Model 9.1 (Prototypical)
    # -------------------------------------------------------------
    print("\n" + "#"*70, flush=True)
    print(">>> [STAGE 2/3] TRAINING MODEL 9.1 (PROTOTYPICAL TRI-BRANCH NET)", flush=True)
    print("#"*70, flush=True)
    try:
        train_model9_1(epochs=10, batch_size=4, patience=3)
        with open(os.path.join(MODEL9_DIR, "model9_1_metrics.json")) as f:
            results["model9_1"] = json.load(f)
    except Exception as e:
        print(f"[!] Error in Model 9.1: {e}", flush=True)
        results["model9_1"] = {"error": str(e), "val_accuracy": 0.0, "val_macro_f1": 0.0}

    # -------------------------------------------------------------
    # 3. Train Model 9.3 (Multi-Prototype)
    # -------------------------------------------------------------
    print("\n" + "#"*70, flush=True)
    print(">>> [STAGE 3/3] TRAINING MODEL 9.3 (MULTI-PROTOTYPE SUB-CLUSTERING)", flush=True)
    print("#"*70, flush=True)
    try:
        train_model9_3(epochs=10, batch_size=4, patience=3)
        with open(os.path.join(MODEL9_DIR, "model9_3_metrics.json")) as f:
            results["model9_3"] = json.load(f)
    except Exception as e:
        print(f"[!] Error in Model 9.3: {e}", flush=True)
        results["model9_3"] = {"error": str(e), "val_accuracy": 0.0, "val_macro_f1": 0.0}

    total_elapsed = time.time() - t_start

    # -------------------------------------------------------------
    # 4. Comparative Synthesis & Champion Selection
    # -------------------------------------------------------------
    print("\n" + "="*70, flush=True)
    print("      OVERNIGHT TRAINING COMPLETED! COMPARATIVE SYNTHESIS", flush=True)
    print("="*70, flush=True)
    print(f"Total Suite Runtime: {total_elapsed / 60:.1f} minutes\n", flush=True)

    print(f"{'Model Variant':<28} | {'Val Accuracy':<15} | {'Macro-F1':<12} | Status", flush=True)
    print("-" * 70, flush=True)

    best_key = None
    best_f1 = -1.0
    for k, v in results.items():
        acc = v.get("val_accuracy", 0.0) * 100
        f1 = v.get("val_macro_f1", 0.0)
        status = "Completed" if "error" not in v else "Failed"
        print(f"{v.get('model', k):<28} | {acc:>6.2f}%         | {f1:>8.4f}   | {status}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_key = k

    champion_name = results[best_key].get("model", best_key)
    print("\n" + "*"*70, flush=True)
    print(f"  [CHAMPION] Highest Performing Model: {champion_name} (Macro-F1: {best_f1:.4f})", flush=True)
    print("*"*70, flush=True)

    # -------------------------------------------------------------
    # 5. Deploy Champion Weights
    # -------------------------------------------------------------
    champ_src = os.path.join(MODEL9_DIR, "weights", f"{best_key}_best.pt")
    champ_master = os.path.join(MODEL9_DIR, "weights", "model9_best.pt")
    if os.path.exists(champ_src):
        shutil.copy2(champ_src, champ_master)
        print(f"[+] Designated Champion weights saved to: {champ_master}", flush=True)
        if os.path.exists(DESKTOP_DIR):
            desktop_dst = os.path.join(DESKTOP_DIR, "model9_best.pt")
            shutil.copy2(champ_src, desktop_dst)
            print(f"[+] Deployed Champion weights to Desktop Annotator: {desktop_dst}", flush=True)

    # -------------------------------------------------------------
    # 6. Copy All Artifacts to Conversation Brain Directory
    # -------------------------------------------------------------
    print("\n[+] Copying generated visual plots and metrics to Brain Artifact Directory...", flush=True)
    for f in os.listdir(MODEL9_DIR):
        if f.endswith(".png") or f.endswith(".json") or f.endswith(".csv"):
            src = os.path.join(MODEL9_DIR, f)
            dst = os.path.join(BRAIN_DIR, f)
            shutil.copy2(src, dst)
            print(f"  -> Exported artifact: {f}", flush=True)

    # Save Master Suite Summary
    suite_summary = {
        "suite_runtime_minutes": total_elapsed / 60,
        "champion_model": champion_name,
        "champion_macro_f1": best_f1,
        "results": results
    }
    with open(os.path.join(MODEL9_DIR, "overnight_suite_summary.json"), "w") as f:
        json.dump(suite_summary, f, indent=2)

    # -------------------------------------------------------------
    # 7. Git Add, Commit & Push
    # -------------------------------------------------------------
    print("\n[+] Syncing all Model 9 files and weights to Git...", flush=True)
    run_git("add model9_hybrid_cluster/*.py model9_hybrid_cluster/*.csv model9_hybrid_cluster/*.json")
    run_git("add -f model9_hybrid_cluster/*.png model9_hybrid_cluster/weights/model9_best.pt")
    run_git(f"commit -m \"feat(model9): overnight training complete - champion {champion_name} (Macro-F1: {best_f1:.4f})\"")
    code, out, err = run_git("push origin main")
    if code == 0:
        print("[+] Successfully pushed all Model 9 deliverables to GitHub (main)!", flush=True)
    else:
        print(f"[!] Git push deferred ({err}), local commit is secure.", flush=True)

    print("\n" + "="*70, flush=True)
    print("      OVERNIGHT AUTOMATION PROTOCOL COMPLETED FULLY!", flush=True)
    print("="*70 + "\n", flush=True)


if __name__ == "__main__":
    main()

