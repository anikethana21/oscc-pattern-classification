"""
run_master_104_109.py
---------------------
Master Autonomous Orchestrator for Model 104.4 and Model 109.4
Execution Pipeline:
  1. Train Model 104.4 (7-Class Gated Tri-Branch with Focal Loss)
  2. Train Model 109.4 (5-Class Multi-Prototype Deep Metric Clustering + Outlier Gate)
  3. Deploy best weights to desktop generator app
  4. Copy plots and metrics to artifact directory
  5. Compile executive summary and update reports
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

LOG_FILE = r"D:\Project\overnight_104_109.log"
APP_WEIGHTS_DIR = r"C:\Users\anike\Desktop\generator with annotator\model4\weights"
ARTIFACT_DIR = r"C:\Users\anike\.gemini\antigravity\brain\40d4b308-7194-4358-b8e7-0f6a076836ae"

def log(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_script(script_path, desc):
    log(f"\n{'='*70}\nSTARTING: {desc}\nScript: {script_path}\n{'='*70}")
    t0 = time.time()
    cmd = [r"D:\Project\venv\Scripts\python.exe", "-u", script_path]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    for line in iter(proc.stdout.readline, ""):
        line_clean = line.rstrip()
        if line_clean:
            log(line_clean)
            
    proc.stdout.close()
    return_code = proc.wait()
    dur = (time.time() - t0) / 60.0
    
    if return_code != 0:
        log(f"ERROR: {desc} exited with return code {return_code} after {dur:.2f} min!")
        return False
    else:
        log(f"SUCCESS: {desc} finished successfully in {dur:.2f} min.")
        return True

def main():
    log("======================================================================")
    log("    STARTING AUTONOMOUS PIPELINE: MODEL 104.4 & MODEL 109.4 (20X)")
    log("======================================================================")
    overall_start = time.time()

    # Step 1: Run Model 104.4
    ok_104 = run_script(r"D:\Project\models\104.4_tri_branch_wsi_7class\train_model104_4.py", "Model 104.4 Training (7-Class Gated Tri-Branch)")

    # Step 2: Run Model 109.4
    ok_109 = run_script(r"D:\Project\models\109.4_metric_clustering_5class\train_model109_4.py", "Model 109.4 Training (5-Class Multi-Prototype Metric Clustering)")

    # Step 3: Deploy weights to Desktop App
    log("\nDeploying model weights to Desktop Generator application...")
    os.makedirs(APP_WEIGHTS_DIR, exist_ok=True)
    
    src_w104 = r"D:\Project\models\104.4_tri_branch_wsi_7class\weights\model104_4_best.pt"
    dst_w104 = os.path.join(APP_WEIGHTS_DIR, "model104_4_best.pt")
    if os.path.exists(src_w104):
        shutil.copy2(src_w104, dst_w104)
        log(f"  Deployed: {dst_w104}")

    src_w109 = r"D:\Project\models\109.4_metric_clustering_5class\weights\model109_4_best.pt"
    dst_w109 = os.path.join(APP_WEIGHTS_DIR, "model109_4_best.pt")
    if os.path.exists(src_w109):
        shutil.copy2(src_w109, dst_w109)
        log(f"  Deployed: {dst_w109}")

    # Step 4: Copy artifacts (plots, metrics) to Brain Artifact Directory
    log("\nCopying evaluation charts and metrics to conversation artifacts...")
    artifacts_to_copy = [
        r"D:\Project\models\104.4_tri_branch_wsi_7class\confusion_matrix_model104_4.png",
        r"D:\Project\models\104.4_tri_branch_wsi_7class\roc_curve_model104_4.png",
        r"D:\Project\models\104.4_tri_branch_wsi_7class\model104_4_metrics.json",
        r"D:\Project\models\109.4_metric_clustering_5class\confusion_matrix_model109_4.png",
        r"D:\Project\models\109.4_metric_clustering_5class\roc_curve_model109_4.png",
        r"D:\Project\models\109.4_metric_clustering_5class\clusters_2d_model109_4.png",
        r"D:\Project\models\109.4_metric_clustering_5class\model109_4_metrics.json",
        r"D:\Project\models\109.4_metric_clustering_5class\outlier_rejection_audit_109_4.csv"
    ]
    for p in artifacts_to_copy:
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(ARTIFACT_DIR, os.path.basename(p)))
            log(f"  Copied artifact: {os.path.basename(p)}")

    total_dur = (time.time() - overall_start) / 60.0
    log(f"\n======================================================================")
    log(f"    ALL TRAINING PIPELINES COMPLETED IN {total_dur:.2f} MINUTES")
    log(f"    Status: Model 104.4 ({'SUCCESS' if ok_104 else 'FAILED'}) | Model 109.4 ({'SUCCESS' if ok_109 else 'FAILED'})")
    log(f"======================================================================\n")

if __name__ == "__main__":
    main()

