import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
run_experiments_overnight.py
----------------------------
Master automated runner for overnight training of Model 4 and Model 5.

Pipeline execution order:
1. Step 1: Augmentation -> Scale dataset to ~2500 tiles/class
2. Step 2: Model 4 - Stage 1 (Screening classifier for cohesive & no_feature)
3. Step 3: Model 4 - Stage 2 (4-class specialist focused on dispersive & noncohesive)
4. Step 4: Model 4 - Full Cascade Evaluation & Confusion Matrix
5. Step 5: Model 5 - Looping Feedback Network Training
6. Step 6: Model 5 - Evaluation & Confusion Matrix
7. Step 7: Final Comparison & Summary Report
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("D:/Project/overnight_training.log")


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def run_step(step_name, command_args):
    log(f"\n{'='*70}\nSTARTING: {step_name}\nCommand: {' '.join(command_args)}\n{'='*70}")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            command_args,
            cwd="D:/Project",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy()
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)

        proc.wait()
        elapsed = time.time() - t0
        if proc.returncode == 0:
            log(f"SUCCESS: {step_name} completed in {elapsed/60:.1f} minutes.\n")
            return True
        else:
            log(f"ERROR: {step_name} failed with return code {proc.returncode} after {elapsed/60:.1f} minutes.\n")
            return False
    except Exception as e:
        log(f"EXCEPTION in {step_name}: {e}\n")
        return False


def main():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Overnight Training Log started at {datetime.now()} ===\n")

    python_exe = sys.executable

    # 1. Dataset Augmentation (~2500 per class)
    ok = run_step(
        "STEP 1: Data Augmentation (~2,500 tiles per class)",
        [python_exe, "model4_cascade/m4_augment.py", "--target", "2500"]
    )
    if not ok:
        log("Aborting remaining steps due to augmentation failure.")
        return

    # 2. Model 4 - Stage 1 (Screening filter)
    ok = run_step(
        "STEP 2: Model 4 - Train Stage 1 (Screening Filter)",
        [python_exe, "model4_cascade/m4_train_stage1.py", "--epochs", "30", "--batch-size", "8", "--lr", "3e-4", "--patience", "8"]
    )
    if not ok:
        log("Warning: Stage 1 encountered an error, attempting to proceed.")

    # 3. Model 4 - Stage 2 (4-class specialist)
    ok = run_step(
        "STEP 3: Model 4 - Train Stage 2 (Specialist Network)",
        [python_exe, "model4_cascade/m4_train_stage2.py", "--epochs", "30", "--batch-size", "4", "--unfreeze", "3", "--patience", "8"]
    )
    if not ok:
        log("Warning: Stage 2 encountered an error, attempting to proceed.")

    # 4. Model 4 - Cascade Evaluation
    run_step(
        "STEP 4: Model 4 - Evaluate Locked Test Set & Plot Confusion Matrix",
        [python_exe, "model4_cascade/m4_eval_cascade.py", "--threshold", "0.75"]
    )

    # 5. Model 5 - Looping Feedback Network Training
    ok = run_step(
        "STEP 5: Model 5 - Train Recurrent Looping Feedback Network",
        [python_exe, "model5_loop/m5_train.py", "--epochs", "35", "--batch-size", "4", "--unfreeze", "3", "--patience", "8"]
    )
    if not ok:
        log("Warning: Model 5 training encountered an error, attempting to proceed.")

    # 6. Model 5 - Evaluation
    run_step(
        "STEP 6: Model 5 - Evaluate Locked Test Set & Plot Confusion Matrix",
        [python_exe, "model5_loop/m5_eval.py"]
    )

    log("\n" + "="*70)
    log("ALL OVERNIGHT EXPERIMENTS COMPLETED!")
    log("Check confusion matrices in:")
    log("  - D:/Project/model4_cascade/m4_confusion_matrix.png")
    log("  - D:/Project/model5_loop/m5_confusion_matrix.png")
    log("Full output saved in D:/Project/overnight_training.log")
    log("="*70 + "\n")


if __name__ == "__main__":
    main()
