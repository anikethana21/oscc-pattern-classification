"""
reorganize_repo.py
------------------
Automated Git & Filesystem Reorganization Script for OSCC Pattern Classification Repository.
Cleans up root directory clutter, groups models chronologically into models/, segregates
apps, pipelines, reports, benchmarks, and data manifests.
"""

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(r"D:\Project")

def run_git_mv(src_rel, dst_rel):
    src = REPO_ROOT / src_rel
    dst = REPO_ROOT / dst_rel
    if not src.exists():
        print(f"Skipping non-existent source: {src_rel}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Try git mv first to preserve history
    res = subprocess.run(["git", "mv", str(src), str(dst)], cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[GIT MV] {src_rel} -> {dst_rel}")
    else:
        # Fallback to filesystem move and git add
        shutil.move(str(src), str(dst))
        subprocess.run(["git", "add", str(dst)], cwd=str(REPO_ROOT))
        if src.exists():
            subprocess.run(["git", "rm", "-rf", "--cached", str(src)], cwd=str(REPO_ROOT))
        print(f"[FS MOVE] {src_rel} -> {dst_rel}")

def main():
    print("=== STARTING REPOSITORY REORGANIZATION ===")

    # 1. Models Reorganization
    model_mappings = [
        ("model", "models/01_baseline_resnet"),
        ("second_model", "models/02_dualpath_resnet"),
        ("hybridpath_model", "models/03_hybridpath"),
        ("model4_cascade", "models/04_cascade_series/4.0_cascade_base"),
        ("model4_2_subclassifier", "models/04_cascade_series/4.2_subclassifier"),
        ("model4_3_retrained", "models/04_cascade_series/4.3_tri_branch"),
        ("model4_3_1_retrained", "models/04_cascade_series/4.3.1_focal"),
        ("model4_3_2_retrained", "models/04_cascade_series/4.3.2_champion_40x"),
        ("model4_3_2_1_retrained", "models/04_cascade_series/4.3.2.1_extended"),
        ("model4_export", "models/04_cascade_series/exports"),
        ("model5_loop", "models/05_feedback_loop"),
        ("model6_earlyexit", "models/06_early_exit"),
        ("model7_phikon_cluster", "models/07_phikon_cluster"),
        ("model8_phikon_deep", "models/08_phikon_deep_vit"),
        ("model9_hybrid_cluster", "models/09_metric_clustering"),
        ("model104_4_classifier", "models/104.4_tri_branch_wsi_7class"),
        ("model109_4_cluster", "models/109.4_metric_clustering_5class"),
    ]
    for src, dst in model_mappings:
        run_git_mv(src, dst)

    # 2. Interactive Apps
    run_git_mv("demo_app_104_4.py", "apps/streamlit_wsi_demo/demo_app_104_4.py")
    run_git_mv("model104_4_architecture_diagram.png", "apps/streamlit_wsi_demo/model104_4_architecture_diagram.png")
    run_git_mv("generate_architecture_diagram.py", "apps/streamlit_wsi_demo/generate_architecture_diagram.py")

    # 3. Pipelines & Scripts
    pipeline_mappings = [
        ("prepare_20x_dataset.py", "pipelines/prepare_20x_dataset.py"),
        ("run_master_104_109.py", "pipelines/run_master_104_109.py"),
        ("run_experiments_overnight.py", "pipelines/run_experiments_overnight.py"),
        ("build_manifest.py", "pipelines/build_manifest.py"),
        ("extract_features.py", "pipelines/extract_features.py"),
        ("train.py", "pipelines/train_baseline.py"),
        ("predict.py", "pipelines/predict_baseline.py"),
    ]
    for src, dst in pipeline_mappings:
        run_git_mv(src, dst)

    # 4. Reports & Study Guides
    report_mappings = [
        ("Model_4_3_2_Performance_Report.docx", "reports/docx/Model_4_3_2_Performance_Report.docx"),
        ("OSCC_Models_1_to_9_Comprehensive_Architecture_and_Data_Report.docx", "reports/docx/OSCC_Models_1_to_9_Comprehensive_Architecture_and_Data_Report.docx"),
        ("OSCC_Models_104_and_109_Comprehensive_Report.docx", "reports/docx/OSCC_Models_104_and_109_Comprehensive_Report.docx"),
        ("build_comprehensive_report_docx.py", "reports/docx/build_comprehensive_report_docx.py"),
        ("build_models_104_109_report_docx.py", "reports/docx/build_models_104_109_report_docx.py"),
        
        ("MASTER_ARCHITECTURE_STUDY_GUIDE.md", "reports/markdown/MASTER_ARCHITECTURE_STUDY_GUIDE.md"),
        ("MASTER_STUDY_GUIDE.html", "reports/markdown/MASTER_STUDY_GUIDE.html"),
        ("MODEL_4_3_2_1_EXECUTIVE_SUMMARY_SHEET.md", "reports/markdown/MODEL_4_3_2_1_EXECUTIVE_SUMMARY_SHEET.md"),
        ("MODEL_4_3_2_PERFORMANCE_DOCUMENT.md", "reports/markdown/MODEL_4_3_2_PERFORMANCE_DOCUMENT.md"),
        ("MODEL_COMPARISON.md", "reports/markdown/MODEL_COMPARISON.md"),
        ("OSCC_AI_INTERVIEW_COMPREHENSIVE_GUIDE.md", "reports/markdown/OSCC_AI_INTERVIEW_COMPREHENSIVE_GUIDE.md"),

        ("confusion_matrices.png", "reports/figures/confusion_matrices.png"),
        ("all_models_confusion_matrix.png", "reports/figures/all_models_confusion_matrix.png"),
        ("all_5_models_confusion_matrix.png", "reports/figures/all_5_models_confusion_matrix.png"),
        ("model4_vs_model5_confusion_matrix.png", "reports/figures/model4_vs_model5_confusion_matrix.png"),
    ]
    for src, dst in report_mappings:
        run_git_mv(src, dst)

    # 5. Benchmarks
    benchmark_mappings = [
        ("benchmark_out", "benchmarks/benchmark_out"),
        ("benchmark.py", "benchmarks/benchmark.py"),
        ("plot_all_5_models_confusion_matrix.py", "benchmarks/plot_all_5_models_confusion_matrix.py"),
        ("print_cms.py", "benchmarks/print_cms.py"),
    ]
    for src, dst in benchmark_mappings:
        run_git_mv(src, dst)

    # 6. Data Manifests & CSVs
    data_mappings = [
        ("dataset_20x_master/manifest_104_4.csv", "data_manifests/dataset_20x/manifest_104_4.csv"),
        ("dataset_20x_master/manifest_109_4.csv", "data_manifests/dataset_20x/manifest_109_4.csv"),
        ("manifest.csv", "data_manifests/dataset_40x/manifest.csv"),
        ("tiles.csv", "data_manifests/dataset_40x/tiles.csv"),
        ("embedding_labels.csv", "data_manifests/dataset_40x/embedding_labels.csv"),
        ("predictions_cv.csv", "data_manifests/predictions/predictions_cv.csv"),
        ("outlier_screening_432.csv", "data_manifests/predictions/outlier_screening_432.csv"),
    ]
    for src, dst in data_mappings:
        run_git_mv(src, dst)

    # 7. Utilities & Diagnostics
    util_mappings = [
        ("check_deps.py", "utilities/check_deps.py"),
        ("check_wsi_libs.py", "utilities/check_wsi_libs.py"),
        ("inspect_zips.py", "utilities/inspect_zips.py"),
        ("test_demo_backend.py", "utilities/test_demo_backend.py"),
        ("test_wsi_handler.py", "utilities/test_wsi_handler.py"),
    ]
    for src, dst in util_mappings:
        run_git_mv(src, dst)

    print("\n=== REORGANIZATION COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
