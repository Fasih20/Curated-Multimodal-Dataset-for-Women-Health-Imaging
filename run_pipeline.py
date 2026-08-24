"""
Master pipeline script.

Runs every pipeline stage in order by executing the numbered .py files as
subprocesses. This is intentionally simple (subprocess, not import) because
several stage scripts still use hardcoded /content/... paths copied from
Colab -- see README.md "before this will actually run" section.

Usage:
    python run_pipeline.py                # run everything
    python run_pipeline.py --from 05       # resume from a folder prefix
    python run_pipeline.py --track clip    # skip the biomedclip ablation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# Ordered list of (label, script_path). Analysis-only scripts (diagnostics
# that don't produce data the next stage depends on) are marked optional.
STAGES: list[tuple[str, str, bool]] = [
    ("01a", "01_data_acquisition/01_inventory_and_split_manifest.py", False),
    ("01b", "01_data_acquisition/02_resolve_pmcid_split_manifest.py", False),
    ("01c", "01_data_acquisition/03_build_working_manifest.py", False),
    ("02a", "02_cleaning_dedup/01_integrity_and_duplicate_detection.py", False),
    ("02b", "02_cleaning_dedup/02_diagnose_duplicate_spike_ANALYSIS.py", True),
    ("02c", "02_cleaning_dedup/03_resolve_exact_duplicates.py", False),
    ("02d", "02_cleaning_dedup/04_flag_compound_images.py", False),
    ("02e", "02_cleaning_dedup/05_compound_flag_precision_check_ANALYSIS.py", True),
    ("02f", "02_cleaning_dedup/06_apply_compound_image_policy.py", False),
    ("03a", "03_medicat_yolo_prep/01_inspect_medicat_annotations.py", False),
    ("03b", "03_medicat_yolo_prep/02_download_medicat_release.py", False),
    ("03c", "03_medicat_yolo_prep/03_visualize_yolo_labels.py", True),
    ("03d", "03_medicat_yolo_prep/04_yolo_dataset_stats_ANALYSIS.py", True),
    ("03e", "03_medicat_yolo_prep/05_visualize_yolo_labels_v2.py", True),
    ("03f", "03_medicat_yolo_prep/06_clean_orphan_labels.py", False),
    ("04a", "04_yolo_training/01_train_yolov8_panel_detector.py", False),
    ("04b", "04_yolo_training/02_qualitative_eval_gt_vs_pred.py", True),
    ("05.0", "05_pipeline_clip/stage0_build_compound_worklist.py", False),
    ("05.1", "05_pipeline_clip/stage1_run_detector_and_crop_panels.py", False),
    ("05.2", "05_pipeline_clip/stage2_clip_alignment.py", False),
    ("05.3", "05_pipeline_clip/stage3_quality_analysis.py", False),
    ("05.4", "05_pipeline_clip/stage4_train_val_test_split.py", False),
    ("05.5", "05_pipeline_clip/stage5_clip_retrieval_baseline.py", False),
    ("05.6", "05_pipeline_clip/stage6_vqa_dataset.py", False),
    ("05.7", "05_pipeline_clip/stage7_captioning_dataset.py", False),
    ("05.8", "05_pipeline_clip/stage8_final_summary_and_methodology.py", False),
    ("06.2b", "06_pipeline_biomedclip_ablation/stage2b_biomedclip_alignment.py", False),
    ("06.3b", "06_pipeline_biomedclip_ablation/stage3b_quality_analysis.py", False),
    ("06.4b", "06_pipeline_biomedclip_ablation/stage4b_split.py", False),
    ("06.5b", "06_pipeline_biomedclip_ablation/stage5b_retrieval_baseline.py", False),
    ("06.6b", "06_pipeline_biomedclip_ablation/stage6b_vqa_dataset.py", False),
    ("06.7b", "06_pipeline_biomedclip_ablation/stage7b_captioning_dataset.py", False),
    ("06.8b", "06_pipeline_biomedclip_ablation/stage8b_final_summary.py", False),
    ("07", "07_comparison/stage9_clip_vs_biomedclip_comparison.py", False),
    ("08a", "08_vlm_benchmarking/02_stage10_benchmark_vlms.py", False),
    ("08b", "08_vlm_benchmarking/03_inspect_vqa_predictions_ANALYSIS.py", True),
    ("08c", "08_vlm_benchmarking/04_inspect_captioning_predictions_ANALYSIS.py", True),
    ("08d", "08_vlm_benchmarking/05_qwen_vs_blip_comparison_ANALYSIS.py", True),
    ("11.1", "11_compositional_panel_vqa/01_fix_panel_caption_alignment.py", False),
    ("11.2", "11_compositional_panel_vqa/02_generate_compositional_questions.py", False),
    ("11.3", "11_compositional_panel_vqa/03_train_panel_set_attention.py", False),
    ("11.4", "11_compositional_panel_vqa/04_baseline_single_panel_vlm_eval.py", False),
    ("11.5", "11_compositional_panel_vqa/05_run_full_evaluation_and_ablations.py", False),
]

# Folder-prefix -> stage labels, used for --track filtering
BIOMEDCLIP_LABELS = {l for l, _, _ in STAGES if l.startswith("06.")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_label", default=None,
                         help="Resume from this stage label, e.g. 05.2")
    parser.add_argument("--skip-optional", action="store_true",
                         help="Skip diagnostic/analysis-only stages")
    parser.add_argument("--track", choices=["both", "clip", "biomedclip"],
                         default="both",
                         help="clip = skip the *b ablation stages; "
                              "biomedclip = skip nothing (ablation depends on"
                              " track being built already)")
    args = parser.parse_args()

    started = args.from_label is None
    for label, rel_path, optional in STAGES:
        if not started:
            if label == args.from_label:
                started = True
            else:
                continue
        if args.skip_optional and optional:
            print(f"[skip-optional] {label} {rel_path}")
            continue
        if args.track == "clip" and label in BIOMEDCLIP_LABELS:
            print(f"[skip-track] {label} {rel_path}")
            continue

        script = REPO_ROOT / rel_path
        print(f"\n=== [{label}] running {rel_path} ===")
        command = [sys.executable, str(script)]
        if label in {"11.1", "11.2", "11.3", "11.4"}:
            command.extend(["--track", args.track])
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Stage {label} ({rel_path}) failed with code "
                  f"{result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
