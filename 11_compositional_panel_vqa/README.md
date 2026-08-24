# Stage 11 — Compositional cross-panel VQA

This additive stage builds a categorical VQA task over all accepted panels of a compound figure and trains small heads over frozen CLIP-family embeddings. It never fine-tunes a backbone or launches VLM inference.

Set `PIPELINE_ROOT` to the existing `pipeline_data` directory, then run:

| Step | Script | Purpose |
|---|---|---|
| 11.1 | `01_fix_panel_caption_alignment.py` | One-to-one Hungarian panel/caption assignment |
| 11.2 | `02_generate_compositional_questions.py` | Generate same/different, odd-one-out, and count questions |
| 11.3 | `03_train_panel_set_attention.py` | Cache question embeddings; train mean-pool and proposed set-attention heads |
| 11.4 | `04_baseline_single_panel_vlm_eval.py` | Derive baselines from saved per-panel VLM predictions only |
| 11.5 | `05_run_full_evaluation_and_ablations.py` | Aggregate real metrics and plot the ablation comparison |
| 11.6 | `06_export_paper_assets.py` | Export generated LaTeX tables (not run by `run_pipeline.py`) |
| 11.7 | `07_benchmark_medical_vlms.py` | Direct held-out multi-panel evaluation of MedGemma, Lingshu, and MedVLM-R1 |

For gated MedGemma on Colab, clone this repository in a UI notebook, add an
`HF_TOKEN` secret, and run `%run 11_compositional_panel_vqa/colab_run_medical_vlms_ui.py`.
Predictions checkpoint to `MyDrive/womens_health_medvlm_results` and a ZIP is
downloaded when all three models finish.

For Kaggle dual-T4 execution, import `kaggle_medical_vlm_benchmark.ipynb`,
enable Internet and the `GPU T4 x2` accelerator, and add an `HF_TOKEN` Kaggle
secret. The notebook downloads the Drive archive, runs each model/track in an
isolated resumable process, and writes a results ZIP under `/kaggle/working`.

Example:

```bash
PIPELINE_ROOT=/path/to/pipeline_data python 11_compositional_panel_vqa/01_fix_panel_caption_alignment.py --track both
PIPELINE_ROOT=/path/to/pipeline_data python 11_compositional_panel_vqa/02_generate_compositional_questions.py --track both
PIPELINE_ROOT=/path/to/pipeline_data python 11_compositional_panel_vqa/03_train_panel_set_attention.py --track both --mode both
PIPELINE_ROOT=/path/to/pipeline_data python 11_compositional_panel_vqa/04_baseline_single_panel_vlm_eval.py --track both
PIPELINE_ROOT=/path/to/pipeline_data python 11_compositional_panel_vqa/05_run_full_evaluation_and_ablations.py
python 11_compositional_panel_vqa/06_export_paper_assets.py
```

Run `python -m pytest tests -v` before using real data. All experiments use seed 42 and one fixed configuration; there is no hyperparameter search.
