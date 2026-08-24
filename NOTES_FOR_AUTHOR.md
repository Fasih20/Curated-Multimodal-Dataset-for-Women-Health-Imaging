# Notes for the author

- Stages 1–10 retain notebook-era `/content` and `/kaggle` paths. A future maintenance pass should centralize them in a shared `config.py` or consistently use environment variables. Stage 11 already uses `PIPELINE_ROOT` throughout.
- Stage 11 depends on uncommitted `pipeline_data`: Stage 3 accepted datasets, Stage 2/2b embedding caches, and saved Stage 10 VQA predictions. Missing data means results cannot be generated locally; never replace absent metrics with estimates.
- The constrained alignment output is additive. Figures with one unsplit caption are marked `shared_caption` and their text is not treated as panel-specific evidence during question generation.
- The compositional experiment deliberately uses seed 42, one fixed small configuration, and frozen image/text backbones. Multi-seed uncertainty and hyperparameter search are future work.
- `06_export_paper_assets.py` is the only supported way to populate numerical LaTeX tables. Do not hand-copy numbers into the paper or slides.
- The model is position-aware by reading order. Consequently it is not strictly permutation-invariant: permuting tokens without permuting their reading-order indices changes the representation. “Permutation-aware” is the precise term used in the paper.
