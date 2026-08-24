# Paper

`neurips_2026.sty` is the unmodified style file downloaded on 2026-08-24 from the official NeurIPS media author kit at `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip` (SHA-256 `C3FC2894E83D2517CA18B66741D6C595986D97957DC08EC08BB2125A7EC4555A`). The submission source is anonymous and does not use the `final` option.

Generate numerical assets first:

```bash
python ../11_compositional_panel_vqa/05_run_full_evaluation_and_ablations.py
python ../11_compositional_panel_vqa/06_export_paper_assets.py
```

Then build from this directory with `latexmk -pdf main.tex` (or `pdflatex`, `bibtex`, then two further `pdflatex` passes). If real metrics are absent, the paper intentionally renders a “Results pending” box rather than fabricated values.
