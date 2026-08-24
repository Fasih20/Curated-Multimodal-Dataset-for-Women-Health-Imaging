# Presentation

The Beamer deck mirrors the paper and uses an existing compound-figure demo. Its numerical results are loaded from the generated tables under `../paper/tables/`, including the direct medical-VLM comparison when `womens_health_medvlm_results/` is present.

Build from this directory with:

```bash
latexmk -pdf slides.tex
```

When the generated table is absent, the deck honestly displays “Results pending.”
