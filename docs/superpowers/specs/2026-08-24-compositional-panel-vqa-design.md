# Compositional Cross-Panel VQA over Compound Biomedical Figures

Status: approved for implementation planning
Date: 2026-08-24
Target: NeurIPS main track submission, deadline 2026-08-30 (6 days from design approval)

## 1. Motivation / gap being closed

The existing pipeline (Stages 0-8, `05_pipeline_clip/` + `06_pipeline_biomedclip_ablation/`)
builds a captioning + VQA dataset where every question is answered from a
**single panel crop**. The VLM benchmarking code
(`08_vlm_benchmarking/02_stage10_benchmark_vlms.py`) already excludes the
`panel_count` question type from scored metrics with the comment "the VLM
can't see the whole figure" — an implicit, previously unaddressed
acknowledgment that no model in the current pipeline can answer any
question requiring more than one panel's information at once.

This is the gap: **no existing task or model in this project (or, as far
as this design assumes, in comparable VLM benchmarks) requires or tests
reasoning across multiple panels of the same compound figure.** Single-
panel VLMs are structurally incapable of it — they never see sibling
panels. This document proposes closing that gap with (a) a new task,
compositional cross-panel VQA, and (b) a genuinely new lightweight method
architecturally suited to it, benchmarked against the project's existing
single-panel VLM predictions as baselines.

This is the paper's algorithmic novelty claim: a **permutation-aware
panel-set attention head over frozen CLIP/BiomedCLIP embeddings** that
solves cross-panel questions that single-panel VLMs cannot solve by
construction, at a tiny fraction of the parameter count.

## 2. Prerequisite data fix: panel-caption alignment collapse

**Problem (observed, not hypothetical):** Stage 2 (`stage2_clip_alignment.py`)
assigns each panel to its best-matching caption segment via **independent
per-panel argmax** — there is no constraint preventing two different
panels in the same figure from being assigned the identical segment. When
a caption has fewer than 2 panel-letter markers, `split_caption_into_segments`
returns the *entire caption* as the only candidate, so every panel in that
figure is assigned the same text. This is visible directly in the
committed demo figure `figures/5588608__4246_demo.png`, where all 6
panels of one figure receive an identical `best_match_text`.

**Why it matters for this design:** Section 3's task needs panels within
a figure to carry genuinely distinct modality/anatomy tags where the
underlying caption supports it. Left as-is, most figures would trivially
answer "all panels are the same," making same/different and odd-one-out
questions uninformative.

**Fix (data preparation step, not the paper's headline contribution):**
Replace independent per-panel argmax with a **one-to-one constrained
assignment within each figure**, solved via `scipy.optimize.linear_sum_assignment`
(Hungarian algorithm) over the existing cached CLIP/BiomedCLIP similarity
matrix (panels x candidate segments, already computed in
`alignment_v1/embed_cache/` and `alignment_biomedclip_v1/embed_cache/` —
no re-embedding needed):

- Only figures with `n_candidate_segments >= 2` are eligible for
  assignment (rerunning is meaningless when there's truly one segment for
  all panels — those figures are marked `alignment_mode = "shared_caption"`
  and excluded from cross-panel question generation, not miscounted as
  "different").
- For eligible figures, solve the assignment maximizing total similarity
  subject to each panel getting a distinct segment (when
  `n_segments >= n_panels`) or best-effort partial assignment (when
  `n_segments < n_panels`, some panels share by necessity — logged, not
  hidden).
- Panels whose assigned-segment similarity falls below the existing
  `MIN_SIMILARITY` threshold (0.18, from Stage 3) are marked
  `alignment_mode = "unmatched"` rather than forced into a low-confidence
  assignment.
- Output: a new column set (`assigned_segment_text`, `assigned_similarity`,
  `alignment_mode`) added alongside (not replacing) the existing
  `best_match_text` / `best_similarity` from Stage 2/2b, so the original
  greedy alignment remains available for the existing captioning/VQA
  pipeline and comparison stages. This is purely additive.

## 3. Task definition: Compositional Cross-Panel VQA

**Answer space constraint (hard requirement, matches project's existing
no-hallucination philosophy):** every answer must be derivable from
already-extracted categorical facts — the `keyword_lookup` modality/anatomy
tags from Stage 6 (`MODALITY_KEYWORDS`, `ANATOMY_KEYWORDS`), applied to
each panel's `assigned_segment_text` from Section 2 — plus panel reading
order (already computed in `stage1_run_detector_and_crop_panels.py`'s
`order_boxes`). No open-ended or generative cross-panel claims about
pixel content are permitted.

**Question types (scoped to 3 for the 6-day timeline):**

1. **Modality same/different** — "Do panels {X} and {Y} show the same
   imaging modality?" — binary answer, derived by comparing two panels'
   modality tags. Panel pair selection: one same-modality pair and one
   different-modality pair per eligible figure, when both exist.
2. **Odd-one-out (anatomy)** — "Which panel shows a different anatomical
   structure than the others?" — answer is a panel letter (or "none" if
   all panels share one tag). Only generated for figures where exactly
   one panel's anatomy tag differs from a majority of >=2 panels sharing
   another tag (avoids ambiguous ties).
3. **Count-of-modality** — "How many panels show {modality}?" — integer
   answer, derived by counting panels whose modality tag matches.

**Eligibility filter:** only compound figures with **>= 3 accepted panels**
(post Stage 3 quality filter) **and** at least 2 panels with a non-null
modality or anatomy tag (from Section 2's fixed alignment) are eligible.
Figures below this bar are excluded from question generation, not padded
with degenerate questions.

**Output dataset:** `compositional_vqa_dataset.parquet` — one row per
question, columns: `figure_id`, `panel_ids` (list, all panels in the
figure available to the model), `question`, `question_type`, `answer`,
`answer_space` (the fixed candidate set for that question_type, needed
for classification-style scoring), `split` (inherited from the existing
paper-level split — a figure's questions all stay in one split, no
leakage).

## 4. Method: panel-set attention head

**Input representation:** for each figure, the (already cached, frozen)
CLIP or BiomedCLIP embedding of every accepted panel crop, plus a
learned positional embedding keyed on reading-order index (A=0, B=1, …).
Question text is embedded once with the same backbone's text tower
(frozen).

**Architecture:** a small multi-head self-attention block over the set
of (panel embedding + positional embedding) tokens, cross-attended
against the question embedding, followed by a pooling step (attention-
weighted sum) and a classification MLP head over the question type's
fixed `answer_space` (from Section 3). Total trainable parameters:
self-attention block + MLP head only — on the order of low hundreds of
thousands of parameters, not fine-tuning the backbone. Training input is
entirely frozen cached embeddings, so a training epoch is a forward pass
over small tensors, not images — trains in minutes on a single T4.

**Training data split:** train/val/test inherited from the existing
paper-level split (Section 3's dataset), consistent with the project's
existing leakage-safety guarantee.

## 5. Baselines and ablation ladder

1. **Single-panel VLM, no coordination.** Zero new inference — reuse the
   already-saved per-panel `modality`/`anatomy` predictions from
   `08_vlm_benchmarking/.../vqa_predictions/{model}.parquet` (these
   question types already exist in the original Stage 6 VQA dataset and
   already have Qwen2-VL-2B/BLIP-2 predictions on file). For a
   compositional question about panels X and Y, look up each panel's
   already-predicted modality/anatomy tag independently and combine them
   *programmatically* (not by the VLM) to answer same/different, count,
   or odd-one-out. This is the best a single-panel VLM could do with no
   cross-panel coordination mechanism of its own — expected to perform
   poorly whenever the VLM's own per-panel predictions are noisy, since
   errors compound across the panels being compared. Any figure whose
   required panels are missing a saved prediction for a given model is
   excluded from that model's baseline score, not imputed.
2. **Mean-pool ablation.** Average all panel embeddings into one vector
   (no attention, no positional encoding), same MLP head. Isolates
   whether simple pooling already captures cross-panel signal.
3. **Set-attention (proposed method).** Full architecture from Section 4.
4. **Backbone ablation.** Run (2) and (3) with both BiomedCLIP (primary)
   and CLIP (secondary, time-permitting) embeddings — reuses the
   project's existing CLIP-vs-BiomedCLIP comparison framing from Stage 9.

**Metrics:** accuracy per question type; for odd-one-out, exact-match on
the predicted panel letter; for count, exact-match on the integer.
Report a confusion breakdown for the single-panel VLM baseline showing
its systematic failure mode (e.g., always answering "same" because it
has no basis for comparison).

**Statistical honesty:** single seed, single hyperparameter configuration
— explicitly reported as such given the timeline, not presented as a
tuned/repeated-run result.

## 6. Scope cuts for the 6-day timeline

- 3 question types only (Section 3) — no majority-vote or free-text
  question types.
- BiomedCLIP is the primary embedding backbone; CLIP ablation included
  only if time remains after the primary result is done.
- No new VLM inference runs — baseline (1) reuses existing saved
  predictions (Qwen2-VL-2B, BLIP-2) rather than launching new Kaggle/Colab
  jobs.
- Eligible figures require >= 3 accepted panels post quality-filter,
  keeping question generation and eligibility logic simple.
- Single model configuration, single seed — no hyperparameter search.

## 7. Rough schedule (6 days from approval)

1. Half day — Section 2 alignment-collapse fix (Hungarian assignment,
   additive columns).
2. Half day — Section 3 question-template generation off fixed tags.
3. One day — Section 4 set-attention model + training script.
4. Half day — Section 5 baseline wiring against existing VLM predictions.
5. One day — full results run + ablation table.
6. 1.5-2 days — writing.

## 8. Explicitly out of scope

- No fine-tuning of CLIP/BiomedCLIP backbones.
- No new VLM benchmarking runs beyond what's already saved in the repo.
- No open-ended/generative cross-panel questions (hallucination risk).
- No hyperparameter search or multi-seed variance reporting (acknowledged
  limitation given the timeline, to be stated plainly in the paper's
  limitations section).
- No changes to the existing Stage 0-8 / 8b pipeline outputs — Section 2's
  fix is additive (new columns), so existing captioning/VQA/retrieval
  outputs remain reproducible exactly as before.
