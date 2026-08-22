"""
Source: original Colab notebook, cell index [81]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Demo -- for each of a few example compound figures, show:
  1. the original compound figure (before)
  2. YOLO's detected panel boxes drawn on it
  3. the individually extracted panel crops
  4. what CLIP matched each panel to vs. what BiomedCLIP matched it to
     (side by side, so the alignment-quality difference is visible, not
     just stated as a number)

Requires the ORIGINAL compound figure images (source_image column in
panel_manifest.csv), which live under local_download/ -- NOT saved inside
pipeline_data. If those files aren't present in this session, re-copy just
the handful this script selects (it prints their paths first so you can
grab only those, not the whole corpus).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image

PANEL_MANIFEST_PATH = Path("/content/pipeline_data/panels_v1/panel_manifest.csv")
CLIP_ALIGNMENT_PATH = Path("/content/pipeline_data/alignment_v1/alignment_manifest.parquet")
BIOMED_ALIGNMENT_PATH = Path("/content/pipeline_data/alignment_biomedclip_v1/alignment_manifest.parquet")

OUT_DIR = Path("/content/pipeline_data/demo_compound_examples")
N_EXAMPLES = 3
SEED = 42


def pick_example_figures(panels: pd.DataFrame) -> list[str]:
    """Prefer figures with several panels (visually more interesting to show)."""
    counts = panels.groupby("figure_id").size()
    candidates = counts[counts >= 3].index.tolist()
    pool = candidates if candidates else counts.index.tolist()
    import random
    random.seed(SEED)
    return random.sample(pool, min(N_EXAMPLES, len(pool)))


def wrap(text: str, width: int = 55) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width)) if isinstance(text, str) else str(text)


def main() -> None:
    panels = pd.read_csv(PANEL_MANIFEST_PATH)
    clip_align = pd.read_parquet(CLIP_ALIGNMENT_PATH)
    biomed_align = pd.read_parquet(BIOMED_ALIGNMENT_PATH)

    fig_ids = pick_example_figures(panels)
    print("Selected example figure_ids:", fig_ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_sources = []
    for fig_id in fig_ids:
        fig_panels = panels[panels["figure_id"] == fig_id].reset_index(drop=True)
        source_path = Path(fig_panels.iloc[0]["source_image"])
        if not source_path.is_file():
            missing_sources.append(str(source_path))

    if missing_sources:
        print("\n[NEEDED] These original compound-figure files are not present "
              "locally -- re-copy just these from Drive before running further:")
        for p in missing_sources:
            print(" ", p)
        print("(Nothing else needs to be re-copied -- panel crops and embeddings "
              "are already saved in pipeline_data.)\n")
        return

    for fig_id in fig_ids:
        fig_panels = panels[panels["figure_id"] == fig_id].reset_index(drop=True)
        source_path = Path(fig_panels.iloc[0]["source_image"])
        orig_img = Image.open(source_path).convert("RGB")

        n_panels = len(fig_panels)
        fig, axes = plt.subplots(3, max(n_panels, 1), figsize=(4.2 * max(n_panels, 1), 12))
        if n_panels == 1:
            axes = axes.reshape(3, 1)

        # Row 1: original figure (spans all columns conceptually -- just repeat in col 0, blank rest)
        for c in range(n_panels):
            axes[0, c].axis("off")
        axes[0, 0].imshow(orig_img)
        axes[0, 0].set_title(f"BEFORE: original compound figure\n({fig_id})", fontsize=10)

        # Row 1 again but with YOLO boxes drawn (put in last column so both are visible if >1 panel)
        box_col = n_panels - 1 if n_panels > 1 else 0
        axes[0, box_col].imshow(orig_img)
        for _, r in fig_panels.iterrows():
            rect = patches.Rectangle(
                (r["x1"], r["y1"]), r["x2"] - r["x1"], r["y2"] - r["y1"],
                linewidth=2, edgecolor="lime", facecolor="none",
            )
            axes[0, box_col].add_patch(rect)
        axes[0, box_col].set_title("YOLO detections (panel boxes)", fontsize=10)
        axes[0, box_col].axis("off")

        # Row 2: extracted panel crops
        for c, (_, r) in enumerate(fig_panels.iterrows()):
            crop = Image.open(r["crop_path"])
            axes[1, c].imshow(crop)
            axes[1, c].set_title(f"panel: {r['panel_id']}\nconf={r['confidence']:.2f}", fontsize=9)
            axes[1, c].axis("off")

        # Row 3: CLIP vs BiomedCLIP alignment text for each panel
        for c, (_, r) in enumerate(fig_panels.iterrows()):
            panel_id = r["panel_id"]
            clip_row = clip_align[clip_align["panel_id"] == panel_id]
            biomed_row = biomed_align[biomed_align["panel_id"] == panel_id]

            clip_text = wrap(clip_row.iloc[0]["best_match_text"]) if len(clip_row) else "(no match)"
            clip_sim = f"{clip_row.iloc[0]['best_similarity']:.3f}" if len(clip_row) else "-"
            biomed_text = wrap(biomed_row.iloc[0]["best_match_text"]) if len(biomed_row) else "(no match)"
            biomed_sim = f"{biomed_row.iloc[0]['best_similarity']:.3f}" if len(biomed_row) else "-"

            axes[2, c].axis("off")
            axes[2, c].text(
                0, 1,
                f"CLIP (sim={clip_sim}):\n{clip_text}\n\n"
                f"BiomedCLIP (sim={biomed_sim}):\n{biomed_text}",
                fontsize=7.5, va="top", ha="left", wrap=True,
                transform=axes[2, c].transAxes,
            )

        fig.suptitle(f"Compound figure {fig_id}: detection -> extraction -> alignment", fontsize=12)
        fig.tight_layout()
        out_path = OUT_DIR / f"{fig_id}_demo.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    print("\n[DEMO COMPLETE]")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()