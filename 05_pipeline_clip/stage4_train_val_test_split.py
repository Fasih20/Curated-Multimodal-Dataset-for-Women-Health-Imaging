"""
Source: original Colab notebook, cell index [42]
Auto-extracted -- review before treating as final.
"""

"""
Colab: Stage 4 -- leakage-safe train/val/test split.

Preference order (documented, not silent):
  1. Reuse the existing `split` column carried through from
     labeled_split_manifest_v1.parquet -- that split was already computed
     at the pmcid (paper) level upstream, so reusing it is the safest
     option and avoids re-deriving something that already exists.
  2. Only if `split` is missing/unusable, fall back to a fresh
     GroupShuffleSplit keyed on pmcid (paper-level, 80/10/10, seed=42) --
     same tool already used elsewhere in this project.

Panels never cross splits because splitting happens at figure/paper
level and every panel inherits its figure's split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ACCEPTED_PATH = Path("/content/pipeline_data/quality_v1/accepted_dataset.parquet")
WORKLIST_PATH = Path(
    "/content/pipeline_data/compound_worklist_v1/compound_figures_manifest.parquet"
)

OUT_DIR = Path("/content/pipeline_data/splits_final_v1")
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # test = remainder


def fresh_split(pmcids: pd.Series) -> pd.Series:
    df = pd.DataFrame({"pmcid": pmcids})
    gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=SEED)
    train_idx, rest_idx = next(gss1.split(df, groups=df["pmcid"]))
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=VAL_FRAC / (1 - TRAIN_FRAC), random_state=SEED)
    val_rel, test_rel = next(gss2.split(rest, groups=rest["pmcid"]))
    out = pd.Series("test", index=df.index)
    out.iloc[train_idx] = "train"
    out.iloc[rest.index[val_rel]] = "val"
    return out


def main() -> None:
    out_files = {s: OUT_DIR / f"{s}.parquet" for s in ("train", "val", "test")}
    if all(p.exists() for p in out_files.values()):
        print(f"[SKIP] Stage 4 already done -- see {OUT_DIR}")
        return

    accepted = pd.read_parquet(ACCEPTED_PATH)
    # NOTE: earlier version of this script only pulled ["figure_id", "pmcid"]
    # here, so the existing `split` column never actually reached this frame
    # and the fallback always fired. Fixed: pull `split` through too.
    worklist = pd.read_parquet(WORKLIST_PATH)[["figure_id", "pmcid", "split"]]
    df = accepted.merge(worklist, on="figure_id", how="left")

    if "split" in df.columns and df["split"].notna().mean() > 0.95:
        print("Using existing paper-level `split` column (reused, not recomputed).")
    else:
        print("[FALLBACK] existing split column missing/unusable -- computing fresh "
              "paper-level GroupShuffleSplit (seed=42).")
        df["split"] = fresh_split(df["pmcid"].fillna(df["figure_id"]))

    # Leakage check: no pmcid should appear in more than one split.
    leak = df.groupby("pmcid")["split"].nunique()
    n_leaking = int((leak > 1).sum())
    if n_leaking:
        print(f"[WARN] {n_leaking} papers span multiple splits -- forcing each to its "
              f"majority split to remove leakage.")
        majority = df.groupby("pmcid")["split"].agg(lambda s: s.value_counts().idxmax())
        df["split"] = df["pmcid"].map(majority)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for split_name, path in out_files.items():
        sub = df[df["split"] == split_name].reset_index(drop=True)
        sub.to_parquet(path, index=False)
        sub.to_csv(path.with_suffix(".csv"), index=False)
        report[split_name] = {
            "n_panels": int(len(sub)),
            "n_figures": int(sub["figure_id"].nunique()),
            "n_papers": int(sub["pmcid"].nunique()) if "pmcid" in sub else None,
        }

    with open(OUT_DIR / "split_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n[STAGE 4 COMPLETE]")
    print(json.dumps(report, indent=2))
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()