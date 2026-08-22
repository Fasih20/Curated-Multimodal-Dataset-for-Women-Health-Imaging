"""
Source: original Colab notebook, cell index [2]
Auto-extracted -- review before treating as final.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive/dataset-women-health-imaging-ml")
LABELED_JSON_PATH = DRIVE_ROOT / "data/41480-labeled-dataset-1777498492376.json"

# OUT_DIR = DRIVE_ROOT / "pipeline_data/splits_pmcid_v1"
OUT_DIR = Path("/content/pipeline_data/splits_pmcid_v1")
SPLIT_MANIFEST_PATH = OUT_DIR / "labeled_split_manifest_v1.parquet"
SPLIT_SUMMARY_PATH = OUT_DIR / "labeled_split_manifest_v1.summary.json"

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10
RANDOM_SEED = 42


def load_labeled_export(path: Path) -> pd.DataFrame:
    """
    Flatten the article-level labeled JSON into one row per figure:
    pmcid, article_url, url, caption, label.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Labeled export not found at {path}. Check DRIVE_ROOT.")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for article in data:
        pmcid = str(article.get("pmcid") or "").strip()
        if not pmcid:
            continue
        article_url = str(article.get("article_url") or "").strip()
        images = article.get("images") or []
        for img in images:
            if not isinstance(img, dict):
                continue
            url = str(img.get("url") or "").strip()
            if not url:
                continue
            rows.append(
                {
                    "pmcid": pmcid,
                    "article_url": article_url,
                    "url": url,
                    "caption": str(img.get("caption") or ""),
                    "label": img.get("label"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No figure rows parsed from {path} -- check the JSON structure.")
    return df


def grouped_split(
    df: pd.DataFrame,
    group_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9
    groups = df[group_col].astype(str).values

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, groups=groups))

    rest_df = df.iloc[rest_idx]
    rest_groups = rest_df[group_col].astype(str).values
    val_rel = val_frac / (val_frac + test_frac)

    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_rel, random_state=seed)
    val_idx_local, test_idx_local = next(gss2.split(rest_df, groups=rest_groups))

    out = df.copy()
    out["split"] = ""
    out.iloc[train_idx, out.columns.get_loc("split")] = "train"
    out.loc[rest_df.index[val_idx_local], "split"] = "val"
    out.loc[rest_df.index[test_idx_local], "split"] = "test"

    assert (out["split"] == "").sum() == 0, "unassigned rows after split"
    return out


def verify_no_leakage(df: pd.DataFrame, group_col: str) -> dict:
    by_split = df.groupby("split")[group_col].apply(lambda s: set(s.astype(str)))
    train_ids = by_split.get("train", set())
    val_ids = by_split.get("val", set())
    test_ids = by_split.get("test", set())
    n_overlap = (
        len(train_ids & val_ids) + len(train_ids & test_ids) + len(val_ids & test_ids)
    )
    return {"n_overlapping_pmcids": n_overlap}


def summarize(df: pd.DataFrame, group_col: str) -> dict:
    rows_by_split = df["split"].value_counts().to_dict()
    papers_by_split = df.groupby("split")[group_col].nunique().to_dict()
    label_counts = df.groupby(["split", "label"], dropna=False).size().to_dict()
    return {
        "total_rows": int(len(df)),
        "total_papers": int(df[group_col].nunique()),
        "rows_by_split": {k: int(v) for k, v in rows_by_split.items()},
        "papers_by_split": {k: int(v) for k, v in papers_by_split.items()},
        "label_counts_by_split": {str(k): int(v) for k, v in label_counts.items()},
        "train_frac_target": TRAIN_FRAC,
        "val_frac_target": VAL_FRAC,
        "test_frac_target": TEST_FRAC,
        "random_seed": RANDOM_SEED,
        "source_json": str(LABELED_JSON_PATH),
    }


def main() -> None:
    print(f"Loading labeled export from {LABELED_JSON_PATH} ...")
    df = load_labeled_export(LABELED_JSON_PATH)
    print(f"Loaded {len(df)} figure rows / {df['pmcid'].nunique()} unique PMCIDs.")
    print(df["label"].value_counts(dropna=False))

    split_df = grouped_split(df, "pmcid", TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED)

    leakage = verify_no_leakage(split_df, "pmcid")
    if leakage["n_overlapping_pmcids"] != 0:
        raise RuntimeError(f"Leakage detected after split: {leakage}")
    print("Leakage check passed: 0 overlapping PMCIDs across splits.")

    summary = summarize(split_df, "pmcid")
    summary["leakage_check"] = leakage
    print(json.dumps(summary, indent=2, default=str))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(SPLIT_MANIFEST_PATH, index=False)
    with open(SPLIT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nWrote split manifest: {SPLIT_MANIFEST_PATH}")
    print(f"Wrote summary: {SPLIT_SUMMARY_PATH}")
    print(
        "\nNext: join this manifest against your downloaded images/captions on "
        "disk (by url or pmcid+filename) to confirm every split row actually "
        "has a local image file before you build any dataloader on top of it."
    )


if __name__ == "__main__":
    main()