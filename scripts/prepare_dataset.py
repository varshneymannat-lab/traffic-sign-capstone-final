from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"image_path", "xmin", "ymin", "xmax", "ymax", "sign_label"}
SPLIT_COLUMNS = ["image_path", "xmin", "ymin", "xmax", "ymax", "sign_label", "split"]


def _stratify_or_none(series: pd.Series, test_size: float) -> pd.Series | None:
    """Use stratification only when each class can appear in both split sides."""
    counts = series.value_counts()
    min_required = 2
    target_rows = int(round(len(series) * test_size))
    if counts.min() < min_required or target_rows < len(counts):
        return None
    return series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/annotations.csv")
    parser.add_argument("--output", default="data/processed/annotations_split.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Annotation file is empty.")
    if (df["xmax"] <= df["xmin"]).any() or (df["ymax"] <= df["ymin"]).any():
        raise ValueError("All bounding boxes must satisfy xmax > xmin and ymax > ymin.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=args.seed,
        stratify=_stratify_or_none(df["sign_label"], 0.30),
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=args.seed,
        stratify=_stratify_or_none(temp_df["sign_label"], 0.50),
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([train_df, val_df, test_df])[SPLIT_COLUMNS].to_csv(output, index=False)
    print(f"Wrote split annotations to {output}")


if __name__ == "__main__":
    main()
