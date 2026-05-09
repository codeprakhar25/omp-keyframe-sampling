"""Build cleaned train/validation/test parquet splits from the raw Kaggle JSON files.

Expected raw files:
    data/renttherunway_final_data.json
    data/modcloth_final_data.json

Default outputs:
    data/train.parquet
    data/val.parquet
    data/test.parquet

This script mirrors the preprocessing from `01_eda_baseline.ipynb`, with one
important improvement: it preserves `user_id` and `item_id` so cold-start audits
can measure seen/unseen users and items. The default split strategy is
source-balanced so RentTheRunway and ModCloth both appear in each split.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd


VALID_FITS = {"small", "fit", "large"}

RAW_FILES = {
    "rtr": "renttherunway_final_data.json",
    "mc": "modcloth_final_data.json",
}

KEEP_COLS = [
    "fit",
    "user_id",
    "item_id",
    "height_in",
    "weight_lbs",
    "has_height",
    "has_weight",
    "body_type",
    "age",
    "bust_size",
    "size",
    "size_num",
    "category",
    "review_text",
    "review_summary",
    "review_len",
    "rating",
    "rating_norm",
    "source",
    "review_date",
    "rented_for",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the raw JSON files and receiving parquet outputs.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Oldest fraction assigned to train after date sorting.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Next fraction assigned to validation after date sorting.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("source_balanced", "global_time"),
        default="source_balanced",
        help=(
            "source_balanced splits each source independently before concatenating; "
            "global_time mirrors the original notebook's combined date/order split."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help=(
            "Optional output filename prefix, e.g. 'source_balanced_' writes "
            "source_balanced_train.parquet."
        ),
    )
    return parser.parse_args()


def parse_height(value: Any) -> float | None:
    """Convert values like `5ft 4in`, `5' 4"`, or `64` to inches."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    bare_number = re.fullmatch(r"\d+(\.\d+)?", text)
    if bare_number:
        number = float(text)
        return number if number > 20 else number * 12

    feet_match = re.search(r"(\d+)\s*ft", text) or re.search(r"(\d+)\s*'", text)
    inch_match = re.search(r"(\d+)\s*in", text) or re.search(r"'\s*(\d+)", text)
    feet = int(feet_match.group(1)) if feet_match else 0
    inches = int(inch_match.group(1)) if inch_match else 0
    total = feet * 12 + inches
    return float(total) if total else None


def parse_weight(value: Any) -> float | None:
    """Convert values like `135lbs`, `61kg`, or `150` to pounds."""
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    match = re.search(r"(\d+(\.\d+)?)", text)
    if not match:
        return None

    number = float(match.group(1))
    if "kg" in text:
        return number * 2.20462
    return number


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_", regex=False)
    return df


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create common columns so RTR and ModCloth can be concatenated safely."""
    defaults: dict[str, Any] = {
        "body_type": "unknown",
        "bust_size": "unknown",
        "category": "unknown",
        "review_text": "",
        "review_summary": "",
        "review_date": pd.NA,
        "rented_for": pd.NA,
        "rating": pd.NA,
        "age": pd.NA,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    return df


def preprocess(raw: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = normalize_columns(raw)
    df = add_missing_columns(df)
    df["source"] = source_name

    before = len(df)
    df = df[df["fit"].isin(VALID_FITS)].copy()
    print(f"{source_name}: dropped {before - len(df):,} rows with invalid fit labels")

    df["height_in"] = df["height"].apply(parse_height) if "height" in df.columns else None
    if "weight" in df.columns:
        df["weight_lbs"] = df["weight"].apply(parse_weight)
    else:
        df["weight_lbs"] = None

    df["has_height"] = df["height_in"].notna().astype(int)
    df["has_weight"] = df["weight_lbs"].notna().astype(int)
    df["height_in"] = df["height_in"].fillna(-1)
    df["weight_lbs"] = df["weight_lbs"].fillna(-1)

    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["review_summary"] = df["review_summary"].fillna("").astype(str)
    df["review_len"] = df["review_text"].str.split().str.len()

    if "rating" in df.columns and df["rating"].notna().any():
        df["rating_norm"] = pd.to_numeric(df["rating"], errors="coerce") / 10.0
    elif "quality" in df.columns:
        df["rating"] = df["quality"]
        df["rating_norm"] = pd.to_numeric(df["quality"], errors="coerce") / 5.0
    else:
        df["rating_norm"] = pd.NA

    df["size_num"] = pd.to_numeric(df["size"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["user_id"] = df["user_id"].astype("string") if "user_id" in df.columns else pd.NA
    df["item_id"] = df["item_id"].astype("string") if "item_id" in df.columns else pd.NA

    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    return df


def read_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    missing = []
    for source, filename in RAW_FILES.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(path)
            continue
        frames[source] = pd.read_json(path, lines=True)

    if missing:
        print("Missing raw Kaggle files:")
        for path in missing:
            print(f"  - {path}")
        print()
        print("Download the dataset from Kaggle and place the JSON files under data/.")
        raise SystemExit(2)

    return frames


def split_dataframe(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> dict[str, pd.DataFrame]:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("Expected train_frac and val_frac to be positive and sum to less than 1.")

    if "review_date" in df.columns and df["review_date"].notna().any():
        print("Sorting by review_date before splitting.")
        df = df.sort_values("review_date").reset_index(drop=True)
    else:
        print("No usable review_date found; using raw row order as the split order.")
        df = df.reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    val_end = int(len(df) * (train_frac + val_frac))
    return {
        "train": df.iloc[:train_end].copy(),
        "val": df.iloc[train_end:val_end].copy(),
        "test": df.iloc[val_end:].copy(),
    }


def split_source_balanced(
    frames: dict[str, pd.DataFrame], train_frac: float, val_frac: float
) -> dict[str, pd.DataFrame]:
    by_source = {
        source: split_dataframe(frame, train_frac=train_frac, val_frac=val_frac)
        for source, frame in frames.items()
    }
    splits = {}
    for split in ("train", "val", "test"):
        pieces = [source_splits[split] for source_splits in by_source.values()]
        splits[split] = pd.concat(pieces, ignore_index=True).sample(
            frac=1.0, random_state=42
        ).reset_index(drop=True)
    return splits


def write_splits(
    data_dir: Path, splits: dict[str, pd.DataFrame], output_prefix: str = ""
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        cols = [column for column in KEEP_COLS if column in split.columns]
        path = data_dir / f"{output_prefix}{name}.parquet"
        try:
            split[cols].to_parquet(path, index=False)
        except ImportError as exc:
            print("Could not write parquet files because no parquet engine is installed.")
            print("Install one with:")
            print("  pip install pyarrow")
            raise SystemExit(2) from exc

        dist = split["fit"].value_counts(normalize=True).mul(100).round(1).to_dict()
        sources = split["source"].value_counts(normalize=True).mul(100).round(1).to_dict()
        print(
            f"Saved {path} - {len(split):,} rows, {len(cols)} columns "
            f"| labels={dist} | sources={sources}"
        )


def main() -> int:
    args = parse_args()
    raw = read_raw(args.data_dir)
    cleaned_by_source = {
        source: preprocess(frame, source) for source, frame in raw.items()
    }
    combined = pd.concat(cleaned_by_source.values(), ignore_index=True)

    print(f"\nCombined cleaned rows: {len(combined):,}")
    print("Overall fit distribution:")
    print(combined["fit"].value_counts(normalize=True).mul(100).round(1).to_dict())
    print()

    if args.split_strategy == "source_balanced":
        print("Using source-balanced split strategy.")
        splits = split_source_balanced(
            cleaned_by_source, train_frac=args.train_frac, val_frac=args.val_frac
        )
    else:
        print("Using global time/order split strategy.")
        splits = split_dataframe(
            combined, train_frac=args.train_frac, val_frac=args.val_frac
        )

    write_splits(args.data_dir, splits, output_prefix=args.output_prefix)
    print("\nDone. You can now run:")
    print("  python scripts/experiment_audit.py --data-dir data --results-dir results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
