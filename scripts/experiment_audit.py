"""Audit split quality and ablation readiness for the clothing-fit project.

This script is intentionally lightweight: it does not train a model. It checks
whether the current data splits support the next experiments we care about:

- leakage checks across train/validation/test
- cold-start feasibility for users and items
- review-label shortcut risk
- input ablation readiness

Example:
    python scripts/experiment_audit.py --data-dir data --results-dir results
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SPLITS = ("train", "val", "test")
FIT_LABELS = ("small", "fit", "large")

USER_ID_CANDIDATES = ("user_id", "userId", "user", "reviewer_id", "reviewerID")
ITEM_ID_CANDIDATES = ("item_id", "itemId", "item", "clothing_id", "product_id")

METADATA_COLUMNS = (
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
    "source",
)

OPTIONAL_HIGH_RISK_COLUMNS = (
    "rating",
)

REVIEW_COLUMNS = (
    "review_text",
    "review_summary",
)

LABEL_CLUES = {
    "small": (
        "too small",
        "runs small",
        "run small",
        "ran small",
        "size up",
        "sized up",
        "snug",
        "tight",
        "tighter",
        "couldn't zip",
        "could not zip",
    ),
    "fit": (
        "true to size",
        "tts",
        "fit perfectly",
        "fits perfectly",
        "fit perfect",
        "perfect fit",
        "fit well",
        "fits well",
    ),
    "large": (
        "too large",
        "too big",
        "runs large",
        "run large",
        "ran large",
        "size down",
        "sized down",
        "loose",
        "baggy",
        "roomy",
        "oversized",
    ),
}


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train.parquet, val.parquet, and test.parquet.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where audit JSON/Markdown outputs will be written.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="Number of clue-matching examples to include per split.",
    )
    return parser.parse_args()


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download/build the parquet splits before running the audit."
        )
    return pd.read_parquet(path)


def read_splits(data_dir: Path) -> SplitFrames:
    return SplitFrames(*(read_split(data_dir, split) for split in SPLITS))


def find_first_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def normalize_text(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pct(part: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return round(float(part) / float(total), 6)


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    label_counts = Counter(str(v) for v in df.get("fit", pd.Series(dtype=str)).dropna())
    source_counts = Counter(str(v) for v in df.get("source", pd.Series(dtype=str)).dropna())
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_counts": {label: int(label_counts.get(label, 0)) for label in FIT_LABELS},
        "label_percentages": {
            label: pct(label_counts.get(label, 0), len(df)) for label in FIT_LABELS
        },
        "source_counts": dict(source_counts),
        "source_percentages": {
            source: pct(count, len(df)) for source, count in source_counts.items()
        },
        "missing_percentages": {
            column: pct(int(df[column].isna().sum()), len(df)) for column in df.columns
        },
    }


def overlap_report(
    frames: dict[str, pd.DataFrame], column: str | None, entity_name: str
) -> dict[str, Any]:
    if column is None:
        return {
            "available": False,
            "entity": entity_name,
            "reason": f"No {entity_name} identifier column found.",
        }

    train_values = set(frames["train"][column].dropna().astype(str))
    report: dict[str, Any] = {
        "available": True,
        "entity": entity_name,
        "column": column,
        "train_unique": len(train_values),
    }

    for split in ("val", "test"):
        split_values = set(frames[split][column].dropna().astype(str))
        overlap = train_values & split_values
        unseen = split_values - train_values
        report[split] = {
            "unique": len(split_values),
            "seen_in_train": len(overlap),
            "unseen_vs_train": len(unseen),
            "seen_rate": pct(len(overlap), len(split_values)),
            "unseen_rate": pct(len(unseen), len(split_values)),
        }
    return report


def review_overlap_report(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    if "review_text" not in frames["train"].columns:
        return {"available": False, "reason": "No review_text column found."}

    train_reviews = {
        normalize_text(value)
        for value in frames["train"]["review_text"]
        if normalize_text(value)
    }
    report: dict[str, Any] = {
        "available": True,
        "train_unique_review_text": len(train_reviews),
    }

    for split in ("val", "test"):
        split_reviews = [
            normalize_text(value)
            for value in frames[split]["review_text"]
            if normalize_text(value)
        ]
        overlap = sum(1 for value in split_reviews if value in train_reviews)
        report[split] = {
            "non_empty_reviews": len(split_reviews),
            "exact_review_text_overlap_count": overlap,
            "exact_review_text_overlap_rate": pct(overlap, len(split_reviews)),
        }
    return report


def detect_label_clues(text: str) -> dict[str, list[str]]:
    normalized = normalize_text(text)
    found: dict[str, list[str]] = {}
    for label, phrases in LABEL_CLUES.items():
        matches = [phrase for phrase in phrases if phrase in normalized]
        if matches:
            found[label] = matches
    return found


def label_clue_report(
    frames: dict[str, pd.DataFrame], sample_rows: int
) -> dict[str, Any]:
    if "review_text" not in frames["train"].columns:
        return {"available": False, "reason": "No review_text column found."}

    report: dict[str, Any] = {"available": True, "clue_phrases": LABEL_CLUES}
    for split, df in frames.items():
        rows_with_any_clue = 0
        rows_with_matching_label_clue = 0
        examples = []
        clue_by_true_label = {label: 0 for label in FIT_LABELS}

        for _, row in df.iterrows():
            clues = detect_label_clues(row.get("review_text", ""))
            true_label = str(row.get("fit", ""))
            if clues:
                rows_with_any_clue += 1
            if true_label in clues:
                rows_with_matching_label_clue += 1
                clue_by_true_label[true_label] += 1
                if len(examples) < sample_rows:
                    examples.append(
                        {
                            "fit": true_label,
                            "matched_phrases": clues[true_label],
                            "review_excerpt": normalize_text(row.get("review_text", ""))[:240],
                        }
                    )

        label_counts = Counter(str(v) for v in df.get("fit", pd.Series(dtype=str)).dropna())
        report[split] = {
            "rows": int(len(df)),
            "rows_with_any_clue": rows_with_any_clue,
            "rows_with_any_clue_rate": pct(rows_with_any_clue, len(df)),
            "rows_with_matching_label_clue": rows_with_matching_label_clue,
            "rows_with_matching_label_clue_rate": pct(
                rows_with_matching_label_clue, len(df)
            ),
            "matching_label_clue_by_class": {
                label: {
                    "count": int(clue_by_true_label[label]),
                    "rate_within_class": pct(clue_by_true_label[label], label_counts.get(label, 0)),
                }
                for label in FIT_LABELS
            },
            "examples": examples,
        }
    return report


def ablation_readiness(columns: set[str]) -> dict[str, Any]:
    metadata_present = [column for column in METADATA_COLUMNS if column in columns]
    metadata_missing = [column for column in METADATA_COLUMNS if column not in columns]
    review_present = [column for column in REVIEW_COLUMNS if column in columns]
    review_missing = [column for column in REVIEW_COLUMNS if column not in columns]
    high_risk_present = [column for column in OPTIONAL_HIGH_RISK_COLUMNS if column in columns]

    return {
        "metadata_only": {
            "ready": bool(metadata_present) and "fit" in columns,
            "present_columns": metadata_present,
            "missing_columns": metadata_missing,
            "note": "Consider excluding rating for pre-purchase prediction, because it is post-outcome feedback.",
        },
        "review_only": {
            "ready": bool(review_present) and "fit" in columns,
            "present_columns": review_present,
            "missing_columns": review_missing,
        },
        "metadata_plus_review": {
            "ready": bool(metadata_present) and bool(review_present) and "fit" in columns,
            "present_columns": metadata_present + review_present,
            "missing_columns": metadata_missing + review_missing,
        },
        "high_risk_columns_present": high_risk_present,
    }


def make_audit(frames: SplitFrames, sample_rows: int) -> dict[str, Any]:
    split_frames = frames.as_dict()
    all_columns = set().union(*(set(df.columns) for df in split_frames.values()))
    train_columns = set(frames.train.columns)

    user_column = find_first_column(all_columns, USER_ID_CANDIDATES)
    item_column = find_first_column(all_columns, ITEM_ID_CANDIDATES)

    return {
        "split_summary": {
            split: summarize_split(df) for split, df in split_frames.items()
        },
        "cold_start_feasibility": {
            "user_id_column": user_column,
            "item_id_column": item_column,
            "can_compute_user_cold_start": user_column is not None,
            "can_compute_item_cold_start": item_column is not None,
            "note": (
                "If user/item IDs are missing, regenerate the parquet splits while preserving "
                "those columns from the raw dataset."
            ),
        },
        "user_overlap": overlap_report(split_frames, user_column, "user"),
        "item_overlap": overlap_report(split_frames, item_column, "item"),
        "review_overlap": review_overlap_report(split_frames),
        "review_label_clues": label_clue_report(split_frames, sample_rows),
        "ablation_readiness": ablation_readiness(train_columns),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    cold = audit["cold_start_feasibility"]
    review = audit["review_overlap"]
    clues = audit["review_label_clues"]
    ablations = audit["ablation_readiness"]

    lines = [
        "# Experiment Audit",
        "",
        "Generated by `scripts/experiment_audit.py`.",
        "",
        "## Split Summary",
        "",
        "| Split | Rows | small | fit | large |",
        "|---|---:|---:|---:|---:|",
    ]

    for split, summary in audit["split_summary"].items():
        counts = summary["label_counts"]
        lines.append(
            f"| {split} | {summary['rows']} | {counts['small']} | {counts['fit']} | {counts['large']} |"
        )

    lines.extend(["", "## Source Mix", "", "| Split | Source | Rows | Percent |", "|---|---|---:|---:|"])
    for split, summary in audit["split_summary"].items():
        for source, count in summary.get("source_counts", {}).items():
            percent = summary.get("source_percentages", {}).get(source, 0.0)
            lines.append(f"| {split} | {source} | {count} | {percent:.2%} |")

    lines.extend(
        [
            "",
            "## Cold-Start Feasibility",
            "",
            f"- User ID column: `{cold['user_id_column']}`",
            f"- Item ID column: `{cold['item_id_column']}`",
            f"- Can compute user cold-start: `{cold['can_compute_user_cold_start']}`",
            f"- Can compute item cold-start: `{cold['can_compute_item_cold_start']}`",
            "",
            "## Review Leakage Signals",
            "",
        ]
    )

    if review.get("available"):
        for split in ("val", "test"):
            split_review = review[split]
            lines.append(
                f"- `{split}` exact review-text overlap with train: "
                f"{split_review['exact_review_text_overlap_count']} "
                f"({split_review['exact_review_text_overlap_rate']:.2%})"
            )
    else:
        lines.append(f"- Review overlap unavailable: {review['reason']}")

    lines.extend(["", "## Review Label Clues", ""])
    if clues.get("available"):
        for split in SPLITS:
            split_clues = clues[split]
            lines.append(
                f"- `{split}` rows with label-matching clue phrases: "
                f"{split_clues['rows_with_matching_label_clue']} "
                f"({split_clues['rows_with_matching_label_clue_rate']:.2%})"
            )
    else:
        lines.append(f"- Label clue analysis unavailable: {clues['reason']}")

    lines.extend(["", "## Ablation Readiness", ""])
    for name in ("metadata_only", "review_only", "metadata_plus_review"):
        item = ablations[name]
        lines.append(f"- `{name}` ready: `{item['ready']}`")

    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            "If user/item IDs are missing, update preprocessing first. If IDs exist, run full",
            "cold-start metrics and then run metadata-only vs review-only ablations.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    missing = [
        args.data_dir / f"{split}.parquet"
        for split in SPLITS
        if not (args.data_dir / f"{split}.parquet").exists()
    ]
    if missing:
        print("Cannot run audit because required split files are missing:")
        for path in missing:
            print(f"  - {path}")
        print()
        print("Build or copy the parquet splits first, then rerun:")
        print(f"  python scripts/experiment_audit.py --data-dir {args.data_dir} --results-dir {args.results_dir}")
        return 2

    frames = read_splits(args.data_dir)
    audit = make_audit(frames, sample_rows=args.sample_rows)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.results_dir / "experiment_audit.json", audit)
    write_markdown(args.results_dir / "experiment_audit.md", audit)

    print(f"Wrote {args.results_dir / 'experiment_audit.json'}")
    print(f"Wrote {args.results_dir / 'experiment_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
