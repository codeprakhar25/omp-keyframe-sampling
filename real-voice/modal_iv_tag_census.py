"""Census all <> tags in IndicVoices Hindi (text only — no audio).

Classifies:
  - emit: tags we already train / emit in cut splitter
  - ignored_vocal: human vocal events we skip today (gasp, throat_clearing, …)
  - noise: ambient / other-speaker / structural markers
  - unknown: anything else

  modal run modal_iv_tag_census.py --action run
  modal run modal_iv_tag_census.py --action pull
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import modal

APP = "real-voice-iv-census"
VOL = "real-voice-iv-cuts"  # reuse; write under /vol/census
ROOT = Path(__file__).resolve().parent
REMOTE = "/vol"

vol = modal.Volume.from_name(VOL, create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=3.0.0",
        "huggingface_hub",
        "pyarrow",
        "tqdm",
    )
)

app = modal.App(APP, image=image)

TAG_RE = re.compile(r"<([^<>]+)>")

# keep in sync with scripts/iv_tags.py
EMIT_RAW = {
    "sigh",
    "whispering",
    "inhaling",
    "breathing",
    "uhh",
    "umm",
    "hmm",
    "laughter",
    "laugh",
    "gasp",
    "throat_clearing",
    "cough",
    "tsk",
    "ugh",
    "sniffle",
}

# still skipped vocal (not locked)
IGNORED_VOCAL = {
    "stammers",
    "singing",
    "smack",
    "whistling",
    "hum",
    "groan",
    "swallowing",
    "sneezing",
    "yawning",
    "snorting",
    "wheezing",
    "nose_blowing",
    "uh-huh",
    "yelling",
    "trill",
    "sneeze",
    "yawn",
    "lip_smack",
    "burp",
    "hiccup",
    "sniff",
    "clear_throat",
    "laughing",
    "cry",
    "crying",
    "sob",
    "humming",
    "whistle",
    "moan",
    "panting",
    "exhale",
    "exhaling",
    "breath",
    "pause",
}

NOISE = {
    "persistent-noise-start",
    "persistent-noise-end",
    "noise",
    "talking",
    "bird_squawk",
    "bird",
    "click",
    "clicking",
    "clink",
    "clinking",
    "clanking",
    "tapping",
    "thumping",
    "pounding",
    "horn",
    "motorcycle",
    "dishes",
    "barking",
    "rustling",
    "squeak",
    "squawking",
    "screeching",
    "beep",
    "bell",
    "static",
    "child",
    "child_talking",
    "child_yelling",
    "unintelligible",
    "hiss",
    "music",
    "tv",
    "radio",
    "door",
    "wind",
    "rain",
    "fan",
    "engine",
    "traffic",
}


def classify(tag: str) -> str:
    t = tag.strip().lower()
    if t in EMIT_RAW:
        return "emit"
    if t in IGNORED_VOCAL:
        return "ignored_vocal"
    if t in NOISE:
        return "noise"
    # heuristic: *_start / *_end structural
    if t.endswith("-start") or t.endswith("-end") or t.endswith("_start") or t.endswith("_end"):
        return "noise"
    return "unknown"


@app.function(
    timeout=60 * 90,
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    memory=8192,
    cpu=4,
)
def run_census(config: str = "hindi") -> dict:
    """Text-only: read parquet columns, never decode audio."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download, login

    tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("huggingface")
        or ""
    ).strip()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        try:
            login(token=tok, add_to_git_credential=False)
        except Exception as e:
            print("hf login warn:", e)

    tag_counts: Counter = Counter()
    clip_has: Counter = Counter()
    n_rows = 0
    n_with_any_tag = 0

    api = HfApi(token=tok or None)
    files = [
        f
        for f in api.list_repo_files("ai4bharat/IndicVoices", repo_type="dataset")
        if f.endswith(".parquet") and f"/{config}/" in f.replace("\\", "/")
    ]
    # also accept prefix hindi/
    if not files:
        files = [
            f
            for f in api.list_repo_files("ai4bharat/IndicVoices", repo_type="dataset")
            if f.endswith(".parquet") and f.startswith(f"{config}/")
        ]
    print(f"parquet files for {config}: {len(files)}")
    for i, fpath in enumerate(sorted(files)):
        local = hf_hub_download(
            "ai4bharat/IndicVoices",
            fpath,
            repo_type="dataset",
            token=tok or None,
        )
        pf = pq.ParquetFile(local)
        names = set(pf.schema.names)
        col = None
        for c in ("unsanitized_verbatim", "verbatim", "text", "transcript"):
            if c in names:
                col = c
                break
        if not col:
            print("skip (no text col)", fpath, "cols", sorted(names)[:20])
            continue
        table = pf.read(columns=[col])
        col_data = table.column(col)
        split_n = 0
        for j in range(len(col_data)):
            n_rows += 1
            split_n += 1
            val = col_data[j].as_py()
            text = val if isinstance(val, str) else str(val or "")
            tags = [t.strip().lower() for t in TAG_RE.findall(text)]
            if tags:
                n_with_any_tag += 1
            seen = set()
            for t in tags:
                tag_counts[t] += 1
                seen.add(t)
            for t in seen:
                clip_has[t] += 1
        print(f"[{i+1}/{len(files)}] {fpath} rows={split_n} unique={len(tag_counts)}")

    by_class: dict[str, list] = {
        "emit": [],
        "ignored_vocal": [],
        "noise": [],
        "unknown": [],
    }
    for t, n in tag_counts.most_common():
        cls = classify(t)
        by_class[cls].append(
            {
                "tag": t,
                "n_tokens": n,
                "n_clips": int(clip_has[t]),
                "class": cls,
            }
        )

    out = {
        "dataset": "ai4bharat/IndicVoices",
        "config": config,
        "n_rows": n_rows,
        "n_rows_with_any_tag": n_with_any_tag,
        "n_unique_tags": len(tag_counts),
        "by_class_totals": {
            cls: {
                "n_tag_types": len(rows),
                "n_tokens": sum(r["n_tokens"] for r in rows),
                "n_clip_hits": sum(r["n_clips"] for r in rows),
            }
            for cls, rows in by_class.items()
        },
        "emit": by_class["emit"],
        "ignored_vocal": by_class["ignored_vocal"],
        "noise_top": by_class["noise"][:40],
        "unknown": by_class["unknown"],
        "all_tags": [
            {"tag": t, "n_tokens": n, "n_clips": int(clip_has[t]), "class": classify(t)}
            for t, n in tag_counts.most_common()
        ],
    }

    out_dir = Path(REMOTE, "census")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"iv_{config}_tag_census.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    vol.commit()
    print("wrote", fp)
    print("ignored_vocal:", out["ignored_vocal"])
    print("unknown:", out["unknown"])
    print("totals:", out["by_class_totals"])
    return {
        "n_rows": n_rows,
        "ignored_vocal": out["ignored_vocal"],
        "unknown": out["unknown"],
        "by_class_totals": out["by_class_totals"],
        "path": str(fp),
    }


def _pull() -> None:
    dest = ROOT / "data/scale/iv_hindi_tag_census.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "modal",
        "volume",
        "get",
        VOL,
        "census/iv_hindi_tag_census.json",
        str(dest),
        "--force",
    ]
    # remote filename uses config
    cmd[4] = "census/iv_hindi_tag_census.json"
    # try hindi name first
    for remote in (
        "census/iv_hindi_tag_census.json",
        "census/iv_hindi_tag_census.json",
    ):
        pass
    for remote in ("census/iv_hindi_tag_census.json",):
        c = ["modal", "volume", "get", VOL, remote, str(dest), "--force"]
        print("+", " ".join(c))
        try:
            subprocess.check_call(c)
            print("pulled", dest)
            return
        except subprocess.CalledProcessError:
            continue
    # list census dir
    subprocess.check_call(["modal", "volume", "ls", VOL, "census"])
    raise SystemExit("pull failed — check volume ls above")


@app.local_entrypoint()
def main(action: str = "run", config: str = "hindi"):
    if action == "run":
        print(run_census.remote(config=config))
        # also fetch
        dest = ROOT / f"data/scale/iv_{config}_tag_census.json"
        subprocess.check_call(
            [
                "modal",
                "volume",
                "get",
                VOL,
                f"census/iv_{config}_tag_census.json",
                str(dest),
                "--force",
            ]
        )
        print("pulled", dest)
        doc = json.loads(dest.read_text(encoding="utf-8"))
        print("\n=== IGNORED VOCAL (candidates) ===")
        for r in doc.get("ignored_vocal") or []:
            print(f"  {r['n_tokens']:7d} tokens  {r['n_clips']:7d} clips  <{r['tag']}>")
        print("\n=== UNKNOWN (review) ===")
        for r in doc.get("unknown") or []:
            print(f"  {r['n_tokens']:7d} tokens  {r['n_clips']:7d} clips  <{r['tag']}>")
        print("\n=== EMIT (current) ===")
        for r in doc.get("emit") or []:
            print(f"  {r['n_tokens']:7d} tokens  {r['n_clips']:7d} clips  <{r['tag']}>")
    elif action == "pull":
        dest = ROOT / f"data/scale/iv_{config}_tag_census.json"
        subprocess.check_call(
            [
                "modal",
                "volume",
                "get",
                VOL,
                f"census/iv_{config}_tag_census.json",
                str(dest),
                "--force",
            ]
        )
        print("pulled", dest)
    else:
        raise SystemExit("action=run|pull")
