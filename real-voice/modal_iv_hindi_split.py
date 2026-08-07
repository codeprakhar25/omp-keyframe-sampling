"""Full Hindi IV: gate + caption-split (text only) → proposals + quality stats.

Step 1 of 2 (no WhisperX / no audio here).
Step 2 = modal_iv_cut_slice.py (or follow-on) on survivors.

  modal run modal_iv_hindi_split.py --action run
  modal run modal_iv_hindi_split.py --action pull
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import modal

APP = "real-voice-iv-hindi-split"
VOL = "real-voice-iv-cuts"
ROOT = Path(__file__).resolve().parent
REMOTE = "/vol"

vol = modal.Volume.from_name(VOL, create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub", "pyarrow", "tqdm")
    .add_local_dir(str(ROOT / "scripts"), remote_path="/root/scripts")
)

app = modal.App(APP, image=image)

TAG_RE = re.compile(r"<([^<>]+)>")


def _stable_id(parquet_name: str, row: int) -> str:
    h = hashlib.sha1(f"{parquet_name}:{row}".encode()).hexdigest()[:10]
    return f"hi_{h}"


@app.function(
    timeout=60 * 120,
    volumes={REMOTE: vol},
    secrets=[hf_secret],
    memory=8192,
    cpu=4,
)
def run_split(config: str = "hindi") -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download, login

    sys.path.insert(0, "/root/scripts")
    import iv_caption_split as S
    import iv_tags as T

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

    api = HfApi(token=tok or None)
    files = [
        f
        for f in api.list_repo_files("ai4bharat/IndicVoices", repo_type="dataset")
        if f.endswith(".parquet") and f.startswith(f"{config}/")
    ]
    print(f"parquet files: {len(files)}")

    emit_tags = set(T.EMIT_TAGS)
    gate_raw = set(T.GATE_RAW)
    all_props: list[dict] = []
    by_src: dict[str, list[dict]] = {}
    source_meta: dict[str, dict] = {}
    n_rows = 0
    n_gated = 0
    agg = Counter()
    n_stacks = 0
    gate_tag_clip_hits = Counter()  # which gate tags opened clips

    for fi, fpath in enumerate(sorted(files)):
        local = hf_hub_download(
            "ai4bharat/IndicVoices", fpath, repo_type="dataset", token=tok or None
        )
        pf = pq.ParquetFile(local)
        names = set(pf.schema.names)
        if "unsanitized_verbatim" not in names:
            print("skip no text", fpath)
            continue
        texts = pf.read(columns=["unsanitized_verbatim"]).column(0)
        for row_i in range(len(texts)):
            n_rows += 1
            text = texts[row_i].as_py()
            text = text if isinstance(text, str) else str(text or "")
            # cheap gate prefilter
            raw_tags = {t.strip().lower() for t in TAG_RE.findall(text)}
            if not (raw_tags & gate_raw):
                continue
            n_gated += 1
            for t in raw_tags & gate_raw:
                gate_tag_clip_hits[t] += 1

            cid = _stable_id(Path(fpath).name, row_i)
            clip = {"id": cid, "text": text, "source": "iv", "dataset": "IndicVoices"}
            props, st, stacks = S.split_clip(
                clip,
                emit_tags,
                left_words=8,
                right_words=8,
                max_inhaling=2,
                max_rows=3,
            )
            for k, v in st.items():
                agg[k] += v
            n_stacks += len(stacks)
            if not props:
                continue
            for p in props:
                p["parquet"] = fpath
                p["parquet_row"] = row_i
                p["status"] = "proposed"
            by_src[cid] = props
            all_props.extend(props)
            source_meta[cid] = {
                "id": cid,
                "parquet": fpath,
                "parquet_row": row_i,
                "n_cuts": len(props),
                "text": text,
            }

        print(
            f"[{fi+1}/{len(files)}] {fpath} rows_so_far={n_rows} "
            f"gated={n_gated} props={len(all_props)}"
        )

    # Global per-tag source caps (thinking/breathing/inhaling) — cut off excess
    n_before_cap = len(all_props)
    all_props, cap_stats = T.apply_tag_source_caps(all_props)
    by_src = {}
    for p in all_props:
        by_src.setdefault(p["source_id"], []).append(p)
    # prune source_meta to kept sources
    source_meta = {k: v for k, v in source_meta.items() if k in by_src}
    for sid, props in by_src.items():
        if sid in source_meta:
            source_meta[sid]["n_cuts"] = len(props)
    print(
        f"tag_source_caps: {n_before_cap} → {len(all_props)} rows; stats={cap_stats}"
    )

    by_tag = Counter(p["tag"] for p in all_props)
    by_tag_clips = {
        t: len({p["source_id"] for p in all_props if p["tag"] == t}) for t in by_tag
    }
    # rows per source distribution
    cuts_per_src = Counter(len(v) for v in by_src.values())

    out = {
        "version": S.ALGO_VERSION,
        "algo_version": S.ALGO_VERSION,
        "dataset": "ai4bharat/IndicVoices",
        "config": config,
        "label_source": "indicvoices_verbatim_tags",
        "n_rows_scanned": n_rows,
        "n_gated_clips": n_gated,
        "n_source_clips": len(by_src),
        "n_proposals": len(all_props),
        "mean_cuts_per_kept_source": round(
            (len(all_props) / len(by_src)) if by_src else 0.0, 3
        ),
        "cuts_per_source_hist": dict(sorted(cuts_per_src.items())),
        "gate_raw_tags": sorted(gate_raw),
        "emit_tags": list(T.EMIT_TAGS),
        "gate_tag_clip_hits": dict(gate_tag_clip_hits.most_common()),
        "by_tag": dict(by_tag),
        "by_tag_distinct_clips": by_tag_clips,
        "drop_stats": dict(agg),
        "n_stacks_skipped": n_stacks,
        "n_proposals_before_tag_caps": n_before_cap,
        "tag_source_caps": dict(T.TAG_SOURCE_CAPS),
        "tag_source_cap_stats": cap_stats,
        "tag_lock": {
            "source": "scripts/iv_tags.py",
            "locked_addons": [
                "gasp",
                "throat_clearing",
                "cough",
                "tsk",
                "ugh",
                "sniffle",
            ],
            "gate_raw": sorted(gate_raw),
            "emit_tags": list(T.EMIT_TAGS),
            "tag_source_caps": dict(T.TAG_SOURCE_CAPS),
            "thinking_in_gate": False,
        },
        "quality": {
            "keep_rate_gated_to_source": round(
                (len(by_src) / n_gated) if n_gated else 0.0, 4
            ),
            "props_per_gated": round((len(all_props) / n_gated) if n_gated else 0.0, 4),
            "stack_rate_approx": round((n_stacks / max(n_gated, 1)), 4),
        },
        "note": (
            "Hindi full text split only. Audio slice = next step (WhisperX). "
            "source_meta has parquet+row for audio fetch."
        ),
        "proposals": all_props,
        "by_source_id": by_src,
        "source_meta": source_meta,
    }

    out_dir = Path(REMOTE, "hindi_split")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / "iv_hindi_cut_proposals.json"
    # large file — write
    fp.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    # slim stats sidecar
    slim = {k: out[k] for k in out if k not in ("proposals", "by_source_id", "source_meta")}
    (out_dir / "iv_hindi_split_stats.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    vol.commit()
    print("WROTE", fp)
    print("STATS", json.dumps(slim["quality"], indent=2))
    print("by_tag", dict(by_tag))
    print("drops", dict(agg))
    return slim


@app.local_entrypoint()
def main(action: str = "run", config: str = "hindi"):
    if action == "run":
        slim = run_split.remote(config=config)
        print(json.dumps(slim, indent=2)[:4000])
        dest = ROOT / "data/scale/iv_hindi_split_stats.json"
        subprocess.check_call(
            [
                "modal",
                "volume",
                "get",
                VOL,
                "hindi_split/iv_hindi_split_stats.json",
                str(dest),
                "--force",
            ]
        )
        # also pull full proposals (may be large)
        dest2 = ROOT / "data/scale/iv_hindi_cut_proposals.json"
        print("pulling full proposals (large)…")
        subprocess.check_call(
            [
                "modal",
                "volume",
                "get",
                VOL,
                "hindi_split/iv_hindi_cut_proposals.json",
                str(dest2),
                "--force",
            ]
        )
        print("pulled", dest, dest2)
    elif action == "pull":
        for remote, local in [
            (
                "hindi_split/iv_hindi_split_stats.json",
                ROOT / "data/scale/iv_hindi_split_stats.json",
            ),
            (
                "hindi_split/iv_hindi_cut_proposals.json",
                ROOT / "data/scale/iv_hindi_cut_proposals.json",
            ),
        ]:
            subprocess.check_call(
                ["modal", "volume", "get", VOL, remote, str(local), "--force"]
            )
            print("pulled", local)
    else:
        raise SystemExit("action=run|pull")
