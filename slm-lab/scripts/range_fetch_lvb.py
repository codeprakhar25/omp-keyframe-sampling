#!/usr/bin/env python3
"""Fetch specific videos from the LongVideoBench split-tar WITHOUT streaming all 162GB.

The 31 parts (aa..be) concatenate into one `videos.tar`; parts are exactly PART_SIZE bytes
(last shorter). We walk only the 512-byte tar headers via HTTP range requests (skipping file
bodies entirely), map each wanted member to its (offset, size), then range-fetch just those
members' bytes -> a few GB instead of 162GB. Robust to HF CDN throttling (tiny transfer for
the header walk; only the wanted bodies are large).

Resolves each part's signed CDN URL once (the HF `resolve` endpoint 302s to an expiring xet
URL) and reuses a keep-alive Session so thousands of header reads stay fast.

Env: HF_TOKEN required.
"""
from __future__ import annotations

import argparse
import json
import os
import string
import sys
import time

import requests

BASE = "https://huggingface.co/datasets/longvideobench/LongVideoBench/resolve/main"
PART_SIZE = 5_242_880_000  # exact, verified via HEAD (all parts equal except last)
N_PARTS = 31


def part_name(i: int) -> str:
    return "videos.tar.part." + string.ascii_lowercase[i // 26] + string.ascii_lowercase[i % 26]


def resolve_cdn(session, token):
    """Return list of (signed_url, size) per part, following the 302 once."""
    urls = []
    for i in range(N_PARTS):
        r = session.get(f"{BASE}/{part_name(i)}", headers={"Authorization": f"Bearer {token}"},
                        allow_redirects=False)
        if r.status_code in (301, 302, 307):
            u = r.headers["Location"]
        else:
            u = f"{BASE}/{part_name(i)}"
        # size via HEAD on signed url
        h = session.head(u)
        size = int(h.headers.get("Content-Length", PART_SIZE))
        urls.append((u, size))
    total = sum(s for _, s in urls)
    return urls, total


def read_range(session, urls, global_off, length, token, retries=5):
    """Read [global_off, global_off+length) across parts from signed CDN urls."""
    out = bytearray()
    off, remaining = global_off, length
    while remaining > 0:
        pi = off // PART_SIZE
        po = off % PART_SIZE
        url, psize = urls[pi]
        take = min(remaining, psize - po)
        if take <= 0:
            break
        hdr = {"Range": f"bytes={po}-{po + take - 1}"}
        for attempt in range(retries):
            try:
                r = session.get(url, headers=hdr, timeout=120)
                if r.status_code in (200, 206):
                    break
                # signed url may have expired -> re-resolve
                if r.status_code in (403, 401):
                    urls[pi] = _reresolve(session, pi, token)
                    url, psize = urls[pi]
            except requests.RequestException:
                pass
            time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"range read failed at off={off}")
        chunk = r.content
        if not chunk:
            break
        out += chunk
        off += len(chunk)
        remaining -= len(chunk)
    return bytes(out)


def _reresolve(session, pi, token):
    r = session.get(f"{BASE}/{part_name(pi)}", headers={"Authorization": f"Bearer {token}"},
                    allow_redirects=False)
    u = r.headers["Location"] if r.status_code in (301, 302, 307) else f"{BASE}/{part_name(pi)}"
    h = session.head(u)
    return (u, int(h.headers.get("Content-Length", PART_SIZE)))


def walk_and_fetch(members, out_dir, token):
    os.makedirs(out_dir, exist_ok=True)
    want = {f"videos/{m}" for m in members}
    have = {f"videos/{m}" for m in members if os.path.exists(os.path.join(out_dir, m))}
    todo = want - have
    print(f"want {len(want)} members, {len(have)} already present, fetching {len(todo)}")
    if not todo:
        return
    session = requests.Session()
    urls, total = resolve_cdn(session, token)
    print(f"resolved {len(urls)} signed part urls, total {total/1e9:.1f} GB")

    off = 0
    found = 0
    t0 = time.time()
    nhdr = 0
    while off + 512 <= total and found < len(todo):
        hdr = read_range(session, urls, off, 512, token)
        nhdr += 1
        if len(hdr) < 512 or hdr[0] == 0:
            break  # end-of-archive (zero block)
        name = hdr[0:100].split(b"\x00")[0].decode("utf-8", "replace")
        try:
            size = int(hdr[124:136].strip(b"\x00 ") or b"0", 8)
        except ValueError:
            size = 0
        data_off = off + 512
        if name in todo:
            body = read_range(session, urls, data_off, size, token)
            fn = os.path.basename(name)
            with open(os.path.join(out_dir, fn), "wb") as f:
                f.write(body)
            found += 1
            print(f"[{found}/{len(todo)}] {fn} ({size/1e6:.1f} MB) "
                  f"@hdr#{nhdr} off={data_off} elapsed={time.time()-t0:.0f}s")
        off = data_off + ((size + 511) // 512) * 512
    print(f"done: fetched {found}/{len(todo)} in {time.time()-t0:.0f}s ({nhdr} headers walked)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.lvb.json")
    ap.add_argument("--out-dir", default="data/videos")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set")
    data = json.load(open(args.manifest))
    members = sorted({m["video_file"] for m in data})
    walk_and_fetch(members, args.out_dir, token)


if __name__ == "__main__":
    main()
