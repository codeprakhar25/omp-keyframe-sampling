#!/usr/bin/env python3
"""Stock flow-shaped TTS stacks on the bad / everyday references (user pick 2026-09-29): does a stack whose acoustics
leave the AR model still copy the reference's timing (dead air, stretch) the way our Air fine-tunes do?

Three arms, same refs / lines / seeds as fxen_* (Svarah) and fxpk_* (user laptop-mic):
  cz  Fun-CosyVoice3-0.5B inference_zero_shot     ref text + ref speech tokens IN the LM prompt (ICL, Air-like)
  cx  Fun-CosyVoice3-0.5B inference_cross_lingual ref removed from the LM; ref reaches only the flow (voice outside AR)
  ix  IndexTTS-2 infer                             ref enters the AR through a conformer-perceiver (pooled, not a code prefix)
cz vs cx is the within-model A/B on where the ref enters. Each system runs its own repo + venv, untouched, official
sampling defaults, reseeded per item.

  aws_flowstack_probe.py --system cosyvoice3 --modes cz,cx --items items_fxen.json --refs svh_1,... --names svh1,...
      --out-prefix fxen --seeds 7,11,23
  -> s3 eval/incumbent_v1/inc_<out-prefix>_<mode>_<name>_s<seed>.json + inc_.../wav/  (incumbent layout, so
     aws_whisper_wer.py / aws_fx_sil.py / aws_fxen_sim.py / aws_word_gaps.py score it unchanged)

svh_* ref wavs are NeuCodec decodes of the codes the Air arms were prompted with; these stacks get that same audio.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

BUCKET = os.environ.get("RV_BUCKET", "real-voice-mio-692707725608")
REGION = os.environ.get("RV_REGION", "us-east-1")
WORK = Path(os.environ.get("RV_WORK", "/opt/rv"))
UV = "/usr/local/bin/uv"

# Shared runner: argv = work refs items langs seeds names out_prefix modes
COMMON = r'''
import json, sys, time, os, random, hashlib
import numpy as np, soundfile as sf, torch
work, refs, items_path, langs = sys.argv[1], sys.argv[2].split(","), sys.argv[3], sys.argv[4].split(",")
seeds, names, out_prefix, modes = [int(x) for x in sys.argv[5].split(",")], sys.argv[6].split(","), sys.argv[7], sys.argv[8].split(",")
items = [it for it in json.load(open(items_path, encoding="utf-8")) if it.get("lang", "hi") in langs]

def reseed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def run_all(system, gen):
    for ref, short in zip(refs, names):
        meta = json.load(open(f"{work}/ref/{ref}.json", encoding="utf-8"))
        ref_wav, ref_text = f"{work}/ref/{ref}.wav", meta["text"].strip()
        for mode in modes:
            for seed in seeds:
                name = f"{out_prefix}_{mode}_{short}_s{seed}"
                out = f"{work}/out/{name}"
                if os.path.exists(f"{out}/inc.json"):
                    continue
                os.makedirs(f"{out}/wav", exist_ok=True)
                rows = []
                for it in items:
                    reseed(seed)
                    t0 = time.time()
                    wav, sr = gen(mode, it["text"], ref_wav, ref_text, f"{out}/wav/{it['id']}.wav")
                    gen_s = time.time() - t0
                    dur = len(wav) / sr
                    rows.append({**it, "file": f"wav/{it['id']}.wav", "system": name, "duration_s": round(dur, 2),
                                 "gen_s": round(gen_s, 3)})
                    print(f"  {name} {it['id']} {dur:.2f}s gen {gen_s:.1f}s md5 {hashlib.md5(np.asarray(wav).tobytes()).hexdigest()[:8]}", flush=True)
                json.dump({"pipeline": system, "mode": mode, "ref": ref, "seed": seed, "ref_text": ref_text, "items": rows},
                          open(f"{out}/inc.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"FLOWSTACK_DONE {name}", flush=True)
'''

COSY = COMMON + r'''
sys.path.insert(0, os.environ["COSY_REPO"]); sys.path.insert(0, os.environ["COSY_REPO"] + "/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel
cv = AutoModel(model_dir=os.environ["COSY_MODEL"])
print("model", type(cv).__name__, "sr", cv.sample_rate, flush=True)
SYS = "You are a helpful assistant.<|endofprompt|>"   # CosyVoice3 prompt prefix, as in the repo's example.py

def gen(mode, text, ref_wav, ref_text, path):
    if mode == "cz":    # ref text + ref speech tokens in the LM prompt
        it = cv.inference_zero_shot(text, SYS + ref_text, ref_wav, stream=False)
    elif mode == "cx":  # frontend_cross_lingual deletes prompt_text + llm_prompt_speech_token: ref only in flow
        it = cv.inference_cross_lingual(SYS + text, ref_wav, stream=False)
    else:
        raise SystemExit(f"unknown cosyvoice mode {mode}")
    wav = torch.cat([j["tts_speech"] for j in it], dim=1).squeeze(0).float().cpu().numpy()
    sf.write(path, wav, cv.sample_rate)
    return wav, cv.sample_rate

run_all("Fun-CosyVoice3-0.5B-2512 (official repo)", gen)
'''

INDEX = COMMON + r'''
os.chdir(os.environ["IX_REPO"])
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False,
                use_deepspeed=False, use_qwen_emo=False)

def gen(mode, text, ref_wav, ref_text, path):
    if mode != "ix":
        raise SystemExit(f"unknown indextts mode {mode}")
    tts.infer(spk_audio_prompt=ref_wav, text=text, output_path=path, verbose=False)   # official defaults
    wav, sr = sf.read(path, dtype="float32")
    return wav, sr

run_all("IndexTTS-2 (official repo)", gen)
'''


def _aws(*a):
    subprocess.run(["aws", *a, "--region", REGION, "--only-show-errors"], check=True)


def _run(cmd, **kw):
    print("+", " ".join(map(str, cmd))[:300], flush=True)
    subprocess.run(cmd, check=True, **kw)


def setup_cosyvoice(d: Path) -> dict:
    repo, venv, model = d / "CosyVoice", d / "venv_cosy", d / "Fun-CosyVoice3-0.5B"
    if not repo.exists():
        _run(["git", "clone", "--recursive", "--depth", "1", "https://github.com/FunAudioLLM/CosyVoice.git", str(repo)])
    py = venv / "bin/python"
    if not py.exists():
        _run([UV, "python", "install", "3.10"])
        _run([UV, "venv", "--python", "3.10", str(venv)])
        # The repo's requirements minus what inference never imports (TensorRT, DeepSpeed, serving/UI). The aiinfra
        # onnxruntime-cuda index is dropped: CPU onnxruntime runs the speech tokenizer + campplus on a <=15 s ref.
        drop = ("tensorrt", "deepspeed", "gradio", "fastapi", "uvicorn", "grpcio", "onnxruntime", "--extra-index-url")
        req = [l for l in (repo / "requirements.txt").read_text().splitlines()
               if l.strip() and not l.strip().startswith(drop)]
        # setuptools<70 at build AND runtime: openai-whisper's sdist imports pkg_resources, which current setuptools
        # dropped (caught by a local `uv pip compile` dry resolve 2026-09-29).
        req += ["onnxruntime==1.18.0", "setuptools<70"]
        (d / "cosy_req.txt").write_text("\n".join(req) + "\n")
        (d / "cosy_build.txt").write_text("setuptools<70\n")
        _run([UV, "pip", "install", "--python", str(py), "--index-strategy", "unsafe-best-match",
              "--extra-index-url", "https://download.pytorch.org/whl/cu121", "--build-constraints", str(d / "cosy_build.txt"),
              "-r", str(d / "cosy_req.txt")])
    if not (model / "cosyvoice3.yaml").exists():
        _run([str(py), "-c", "from huggingface_hub import snapshot_download as s; "
              f"s('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='{model}')"])
    return {"py": py, "env": {"COSY_REPO": str(repo), "COSY_MODEL": str(model)}, "inner": COSY}


def setup_indextts(d: Path) -> dict:
    repo = d / "index-tts"
    if not repo.exists():
        _run(["git", "clone", "--depth", "1", "https://github.com/index-tts/index-tts.git", str(repo)],
             env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
    py = repo / ".venv/bin/python"
    if not py.exists():
        _run([UV, "sync"], cwd=repo)   # the repo's own lockfile, no extras (no flash-attn / deepspeed / webui)
    if not (repo / "checkpoints/gpt.pth").exists():
        _run([str(py), "-c", "from huggingface_hub import snapshot_download as s; "
              f"s('IndexTeam/IndexTTS-2', local_dir='{repo / 'checkpoints'}')"])
    return {"py": py, "env": {"IX_REPO": str(repo)}, "inner": INDEX}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["cosyvoice3", "indextts2"])
    ap.add_argument("--modes", required=True, help="cosyvoice3: cz,cx   indextts2: ix")
    ap.add_argument("--prefix", default="eval/incumbent_v1")
    ap.add_argument("--items", required=True)
    ap.add_argument("--langs", default="en")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--names", required=True, help="short name per ref, e.g. svh1,...,ctrl")
    ap.add_argument("--seeds", default="7,11,23")
    ap.add_argument("--out-prefix", required=True, help="runs named <out-prefix>_<mode>_<name>_s<seed>")
    a = ap.parse_args()
    # The box bootstrap exports HF_HUB_ENABLE_HF_TRANSFER=1, but these per-repo venvs have no hf_transfer, so every
    # hub download inside them (model snapshot, IndexTTS2's runtime aux models) died with ValueError. Plain downloads.
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    d = WORK / "flowstack"
    tag = f"{a.system}_{a.out_prefix}"
    (d / tag / "ref").mkdir(parents=True, exist_ok=True)
    _aws("s3", "cp", f"s3://{BUCKET}/{a.prefix}/{a.items}", str(d / tag / "items.json"))
    for r in a.refs.split(","):
        for ext in ("wav", "json"):
            _aws("s3", "cp", f"s3://{BUCKET}/{a.prefix}/ref/{r}.{ext}", str(d / tag / "ref" / f"{r}.{ext}"))
    s = setup_cosyvoice(d) if a.system == "cosyvoice3" else setup_indextts(d)
    (d / tag / "inner.py").write_text(s["inner"])
    env = {**os.environ, **s["env"]}
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    _run([str(s["py"]), str(d / tag / "inner.py"), str(d / tag), a.refs, str(d / tag / "items.json"), a.langs,
          a.seeds, a.names, a.out_prefix, a.modes], env=env)
    for out in sorted((d / tag / "out").iterdir()):
        _aws("s3", "cp", str(out / "inc.json"), f"s3://{BUCKET}/{a.prefix}/inc_{out.name}.json")
        _aws("s3", "sync", str(out / "wav"), f"s3://{BUCKET}/{a.prefix}/inc_{out.name}/wav/")
        print(f"UPLOADED {out.name}", flush=True)


if __name__ == "__main__":
    main()
