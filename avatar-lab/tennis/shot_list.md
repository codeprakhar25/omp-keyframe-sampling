# Tennis Forehand — shot list / storyboard

**Goal:** ~3–5s clip of ME hitting a forehand (racket + ball) on a grass court.
No speech → echo/LatentSync NOT used here. Identity = Flux LoRA. Motion =
pose-driven video (MimicMotion / AnimateAnyone) from a real driving swing.

**Honest scope:** body motion + racket + ball is HARD for OSS. Ball is the
weak point (not in pose skeleton). Plan = nail the swing + likeness first,
treat the ball as a stretch (composite later if it fails).

---

## Pipeline (revised — NOT the talking-head loop)

```
[full-body + face photos of me]
        |
   (Flux LoRA train)            <- Phase 2, GATED on photos
        |
   keyframe gen (Flux + LoRA)   -> full-body still: me, grass court, racket,
        |                          ready/contact pose, plain athletic wear
   driving clip (real forehand) -> DWPose skeleton extraction
        |
   pose-driven video gen        -> MimicMotion(keyframe + pose) = me swinging
   (MimicMotion / AnimateAnyone)
        |
   [optional] composite ball    -> if model drops it
        |
   compose (ffmpeg) + upscale   -> final.mp4
```

---

## Shot spec

| Field | Value |
|---|---|
| Duration | 3–5 s (one swing: load → contact → follow-through) |
| Res / FPS | gen native (MimicMotion ~576×1024 or 768); 24–30 fps |
| Framing | full body, 3/4 side angle (forehand reads best side-on) |
| Court | grass, baseline, soft daylight |
| Wardrobe | simple tennis kit (solid tee/shorts) — keep LoRA-learnable |
| Camera | static or slow pan; static easiest for pose-driven |

---

## Stage 1 — Flux LoRA (identity)   [GATED on photos, see photo_spec]
Train LoRA on 15–30 photos of me (full-body + face mix). Output: LoRA weights.
Validate: gen a plain portrait → recognizably me before spending on the swing.

## Stage 2 — Keyframe (Flux + LoRA)
Prompt draft:
> full body photo of a man playing tennis, holding a tennis racket, grass
> court, baseline, athletic stance, side view, soft daylight, sharp focus,
> photorealistic, 35mm
Negative:
> blurry, distorted hands, extra fingers, two rackets, deformed, cartoon,
> watermark, low quality
Pick a pose near swing START (load) — pose-driven anim moves it forward.

## Stage 3 — Driving clip (motion source)
Need a real forehand video (you record someone, or stock clip). Side-on,
full body in frame, single clean swing. Extract pose w/ DWPose.
> NOTE: this video supplies ONLY the skeleton motion, not the face/identity.

## Stage 4 — Pose-driven gen (POD)
MimicMotion(keyframe.png + driving_pose) → swing.mp4. ~16–24GB GPU.
Params to tune later. Identity from keyframe, motion from pose.

## Stage 5 — Ball (stretch)
If ball missing/warped: composite a ball sprite along an arc in post (ffmpeg
overlay / simple keyframed path), or accept no ball for v1.

## Stage 6 — Compose (local, $0)
ffmpeg trim + optional GFPGAN (face) + RIFE (smooth) + upscale.

---

## Open items
- [ ] gather 15–30 photos (see photo_spec.md) — BLOCKS Stage 1
- [ ] source a real forehand driving clip (side-on, full body)
- [ ] pick pose-driven model: MimicMotion (rec) vs AnimateAnyone vs UniAnimate
- [ ] confirm pod spend before spin-up (Flux LoRA + MimicMotion = bigger GPU)
- [ ] decide ball: composite vs skip for v1
