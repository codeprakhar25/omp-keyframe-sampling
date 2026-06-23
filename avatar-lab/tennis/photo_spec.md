# Photo spec — Flux LoRA dataset (full-body tennis)

Talking-head LoRA needs only the face. This clip is **full-body action** → the
LoRA must learn your **body + proportions + face**, else the swing drifts into
someone else's body.

## Count
15–30 photos. More variety > more count. 20 good ones beats 40 samey ones.

## Composition mix (aim roughly)
- **8–12 full-body** standing, varied angles (front, 3/4, side, back-ish).
  Side views matter most — forehand is shot side-on.
- **4–6 half-body** (waist up), varied angle.
- **4–6 face/head** close, varied expression (LoRA still needs sharp face).
- **2–3 in motion / athletic stance** if possible (arm raised, mid-stride) —
  helps the model place limbs.

## Quality rules
- Sharp, well-lit, no motion blur.
- Vary background + lighting (don't shoot all in one room → LoRA overfits bg).
- One person only (you) in frame. No heavy crops cutting limbs.
- No sunglasses, no face-covering hats.
- Mix wardrobe but include a couple in simple athletic wear (the target look).

## Anti-overfit
- Don't repeat near-identical shots (same pose/bg ×5) → LoRA memorizes that
  exact frame.
- Avoid strong filters / heavy makeup not representative of you.

## Where
Drop into: `avatar-lab/data/photos_body/` (gitignored — PII).
Currently have 3 (face-only) in `data/photos/` → not enough for full-body.

Once dropped, ping me → we prep the LoRA training config + pod plan.
