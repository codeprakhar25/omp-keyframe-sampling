# Latent space + diffusion/flow-matching — deep-dive reading path

Ordered visual → mathematical, ending at flow-matching (what FLUX.1 / Wan 2.2 use).

**Mental model to hold throughout:** the VAE squeezes an image into a small
**latent tensor**; **flow-matching** learns a *velocity field* that carries
pure noise → your image *through* that latent space; a **LoRA** bends that field
toward a learned concept (e.g. `prakhar man`).

Latent space = two layers:
- **(a)** the VAE's compressed space — where the image *lives* as a small tensor.
- **(b)** the diffusion/flow *trajectory* through that space.
Learn (a), then (b).

## Tier 0 — visual intuition (start here)
- Diffusion Explainer (interactive): https://poloclub.github.io/diffusion-explainer/
- Jay Alammar, The Illustrated Stable Diffusion: https://jalammar.github.io/illustrated-stable-diffusion/

## Tier 1 — the VAE latent itself (layer a)
- Kingma & Welling, *Auto-Encoding Variational Bayes* — arXiv 1312.6114
- Jeremy Jordan, *Variational autoencoders* (jeremyjordan.me) — readable bridge

## Tier 2 — diffusion math (layer b, discrete)
- Lilian Weng, *What are Diffusion Models?* https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- Calvin Luo, *Understanding Diffusion Models: A Unified Perspective* — arXiv 2208.11970
- Yang Song, *Score-Based Generative Modeling* https://yang-song.net/blog/2021/score/

## Tier 3 — flow matching (what FLUX/Wan actually do)
- Lipman et al, *Flow Matching for Generative Modeling* — arXiv 2210.02747
- Liu et al, *Rectified Flow* — arXiv 2209.03003 (the straight noise→data path = "velocity")
- Meta, *Flow Matching Guide and Code* — arXiv 2412.06264

## Tier 4 — tie it together (latent diffusion)
- Rombach et al, *High-Resolution Image Synthesis with Latent Diffusion Models* — arXiv 2112.10752 (LDM/SD; why diffuse in VAE latent)
- *Flow Matching in Latent Space* — https://vinairesearch.github.io/LFM/

**Order:** Tier 0 (~1hr) → Tier 1 → Tier 2 (Weng + Luo) → Tier 3 → Tier 4.
