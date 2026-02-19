# MatFuse Diffusers Pipeline

A [diffusers](https://github.com/huggingface/diffusers)-compatible implementation of the MatFuse model for generating PBR material maps (diffuse, normal, roughness, specular).

## Components

| File | Class | Description |
|------|-------|-------------|
| `pipeline_matfuse.py` | `MatFusePipeline` | End-to-end generation pipeline (conditioning → denoising → decoding) |
| `vae_matfuse.py` | `MatFuseVQModel` | VQ-VAE with 4 independent encoders/quantizers and a shared decoder |
| `condition_encoders.py` | `MultiConditionEncoder` | Multi-modal conditioning: CLIP image, sentence-transformers text, palette MLP, sketch CNN |

The **UNet** is a standard `diffusers.UNet2DConditionModel` — no custom subclass needed.

## Why Custom Classes?

**`MatFuseVQModel`** — The standard diffusers `VQModel` has a single encoder and quantizer. MatFuse requires 4 independent encoders (one per material map) with 4 separate VQ codebooks feeding into a single shared decoder. This multi-encoder design enables map-level editing and disentangled latent representations.

**`MultiConditionEncoder`** — Manages four conditioning modalities and their fusion into cross-attention tokens and spatial concatenation features. It also handles unconditional embedding generation for classifier-free guidance by encoding placeholder inputs through the actual encoders (matching the model's UCG training behaviour).

## Quick Start

```python
import sys
sys.path.insert(0, "src")

import torch
from diffusers_pipeline.pipeline_matfuse import MatFusePipeline

pipe = MatFusePipeline.from_pretrained("ckpt_pruned")
pipe = pipe.to("cuda")

result = pipe(
    text="red brick wall",
    num_inference_steps=50,
    guidance_scale=4.0,
    generator=torch.Generator("cuda").manual_seed(42),
)

# result keys: "diffuse", "normal", "roughness", "specular"
# Each is a list of PIL Images
for name in ("diffuse", "normal", "roughness", "specular"):
    result[name][0].save(f"{name}.png")
```

## Conditioning Inputs

All conditions are optional and composable. When a condition is omitted the encoder produces a trained placeholder embedding (not a zero tensor), matching how the model was trained with unconditional guidance.

| Input | Type | Encoder | Injection |
|-------|------|---------|-----------|
| `image` | `PIL.Image` or `(B,3,H,W)` tensor | CLIP ViT-B/16 | Cross-attention (token 0) |
| `text` | `str` or `list[str]` | sentence-transformers/clip-ViT-B-16 | Cross-attention (token 1) |
| `palette` | List of RGB tuples or `(B,5,3)` tensor | MLP (3→64→512) | Cross-attention (token 2) |
| `sketch` | `PIL.Image` (grayscale) or `(B,1,H,W)` tensor | CNN (1→4ch, 3× downsample) | Concatenated with latent |

Cross-attention context is always `(B, 3, 512)` — one token per modality (image, text, palette).

## Architecture

### UNet

Standard `UNet2DConditionModel` configured with:

| Parameter | Value |
|-----------|-------|
| `in_channels` | 16 (12 latent + 4 sketch) |
| `out_channels` | 12 (4 maps × 3 channels) |
| `block_out_channels` | [256, 512, 1024] |
| `attention_head_dim` | [8, 16, 32] (= 32 dims/head at every resolution) |
| `cross_attention_dim` | 512 |
| `layers_per_block` | 2 |

### VQ-VAE

| Parameter | Value |
|-----------|-------|
| Resolution | 256 × 256 |
| Downsampling factor | 8 |
| Latent shape | (12, 32, 32) — 4 maps × 3 channels |
| Codebook size | 4 096 per quantizer |
| Embedding dim | 3 |

### Scheduler

`DDIMScheduler` — `beta_start=0.0015`, `beta_end=0.0195`, `scaled_linear`, 1 000 training timesteps, ε-prediction.

## Converting from Original Checkpoint

```bash
python src/scripts/convert_to_diffusers.py \
    --checkpoint checkpoints/matfuse-pruned.ckpt \
    --output ckpt_pruned
```

See the repository root [README](../../README.md) for full details.
