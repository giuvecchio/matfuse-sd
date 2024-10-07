import argparse
import random
from contextlib import nullcontext

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.transforms.functional import center_crop, to_tensor

from ldm.data.material_utils import *
from ldm.util import load_model_from_config, visualize_palette

parser = argparse.ArgumentParser(description="MatFuse")
parser.add_argument("--ckpt", type=str, help="Path to the MatFuse model")
parser.add_argument("--config", type=str, help="Path to the MatFuse config")
args = parser.parse_args()


model_config = args.config
model_ckpt = args.ckpt

config = OmegaConf.load(model_config)

model = load_model_from_config(config, model_ckpt)
model = model.cuda()
model.eval()


def process_image(image: Image, img_size: int, device: str):
    if image is None:
        return torch.zeros(3, img_size, img_size, device=device)
    image = image.resize((img_size, img_size))
    return map_transform_func(image, img_size).to(device)


def process_palette(palette_source, image_resolution, device):
    if palette_source is None:
        palette_source = Image.fromarray(
            np.zeros((image_resolution, image_resolution, 3), dtype=np.uint8)
        )

    palette = pylette_extract_colors_mod(
        np.array(palette_source), palette_size=5, sort_mode="frequency"
    )
    palette_colors = [c.rgb for c in palette]

    while len(palette_colors) < 5:  # Padding color palette with last color
        palette_colors.append(palette_colors[-1])
    return torch.tensor(palette_colors, device=device).float() / 255.0


def process_sketch(sketch, image_resolution, device):
    if sketch is None:
        sketch = torch.zeros(1, image_resolution, image_resolution)
    else:
        sketch = cv2.resize(sketch, (image_resolution, image_resolution))
        sketch = to_tensor(sketch)
    sketch = sketch.to(device)
    return sketch


def get_mask(mask, mask_all, latent_shape):
    # We invert zeros and ones for each mask if we use the mask from the sketch since gradio reads them inverted
    if mask_all or mask is None:
        return torch.zeros(latent_shape)
    mask = mask[0].convert("L") # NOTE We take only the first layer of the gradio image editor
    # mask = 1 - to_tensor(mask.resize(latent_shape[1:]))
    mask = np.array(mask.resize(latent_shape[1:]))
    mask[mask > 0] = 1
    mask = torch.tensor(mask)
    mask = mask.repeat(3,1,1)
    return 1 - mask


@torch.no_grad()
def generate(
    control,
    image_resolution,
    num_samples,
    ddim_steps,
    ddim_eta,
    ucg_scale,
    seed=-1,
    x=None,
    mask=None,
    use_ddim=True,
    use_ema_scope=True,
):
    ema_scope = model.ema_scope if use_ema_scope else nullcontext

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    latent_shape = (3, image_resolution // 8, image_resolution // 8)

    unconditional_guidance_label = {
        k: torch.zeros_like(v) for k, v in control.items() if "text" not in k
    }
    unconditional_guidance_label["text"] = [""] * num_samples

    cond = model.get_learned_conditioning(control)
    map_samples = torch.tensor([], device=model.device)

    # Basic sampling
    samples, z_denoise_row = model.sample_log(
        cond=cond,
        batch_size=num_samples,
        ddim=use_ddim,
        ddim_steps=ddim_steps,
        eta=ddim_eta,
        x0=x,
        mask=mask,
        image_size=latent_shape[-1],
    )
    samples = F.pad(samples, (7, 7, 7, 7), mode="circular")
    x_samples = model.decode_first_stage(samples)
    x_samples = center_crop(x_samples, (image_resolution, image_resolution))
    map_samples = torch.cat([map_samples, x_samples], dim=0)

    # Sampling with EMA
    with ema_scope("Sampling"):
        samples, z_denoise_row = model.sample_log(
            cond=cond,
            batch_size=num_samples,
            ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=ddim_eta,
            x0=x,
            mask=mask,
            image_size=latent_shape[-1],
        )
    samples = F.pad(samples, (7, 7, 7, 7), mode="circular")
    x_samples_ema = model.decode_first_stage(samples)
    x_samples_ema = center_crop(x_samples_ema, (image_resolution, image_resolution))
    map_samples = torch.cat([map_samples, x_samples_ema], dim=0)

    # Sampling with classifier-free guidance
    if ucg_scale > 1.0:
        uc = model.get_unconditional_conditioning(
            num_samples, unconditional_guidance_label
        )
        with ema_scope("Sampling with classifier-free guidance"):
            samples_cfg, _ = model.sample_log(
                cond=cond,
                batch_size=num_samples,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta,
                unconditional_guidance_scale=ucg_scale,
                unconditional_conditioning=uc,
                x0=x,
                mask=mask,
                image_size=latent_shape[-1],
                reduce_memory=True,
            )
            samples_cfg = F.pad(samples_cfg, (7, 7, 7, 7), mode="circular")
            x_samples_cfg = model.decode_first_stage(samples_cfg)
            x_samples_cfg = center_crop(
                x_samples_cfg, (image_resolution, image_resolution)
            )
            map_samples = torch.cat([map_samples, x_samples_cfg], dim=0)

    maps = unpack_maps(map_samples)
    maps = make_plot_maps(maps)

    maps = (
        (einops.rearrange(maps, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    sketch = (
        (einops.rearrange(control["sketch"], "b c h w -> b h w c") * 255.0)
        .squeeze(0)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    sketch = einops.repeat(sketch, "h w c -> h w (c rgb)", rgb=3)

    palette = visualize_palette((control["palette"] * 255).cpu().numpy()[0])
    palette = (
        (einops.rearrange(palette, "c h w -> h w c") * 255)
        .squeeze(0)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )
    maps = [m for m in maps]
    results = [sketch, palette, *maps]
    torch.cuda.empty_cache()
    return results


@torch.no_grad()
def run_generation(
    render_emb,
    palette_source,
    sketch,
    prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    seed,
    ddim_eta,
    ucg_scale=1.0,
    use_ema_scope=True,
    use_ddim=True,
):
    control = {}

    control["sketch"] = process_sketch(sketch, image_resolution, model.device)
    control["image_embed"] = process_image(render_emb, image_resolution, model.device)
    control["text"] = prompt
    control["palette"] = process_palette(palette_source, image_resolution, model.device)
    control["image_embed"] = torch.stack(
        [control["image_embed"] for _ in range(num_samples)], dim=0
    )
    control["sketch"] = torch.stack(
        [control["sketch"] for _ in range(num_samples)], dim=0
    )
    control["text"] = [prompt] * num_samples
    control["palette"] = torch.stack(
        [control["palette"] for _ in range(num_samples)], dim=0
    )

    return generate(
        control,
        image_resolution,
        num_samples,
        ddim_steps,
        ddim_eta,
        ucg_scale,
        seed=seed,
        use_ddim=use_ddim,
        use_ema_scope=use_ema_scope,
    )


def run_editing(
    diff,
    norm,
    rough,
    spec,
    mask_diff=False,
    mask_norm=False,
    mask_rough=False,
    mask_spec=False,
    render_emb=None,
    prompt=None,
    palette_source=None,
    image_resolution=256,
    seed=-1,
    num_samples=1,
    ucg_scale=1.0,
    ddim_steps=50,
    ddim_eta=0.0,
    use_ema_scope=True,
    use_ddim=True,
):

    diff_map = process_image(
        diff["background"] if diff is not None else None, image_resolution, model.device
    )
    norm_map = process_image(
        norm["background"] if norm is not None else None, image_resolution, model.device
    )
    rough_map = process_image(
        rough["background"] if rough is not None else None, image_resolution, model.device
    )
    spec_map = process_image(
        spec["background"] if spec is not None else None, image_resolution, model.device
    )

    packed_maps = pack_maps(
        {
            "Diffuse": diff_map,
            "Normal": norm_map,
            "Roughness": rough_map,
            "Specular": spec_map,
        }
    )
    packed_maps = packed_maps.unsqueeze(0).to(model.device)

    x = model.encode_first_stage(packed_maps)
    latent_shape = (3, image_resolution // 8, image_resolution // 8)

    control = {}
    control["sketch"] = torch.zeros(
        num_samples, 1, image_resolution, image_resolution, device=model.device
    )
    control["image_embed"] = process_image(render_emb, image_resolution, model.device)
    control["text"] = [prompt]
    control["palette"] = process_palette(palette_source, image_resolution, model.device)

    control["image_embed"] = torch.stack(
        [control["image_embed"] for _ in range(num_samples)], dim=0
    )
    control["text"] = [prompt] * num_samples
    control["palette"] = torch.stack(
        [control["palette"] for _ in range(num_samples)], dim=0
    )

    mask = pack_maps(
        {
            "Diffuse": get_mask(
                diff["layers"] if diff is not None else None, mask_diff, latent_shape
            ),
            "Normal": get_mask(
                norm["layers"] if norm is not None else None, mask_norm, latent_shape
            ),
            "Roughness": get_mask(
                rough["layers"] if rough is not None else None, mask_rough, latent_shape
            ),
            "Specular": get_mask(
                spec["layers"] if spec is not None else None, mask_spec, latent_shape
            ),
        }
    )
    mask = mask.unsqueeze(0).to(model.device)
    return generate(
        control,
        image_resolution,
        num_samples,
        ddim_steps,
        ddim_eta,
        ucg_scale,
        seed=seed,
        x=x,
        mask=mask,
        use_ddim=use_ddim,
        use_ema_scope=use_ema_scope,
    )
