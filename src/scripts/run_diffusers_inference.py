"""
Example inference script for MatFuse using diffusers.

This script demonstrates how to use the MatFuse pipeline to generate PBR material maps.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Generate PBR materials with MatFuse")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the converted diffusers model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text description of the material",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to reference image for style guidance",
    )
    parser.add_argument(
        "--sketch",
        type=str,
        default=None,
        help="Path to sketch/edge map for structure guidance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save generated materials",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width",
    )

    args = parser.parse_args()

    # Import here to allow help without dependencies
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from diffusers import UNet2DConditionModel, DDIMScheduler
    from diffusers_pipeline import (
        MatFusePipeline,
        MatFuseVQModel,
        MultiConditionEncoder,
    )

    print("Loading pipeline...")

    # Resolve model path to absolute path
    model_path = Path(args.model_path).resolve()

    # Load components (use local_files_only to prevent HuggingFace Hub lookups)
    # UNet uses standard diffusers model
    unet = UNet2DConditionModel.from_pretrained(
        model_path / "unet", local_files_only=True
    )
    # VAE uses custom model (4 separate encoders/quantizers)
    vae = MatFuseVQModel.from_pretrained(model_path / "vae", local_files_only=True)
    scheduler = DDIMScheduler.from_pretrained(
        model_path / "scheduler", local_files_only=True
    )
    condition_encoder = MultiConditionEncoder.from_pretrained(
        model_path / "condition_encoder", local_files_only=True
    )

    # Create pipeline
    pipe = MatFusePipeline(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        condition_encoder=condition_encoder,
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # Load reference image if provided
    image = None
    if args.image:
        print(f"Loading reference image from {args.image}")
        image = Image.open(args.image).convert("RGB")

    # Load sketch if provided
    sketch = None
    if args.sketch:
        print(f"Loading sketch from {args.sketch}")
        sketch = Image.open(args.sketch).convert("L")

    print("Generating materials...")

    # Generate materials
    output = pipe(
        image=image,
        text=args.prompt,
        sketch=sketch,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each material map
    for i, diffuse in enumerate(output["diffuse"]):
        diffuse.save(output_dir / f"diffuse_{i}.png")

    for i, normal in enumerate(output["normal"]):
        normal.save(output_dir / f"normal_{i}.png")

    for i, roughness in enumerate(output["roughness"]):
        roughness.save(output_dir / f"roughness_{i}.png")

    for i, specular in enumerate(output["specular"]):
        specular.save(output_dir / f"specular_{i}.png")

    print(f"Materials saved to {output_dir}")


if __name__ == "__main__":
    main()
