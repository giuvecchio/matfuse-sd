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
        default=4.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Import here to allow --help without heavy dependencies
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from diffusers_pipeline.pipeline_matfuse import MatFusePipeline

    print("Loading pipeline...")

    pipe = MatFusePipeline.from_pretrained(str(Path(args.model_path).resolve()))

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
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("diffuse", "normal", "roughness", "specular"):
        for i, img in enumerate(output[name]):
            img.save(output_dir / f"{name}_{i}.png")

    print(f"Materials saved to {output_dir}")


if __name__ == "__main__":
    main()
