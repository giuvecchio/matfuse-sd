"""
Checkpoint Conversion Script for MatFuse.

Converts the original MatFuse checkpoint to the diffusers format.
"""

import argparse
import os
from typing import Dict, Any, Tuple

import torch
from omegaconf import OmegaConf

# Add parent directory to path for imports
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_unet_key_mapping() -> Dict[str, str]:
    """
    Get the key mapping from original checkpoint to diffusers format.

    Returns:
        Dictionary mapping original keys to new keys.
    """
    # The key format is similar between the two, but we need to handle some differences
    # Original: model.diffusion_model.{layer_name}
    # New: {layer_name}
    return {}


def convert_unet_to_diffusers(
    original_state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Convert UNet checkpoint from LDM format to diffusers UNet2DConditionModel format.

    This handles the key mapping between the original OpenAI-style UNet naming
    and the diffusers naming convention.

    Args:
        original_state_dict: Original model state dict.
        config: Model configuration.

    Returns:
        Converted state dict compatible with UNet2DConditionModel.
    """
    new_state_dict = {}

    # Extract only diffusion model weights
    prefix = "model.diffusion_model."

    # Key mapping from LDM to diffusers
    # LDM structure:
    #   time_embed -> time_embedding
    #   input_blocks.{i} -> down_blocks.{block_idx}.{layer_type}.{sublayer}
    #   middle_block -> mid_block
    #   output_blocks.{i} -> up_blocks.{block_idx}.{layer_type}.{sublayer}
    #   out -> conv_out (with conv_norm_out)

    unet_conversion_map = {
        # Time embedding
        "time_embed.0.weight": "time_embedding.linear_1.weight",
        "time_embed.0.bias": "time_embedding.linear_1.bias",
        "time_embed.2.weight": "time_embedding.linear_2.weight",
        "time_embed.2.bias": "time_embedding.linear_2.bias",
        # Output
        "out.0.weight": "conv_norm_out.weight",
        "out.0.bias": "conv_norm_out.bias",
        "out.2.weight": "conv_out.weight",
        "out.2.bias": "conv_out.bias",
    }

    # Process all keys
    for key, value in original_state_dict.items():
        if not key.startswith(prefix):
            continue

        new_key = key[len(prefix) :]

        # Direct mapping
        if new_key in unet_conversion_map:
            new_state_dict[unet_conversion_map[new_key]] = value
            continue

        # Input blocks conversion
        if new_key.startswith("input_blocks."):
            new_key = convert_input_block_key(new_key, config)
            if new_key:
                new_state_dict[new_key] = value
            continue

        # Middle block conversion
        if new_key.startswith("middle_block."):
            new_key = convert_middle_block_key(new_key, config)
            if new_key:
                new_state_dict[new_key] = value
            continue

        # Output blocks conversion
        if new_key.startswith("output_blocks."):
            new_key = convert_output_block_key(new_key, config)
            if new_key:
                new_state_dict[new_key] = value
            continue

    return new_state_dict


def convert_input_block_key(key: str, config: Dict[str, Any]) -> str:
    """Convert input_blocks keys to down_blocks format."""
    import re

    # Parse: input_blocks.{block_num}.{rest}
    match = re.match(r"input_blocks\.(\d+)\.(.*)", key)
    if not match:
        return None

    block_num = int(match.group(1))
    rest = match.group(2)

    layers_per_block = config.get("layers_per_block", 2)
    num_down_blocks = len(config.get("block_out_channels", [256, 512, 1024]))

    # Block 0 is the initial conv
    if block_num == 0:
        # input_blocks.0.0.weight -> conv_in.weight (strip the leading '0.')
        if rest.startswith("0."):
            return f"conv_in.{rest[2:]}"
        return f"conv_in.{rest}"

    # Calculate which down_block and which layer within it
    # Layout: [conv_in, (res, res, down?), (res, res, down?), (res, res)]
    block_num -= 1  # Adjust for conv_in

    down_block_idx = block_num // (layers_per_block + 1)
    layer_in_block = block_num % (layers_per_block + 1)

    if down_block_idx >= num_down_blocks:
        return None

    # ResNet blocks
    if layer_in_block < layers_per_block:
        # Check if this is a ResBlock or Attention
        if rest.startswith("0."):
            # ResBlock
            resblock_rest = rest[2:]
            resblock_rest = convert_resblock_subkey(resblock_rest)
            return (
                f"down_blocks.{down_block_idx}.resnets.{layer_in_block}.{resblock_rest}"
            )
        elif rest.startswith("1."):
            # Attention/Transformer block
            attn_rest = rest[2:]
            attn_rest = convert_attention_subkey(attn_rest)
            return (
                f"down_blocks.{down_block_idx}.attentions.{layer_in_block}.{attn_rest}"
            )
        else:
            # Single ResBlock without attention
            resblock_rest = convert_resblock_subkey(rest)
            return (
                f"down_blocks.{down_block_idx}.resnets.{layer_in_block}.{resblock_rest}"
            )
    else:
        # Downsampler - rest is like "0.op.weight" or "0.conv.weight", strip leading "0."
        if rest.startswith("0."):
            rest = rest[2:]
        if "op." in rest:
            downsample_rest = rest.replace("op.", "conv.")
            return f"down_blocks.{down_block_idx}.downsamplers.0.{downsample_rest}"
        else:
            return f"down_blocks.{down_block_idx}.downsamplers.0.{rest}"

    return None


def convert_middle_block_key(key: str, config: Dict[str, Any]) -> str:
    """Convert middle_block keys to mid_block format."""
    import re

    # Parse: middle_block.{block_num}.{rest}
    match = re.match(r"middle_block\.(\d+)\.(.*)", key)
    if not match:
        return None

    block_num = int(match.group(1))
    rest = match.group(2)

    # Structure: [ResBlock, Attention, ResBlock]
    if block_num == 0:
        # First ResBlock
        rest = convert_resblock_subkey(rest)
        return f"mid_block.resnets.0.{rest}"
    elif block_num == 1:
        # Attention
        rest = convert_attention_subkey(rest)
        return f"mid_block.attentions.0.{rest}"
    elif block_num == 2:
        # Second ResBlock
        rest = convert_resblock_subkey(rest)
        return f"mid_block.resnets.1.{rest}"

    return None


def convert_output_block_key(key: str, config: Dict[str, Any]) -> str:
    """Convert output_blocks keys to up_blocks format."""
    import re

    # Parse: output_blocks.{block_num}.{rest}
    match = re.match(r"output_blocks\.(\d+)\.(.*)", key)
    if not match:
        return None

    block_num = int(match.group(1))
    rest = match.group(2)

    layers_per_block = config.get("layers_per_block", 2)
    num_up_blocks = len(config.get("block_out_channels", [256, 512, 1024]))

    # Calculate which up_block and layer
    # Layout: [(res, res, res, up?), (res, res, res, up?), (res, res, res)]
    # Both LDM output_blocks and diffusers up_blocks go from deepest to shallowest
    # output_blocks 0-2 -> up_blocks.0, output_blocks 3-5 -> up_blocks.1, etc.
    up_block_idx = block_num // (layers_per_block + 1)
    layer_in_block = block_num % (layers_per_block + 1)

    if up_block_idx >= num_up_blocks:
        return None

    # Check what type of layer this is
    if rest.startswith("0."):
        # ResBlock
        resblock_rest = rest[2:]
        resblock_rest = convert_resblock_subkey(resblock_rest)
        return f"up_blocks.{up_block_idx}.resnets.{layer_in_block}.{resblock_rest}"
    elif rest.startswith("1."):
        # Could be attention or upsampler
        subrest = rest[2:]
        if "conv" in subrest or "op" in subrest:
            # Upsampler
            if "op." in subrest:
                subrest = subrest.replace("op.", "conv.")
            return f"up_blocks.{up_block_idx}.upsamplers.0.{subrest}"
        else:
            # Attention
            subrest = convert_attention_subkey(subrest)
            return f"up_blocks.{up_block_idx}.attentions.{layer_in_block}.{subrest}"
    elif rest.startswith("2."):
        # Upsampler
        subrest = rest[2:]
        if "conv." in subrest:
            return f"up_blocks.{up_block_idx}.upsamplers.0.{subrest}"
        elif "op." in subrest:
            subrest = subrest.replace("op.", "conv.")
            return f"up_blocks.{up_block_idx}.upsamplers.0.{subrest}"
    else:
        # Direct layer
        resblock_rest = convert_resblock_subkey(rest)
        return f"up_blocks.{up_block_idx}.resnets.{layer_in_block}.{resblock_rest}"

    return None


def convert_resblock_subkey(key: str) -> str:
    """Convert ResBlock internal key names."""
    conversions = {
        "in_layers.0": "norm1",
        "in_layers.2": "conv1",
        "emb_layers.1": "time_emb_proj",
        "out_layers.0": "norm2",
        "out_layers.3": "conv2",
        "skip_connection": "conv_shortcut",
        "nin_shortcut": "conv_shortcut",
    }
    for old, new in conversions.items():
        if old in key:
            return key.replace(old, new)
    return key


def convert_attention_subkey(key: str) -> str:
    """Convert attention/transformer block internal key names."""
    # Most transformer keys map directly
    # In diffusers, attention layer group norm is called 'norm' not 'group_norm'
    # No conversion needed for most keys
    return key


def convert_attention_weights(
    old_key: str, old_value: torch.Tensor, state_dict: Dict[str, torch.Tensor]
):
    """
    Convert attention layer weights.

    The original model uses different naming conventions for attention layers.
    """
    # Map old attention keys to new format
    if "attn1" in old_key or "attn2" in old_key:
        # Handle cross-attention and self-attention
        return {old_key: old_value}
    return {old_key: old_value}


def convert_unet_checkpoint(
    original_state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Convert UNet checkpoint from original format to diffusers format.

    The original LDM model uses a different naming convention that needs to be mapped.

    Args:
        original_state_dict: Original model state dict.
        config: Model configuration.

    Returns:
        Converted state dict.
    """
    new_state_dict = {}

    # Extract only diffusion model weights
    prefix = "model.diffusion_model."

    # Pattern-based transformations
    def transform_key(key: str) -> str:
        """Transform key from original to new format."""
        # Spatial transformer blocks
        # Original: transformer_blocks.0.attn1.to_q.weight
        # Target: transformer_blocks.0.attn1.to_q.weight (same structure)

        # ResBlock normalization
        # Original: in_layers.0.weight (GroupNorm)
        # Target: in_layers.0.weight

        # The structure is mostly compatible, just remove the prefix
        return key

    for key, value in original_state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]

            # Apply key transformation
            new_key = transform_key(new_key)

            # Handle Conv2d -> Linear for attention projections if needed
            # (In our implementation, we keep them as Linear)

            new_state_dict[new_key] = value

    return new_state_dict


def convert_vae_checkpoint(
    original_state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Convert VQ-VAE checkpoint from original format to diffusers format.

    Args:
        original_state_dict: Original model state dict.
        config: Model configuration.

    Returns:
        Converted state dict.
    """
    new_state_dict = {}

    # Extract first stage model weights
    prefix = "first_stage_model."

    for key, value in original_state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]

            # The VQ-VAE structure is similar, just remove the prefix
            new_state_dict[new_key] = value

    return new_state_dict


def convert_condition_encoder_checkpoint(
    original_state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Convert condition encoder checkpoint.

    Args:
        original_state_dict: Original model state dict.
        config: Model configuration.

    Returns:
        Converted state dict.
    """
    new_state_dict = {}

    # Extract cond_stage_model weights
    prefix = "cond_stage_model."

    for key, value in original_state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]

            # Map to new naming
            # sketch_encoder -> sketch_encoder
            # palette_encoder -> palette_encoder
            # The CLIP encoders are loaded separately

            if new_key.startswith("sketch_encoder."):
                new_state_dict[new_key] = value
            elif new_key.startswith("palette_encoder."):
                new_state_dict[new_key] = value

    return new_state_dict


def load_original_checkpoint(
    ckpt_path: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load the original MatFuse checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file.

    Returns:
        Tuple of (state_dict, config).
    """
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint)

    # Try to load config if it's embedded
    config = checkpoint.get("config", None)

    return state_dict, config


def create_diffusers_config(
    original_config: Dict[str, Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Create diffusers configuration from original config.

    Args:
        original_config: Original model configuration.

    Returns:
        Dictionary with configs for each component.
    """
    # UNet config for standard diffusers UNet2DConditionModel
    # Based on matfuse-ldm-vq_f8.yaml:
    # - model_channels: 256, channel_mult: (1, 2, 4)
    # - attention_resolutions: (4, 2, 1) means attention at all levels
    # - context_dim: 512
    # - num_head_channels: 32
    unet_config = {
        "sample_size": 32,
        "in_channels": 16,  # 12 latent + 4 sketch concat
        "out_channels": 12,
        "center_input_sample": False,
        "flip_sin_to_cos": True,
        "freq_shift": 0,
        "down_block_types": (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "up_block_types": (
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        "only_cross_attention": False,
        "block_out_channels": (256, 512, 1024),  # model_channels * channel_mult
        "layers_per_block": 2,  # num_res_blocks
        "downsample_padding": 1,
        "mid_block_scale_factor": 1,
        "act_fn": "silu",
        "norm_num_groups": 32,
        "norm_eps": 1e-5,
        "cross_attention_dim": 512,  # context_dim
        "transformer_layers_per_block": 1,  # transformer_depth
        "attention_head_dim": [8, 16, 32],  # out_channels // num_head_channels per block
        "use_linear_projection": False,
        "upcast_attention": False,
        "resnet_time_scale_shift": "default",
    }

    vae_config = {
        "ch": 128,
        "ch_mult": (1, 1, 2, 4),
        "num_res_blocks": 2,
        "attn_resolutions": (),
        "dropout": 0.0,
        "in_channels": 3,
        "out_channels": 12,
        "resolution": 256,
        "z_channels": 256,
        "n_embed": 4096,
        "embed_dim": 3,
        "scaling_factor": 1.0,
    }

    condition_encoder_config = {
        "sketch_in_channels": 1,
        "sketch_out_channels": 4,
        "palette_in_channels": 3,
        "palette_hidden_channels": 64,
        "palette_out_channels": 512,
        "n_colors": 5,
        "clip_image_model": "ViT-B/16",
        "clip_text_model": "sentence-transformers/clip-ViT-B-16",
    }

    scheduler_config = {
        "num_train_timesteps": 1000,
        "beta_start": 0.0015,
        "beta_end": 0.0195,
        "beta_schedule": "scaled_linear",
        "clip_sample": False,
        "set_alpha_to_one": False,
        "steps_offset": 0,
        "prediction_type": "epsilon",
    }

    return {
        "unet": unet_config,
        "vae": vae_config,
        "condition_encoder": condition_encoder_config,
        "scheduler": scheduler_config,
    }


def convert_checkpoint(
    ckpt_path: str,
    output_dir: str,
    config_path: str = None,
):
    """
    Convert MatFuse checkpoint to diffusers format.

    Uses:
    - Standard UNet2DConditionModel from diffusers
    - Custom MatFuseVQModel (required due to 4 separate encoders/quantizers)

    Args:
        ckpt_path: Path to the original checkpoint.
        output_dir: Directory to save converted models.
        config_path: Path to the original config file (optional).
    """
    from diffusers import UNet2DConditionModel, DDIMScheduler
    from diffusers_pipeline.vae_matfuse import MatFuseVQModel
    from diffusers_pipeline.condition_encoders import MultiConditionEncoder

    # Load original checkpoint
    state_dict, _ = load_original_checkpoint(ckpt_path)

    # Load config if provided
    if config_path:
        original_config = OmegaConf.load(config_path)
    else:
        original_config = None

    # Create diffusers configs
    configs = create_diffusers_config(original_config)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert and save UNet using standard diffusers UNet2DConditionModel
    print("Converting UNet (using diffusers UNet2DConditionModel)...")
    unet = UNet2DConditionModel(**configs["unet"])
    unet_state = convert_unet_to_diffusers(state_dict, configs["unet"])

    # Load state dict with strict=False to handle minor differences
    missing, unexpected = unet.load_state_dict(unet_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")
        if len(unexpected) > 5:
            print(f"    ... and {len(unexpected) - 5} more")

    unet_path = os.path.join(output_dir, "unet")
    unet.save_pretrained(unet_path)
    print(f"  Saved to {unet_path}")

    # Convert and save VAE (custom MatFuseVQModel required)
    print("Converting VAE (using custom MatFuseVQModel)...")
    vae = MatFuseVQModel(**configs["vae"])
    vae_state = convert_vae_checkpoint(state_dict, configs["vae"])

    missing, unexpected = vae.load_state_dict(vae_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")
        if len(unexpected) > 5:
            print(f"    ... and {len(unexpected) - 5} more")

    vae_path = os.path.join(output_dir, "vae")
    vae.save_pretrained(vae_path)
    print(f"  Saved to {vae_path}")

    # Convert and save condition encoder
    print("Converting condition encoder...")
    cond_encoder = MultiConditionEncoder(**configs["condition_encoder"])
    cond_state = convert_condition_encoder_checkpoint(
        state_dict, configs["condition_encoder"]
    )

    # Only load trainable parts (sketch and palette encoders)
    if cond_state:
        missing, unexpected = cond_encoder.load_state_dict(cond_state, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

    cond_path = os.path.join(output_dir, "condition_encoder")
    cond_encoder.save_pretrained(cond_path)
    print(f"  Saved to {cond_path}")

    # Create and save scheduler
    print("Creating scheduler...")
    scheduler = DDIMScheduler(**configs["scheduler"])
    scheduler_path = os.path.join(output_dir, "scheduler")
    scheduler.save_pretrained(scheduler_path)
    print(f"  Saved to {scheduler_path}")

    # Save model index
    model_index = {
        "_class_name": "MatFusePipeline",
        "_diffusers_version": "0.25.0",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers_pipeline", "MatFuseVQModel"],
        "condition_encoder": ["diffusers_pipeline", "MultiConditionEncoder"],
        "scheduler": ["diffusers", "DDIMScheduler"],
    }

    import json

    index_path = os.path.join(output_dir, "model_index.json")
    with open(index_path, "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"Saved model index to {index_path}")

    print(f"\nConversion complete! Models saved to {output_dir}")
    print("\nTo load the pipeline:")
    print(
        f"""
from diffusers_pipeline import MatFusePipeline
pipe = MatFusePipeline.from_pretrained("{output_dir}")
"""
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert MatFuse checkpoint to diffusers format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the original MatFuse checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the converted models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the original config file (optional)",
    )

    args = parser.parse_args()

    convert_checkpoint(
        ckpt_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
