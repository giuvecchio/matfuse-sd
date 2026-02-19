"""
MatFuse Pipeline for diffusers.

A custom diffusers pipeline for generating PBR material maps using the MatFuse model.

Note: This pipeline uses:
- Standard UNet2DConditionModel from diffusers (with custom in/out channels config)
- Custom MatFuseVQModel (required because MatFuse uses 4 separate encoders/quantizers)
"""

import os
import inspect
from typing import Optional, Union, List, Callable, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler

from .vae_matfuse import MatFuseVQModel
from .condition_encoders import MultiConditionEncoder


class MatFusePipeline(DiffusionPipeline):
    """
    Pipeline for generating PBR material maps using MatFuse.
    
    This pipeline generates 4 material maps (diffuse, normal, roughness, specular)
    from various conditioning inputs like reference images, text, sketches, and color palettes.
    
    Args:
        vae: MatFuseVQModel for encoding/decoding material maps (custom, required).
        unet: UNet2DConditionModel for denoising (standard diffusers model).
        scheduler: Diffusion scheduler.
        condition_encoder: Multi-condition encoder for processing inputs.
    
    Note:
        The VQ-VAE must be the custom MatFuseVQModel because MatFuse uses 4 separate
        encoders and quantizers (one per material map type). The UNet can be the
        standard diffusers UNet2DConditionModel configured with:
        - in_channels=16 (12 latent + 4 sketch concat)
        - out_channels=12 (4 maps × 3 channels)
        - cross_attention_dim=512
    """
    
    model_cpu_offload_seq = "condition_encoder->unet->vae"
    
    def __init__(
        self,
        vae: MatFuseVQModel,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler],
        condition_encoder: Optional[MultiConditionEncoder] = None,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            condition_encoder=condition_encoder,
        )
        
        self.vae_scale_factor = 8  # Downsampling factor of VQ-VAE
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load the MatFuse pipeline from a local directory.
        
        Loads each component (UNet, VAE, scheduler, condition_encoder) individually
        from their respective subdirectories.
        
        Args:
            pretrained_model_name_or_path: Path to the directory containing the model components.
            **kwargs: Additional keyword arguments (e.g., torch_dtype).
        """
        model_dir = pretrained_model_name_or_path
        torch_dtype = kwargs.get("torch_dtype", None)
        
        # Load UNet (standard diffusers)
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(model_dir, "unet"),
            torch_dtype=torch_dtype,
        )
        
        # Load VAE (custom)
        vae = MatFuseVQModel.from_pretrained(
            os.path.join(model_dir, "vae"),
            torch_dtype=torch_dtype,
        )
        
        # Load scheduler
        scheduler = DDIMScheduler.from_pretrained(
            os.path.join(model_dir, "scheduler"),
        )
        
        # Load condition encoder (custom) if it exists
        cond_dir = os.path.join(model_dir, "condition_encoder")
        condition_encoder = None
        if os.path.isdir(cond_dir):
            condition_encoder = MultiConditionEncoder.from_pretrained(
                cond_dir,
                torch_dtype=torch_dtype,
            )
        
        return cls(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            condition_encoder=condition_encoder,
        )
        
    @property
    def _execution_device(self):
        if self.device != torch.device("meta"):
            return self.device
        for name, model in self.components.items():
            if isinstance(model, torch.nn.Module):
                return next(model.parameters()).device
        # Also check condition_encoder (may not be in components dict)
        if self.condition_encoder is not None:
            return next(self.condition_encoder.parameters()).device
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        """Override to() to also move condition_encoder (not auto-tracked by diffusers)."""
        result = super().to(*args, **kwargs)
        if self.condition_encoder is not None:
            self.condition_encoder = self.condition_encoder.to(*args, **kwargs)
        return result

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to material maps."""
        # Add circular padding for seamless textures
        latents = F.pad(latents, (7, 7, 7, 7), mode="circular")
        
        # Decode
        materials = self.vae.decode(latents)
        
        # Center crop to remove padding
        _, _, h, w = materials.shape
        target_h = (h - 14 * self.vae_scale_factor)
        target_w = (w - 14 * self.vae_scale_factor)
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        materials = materials[:, :, start_h:start_h + target_h, start_w:start_w + target_w]
        
        return materials

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare initial noise latents."""
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        
        if latents is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        
        # Scale by scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents

    def prepare_extra_step_kwargs(self, generator: Optional[torch.Generator], eta: float) -> Dict[str, Any]:
        """Prepare extra kwargs for the scheduler step."""
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        
        return extra_step_kwargs

    def _encode_conditions(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[Union[str, List[str]]] = None,
        sketch: Optional[torch.Tensor] = None,
        palette: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        image_size: int = 256,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode all condition inputs through their respective encoders.
        
        When a condition is not provided, the encoder creates a placeholder
        and encodes it (matching training behavior), rather than using zero tensors.
        """
        device = device or self._execution_device
        
        if self.condition_encoder is not None:
            cond = self.condition_encoder(
                image_embed=image,
                text=text,
                sketch=sketch,
                palette=palette,
                batch_size=batch_size,
                image_size=image_size,
                device=device,
            )
            c_crossattn = cond["c_crossattn"]
            c_concat = cond["c_concat"]
        else:
            c_crossattn = None
            c_concat = None
        
        # Ensure proper dtype
        if c_crossattn is not None:
            c_crossattn = c_crossattn.to(dtype=dtype, device=device)
        if c_concat is not None:
            c_concat = c_concat.to(dtype=dtype, device=device)
        
        return c_crossattn, c_concat

    def _get_uncond_embeddings(
        self,
        batch_size: int,
        image_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get unconditional embeddings for classifier-free guidance.
        
        Creates proper unconditional embeddings by encoding placeholder inputs
        through the actual encoders (gray image → CLIP, empty string → SentenceTransformer,
        zero palette → PaletteEncoder, zero sketch → SketchEncoder).
        
        This matches the original training behavior where ucg_training drops conditions
        by setting them to val=0.0 (images/palette/sketch) or val="" (text), and then
        encoding those placeholder values through the encoders.
        """
        if self.condition_encoder is not None:
            uc = self.condition_encoder.get_unconditional_conditioning(
                batch_size=batch_size,
                image_size=image_size,
                device=device,
            )
            uc_crossattn = uc["c_crossattn"].to(dtype=dtype, device=device)
            uc_concat = uc["c_concat"].to(dtype=dtype, device=device)
        else:
            uc_crossattn = None
            uc_concat = None
        
        return uc_crossattn, uc_concat

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[torch.Tensor, Image.Image]] = None,
        text: Optional[Union[str, List[str]]] = None,
        sketch: Optional[Union[torch.Tensor, Image.Image]] = None,
        palette: Optional[Union[torch.Tensor, np.ndarray, List[Tuple[int, int, int]]]] = None,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate PBR material maps.
        
        Args:
            image: Reference image for style/appearance guidance.
            text: Text description of the material.
            sketch: Binary edge/sketch map for structure guidance.
            palette: Color palette (5 colors) for color guidance.
            height: Output image height.
            width: Output image width.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Number of images to generate per prompt.
            eta: DDIM eta parameter.
            generator: Random number generator for reproducibility.
            latents: Pre-generated noise latents.
            output_type: Output format ("pil", "tensor", "np").
            return_dict: Whether to return a dict.
            callback: Callback function called every `callback_steps` steps.
            callback_steps: Frequency of callback calls.
            
        Returns:
            Dictionary containing:
            - images: List of generated images (4 maps per generation).
            - diffuse: Diffuse/albedo maps.
            - normal: Normal maps.
            - roughness: Roughness maps.
            - specular: Specular maps.
        """
        device = self._execution_device
        dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float32
        
        # Determine batch size
        if text is not None and isinstance(text, str):
            batch_size = 1
        elif text is not None:
            batch_size = len(text)
        else:
            batch_size = 1
        
        batch_size = batch_size * num_images_per_prompt
        
        # Preprocess inputs
        if image is not None and isinstance(image, Image.Image):
            image = self._preprocess_image(image, device, dtype)
        
        if sketch is not None and isinstance(sketch, Image.Image):
            sketch = self._preprocess_sketch(sketch, height, width, device, dtype)
        
        if palette is not None and not isinstance(palette, torch.Tensor):
            palette = self._preprocess_palette(palette, device, dtype)
        
        # Encode conditions
        # The encoder handles None conditions by encoding placeholder inputs
        # (matching the original model's UCG training behavior)
        c_crossattn, c_concat = self._encode_conditions(
            image=image,
            text=text,
            sketch=sketch,
            palette=palette,
            batch_size=batch_size,
            image_size=height,
            device=device,
            dtype=dtype,
        )
        
        # Get unconditional embeddings for CFG
        # These are encoded placeholders, NOT zero tensors
        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            uc_crossattn, uc_concat = self._get_uncond_embeddings(
                batch_size, height, device, dtype
            )
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Prepare latent variables
        num_channels_latents = 12  # 4 maps * 3 channels per quantizer
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )
        
        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Prepare latent input with sketch conditioning
                if do_classifier_free_guidance:
                    # For CFG: unconditional uses uc_concat, conditional uses c_concat
                    latent_uncond = torch.cat([latents, uc_concat], dim=1)
                    latent_cond = torch.cat([latents, c_concat], dim=1)
                    latent_model_input = torch.cat([latent_uncond, latent_cond])
                    if c_crossattn is not None:
                        encoder_hidden_states = torch.cat([uc_crossattn, c_crossattn])
                    else:
                        encoder_hidden_states = None
                else:
                    latent_model_input = torch.cat([latents, c_concat], dim=1)
                    encoder_hidden_states = c_crossattn
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Predict noise
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
                # return_dict=False returns tuple, first element is sample
                if isinstance(noise_pred, tuple):
                    noise_pred = noise_pred[0]
                elif isinstance(noise_pred, dict):
                    noise_pred = noise_pred["sample"]
                
                # Classifier-free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Compute previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # Callback
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # Decode latents
        materials = self.decode_latents(latents)
        
        # Split into individual maps
        diffuse = materials[:, 0:3]
        normal = materials[:, 3:6]
        roughness = materials[:, 6:9]
        specular = materials[:, 9:12]
        
        # Post-process outputs
        if output_type == "pil":
            diffuse = self._tensor_to_pil(diffuse)
            normal = self._tensor_to_pil(normal)
            roughness = self._tensor_to_pil(roughness)
            specular = self._tensor_to_pil(specular)
        elif output_type == "np":
            diffuse = self._tensor_to_numpy(diffuse)
            normal = self._tensor_to_numpy(normal)
            roughness = self._tensor_to_numpy(roughness)
            specular = self._tensor_to_numpy(specular)
        
        if return_dict:
            return {
                "diffuse": diffuse,
                "normal": normal,
                "roughness": roughness,
                "specular": specular,
            }
        
        return (diffuse, normal, roughness, specular)

    def _preprocess_image(self, image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Preprocess PIL image to tensor."""
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image * 2.0 - 1.0  # Scale to [-1, 1]
        return image.to(device=device, dtype=dtype)

    def _preprocess_sketch(
        self,
        sketch: Image.Image,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Preprocess sketch image to tensor."""
        sketch = sketch.convert("L")
        sketch = sketch.resize((width, height), Image.BILINEAR)
        sketch = np.array(sketch).astype(np.float32) / 255.0
        sketch = torch.from_numpy(sketch).unsqueeze(0).unsqueeze(0)
        return sketch.to(device=device, dtype=dtype)

    def _preprocess_palette(
        self,
        palette: Union[np.ndarray, List[Tuple[int, int, int]]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Preprocess color palette to tensor."""
        if isinstance(palette, list):
            palette = np.array(palette, dtype=np.float32) / 255.0
        elif isinstance(palette, np.ndarray):
            if palette.max() > 1.0:
                palette = palette.astype(np.float32) / 255.0
            else:
                palette = palette.astype(np.float32)
        
        # Ensure 5 colors
        while len(palette) < 5:
            palette = np.concatenate([palette, palette[-1:]], axis=0)
        palette = palette[:5]
        
        palette = torch.from_numpy(palette).unsqueeze(0)
        return palette.to(device=device, dtype=dtype)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to list of PIL images."""
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
        tensor = tensor.cpu().permute(0, 2, 3, 1).numpy()
        tensor = (tensor * 255).astype(np.uint8)
        return [Image.fromarray(img) for img in tensor]

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0, 1)
        return tensor.cpu().permute(0, 2, 3, 1).numpy()
