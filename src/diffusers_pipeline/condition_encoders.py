"""
MatFuse Condition Encoders for diffusers.

These encoders handle the multi-modal conditioning:
- Image embedding (CLIP image encoder)
- Text embedding (CLIP text encoder)
- Sketch encoder (CNN)
- Palette encoder (MLP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, List
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class SketchEncoder(ModelMixin, ConfigMixin):
    """
    CNN encoder for binary sketch/edge maps.

    Takes a single-channel binary image and encodes it to a spatial feature map
    that will be concatenated with the latent for hybrid conditioning.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 7, 1, 1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sketch input.

        Args:
            x: Input tensor of shape (B, 1, H, W) with values in [0, 1].

        Returns:
            Encoded features of shape (B, out_channels, H/8, W/8).
        """
        return self.net(x)


class PaletteEncoder(ModelMixin, ConfigMixin):
    """
    MLP encoder for color palettes.

    Takes a color palette (N colors, RGB) and encodes it to a single embedding
    for cross-attention conditioning.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 512,
        n_colors: int = 5,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(hidden_channels * n_colors, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode color palette.

        Args:
            x: Input tensor of shape (B, n_colors, 3) with RGB values in [0, 1].

        Returns:
            Encoded embedding of shape (B, out_channels).
        """
        return self.net(x)


class CLIPImageEncoder(ModelMixin, ConfigMixin):
    """
    Wrapper for CLIP image encoder using the OpenAI CLIP library.

    Generates image embeddings for cross-attention conditioning.
    """

    @register_to_config
    def __init__(
        self,
        model_name: str = "ViT-B/16",
        normalize: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.normalize = normalize
        self.model = None  # Lazy loading

        # Register normalization buffers
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False
        )

    def _load_model(self):
        """Lazy load the CLIP model."""
        if self.model is None:
            import clip

            self.model, _ = clip.load(self.model_name, device="cpu", jit=False)
            self.model = self.model.visual

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for CLIP."""
        # Resize to 224x224
        x = F.interpolate(
            x, size=(224, 224), mode="bicubic", align_corners=True, antialias=True
        )
        # Normalize from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        # Normalize according to CLIP - move mean/std to device if needed
        mean = self.mean.to(x.device).view(1, 3, 1, 1)
        std = self.std.to(x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image using CLIP.

        Args:
            x: Input tensor of shape (B, 3, H, W) with values in [-1, 1].

        Returns:
            Image embedding of shape (B, 1, 512).
        """
        self._load_model()

        # Move model to same device as input
        device = x.device
        self.model = self.model.to(device)

        x = self.preprocess(x)
        z = self.model(x).float().unsqueeze(1)  # (B, 1, 512)

        if self.normalize:
            z = z / torch.linalg.norm(z, dim=2, keepdim=True)

        return z


class CLIPTextEncoder(ModelMixin, ConfigMixin):
    """
    Wrapper for CLIP sentence encoder using sentence-transformers.

    Generates text embeddings for cross-attention conditioning.
    """

    @register_to_config
    def __init__(
        self,
        model_name: str = "sentence-transformers/clip-ViT-B-16",
    ):
        super().__init__()

        self.model_name = model_name
        self.model = None  # Lazy loading

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.model.eval()

    def forward(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text using CLIP sentence transformer.

        Args:
            text: Input text or list of texts.

        Returns:
            Text embedding of shape (B, 512).
        """
        self._load_model()

        if isinstance(text, str):
            text = [text]

        embeddings = self.model.encode(text, convert_to_tensor=True)
        return embeddings


class MultiConditionEncoder(ModelMixin, ConfigMixin):
    """
    Multi-condition encoder that combines all conditioning modalities.

    This encoder takes multiple condition inputs and produces:
    - c_crossattn: Features for cross-attention (image, text, palette embeddings)
    - c_concat: Features for concatenation (sketch encoding)
    """

    @register_to_config
    def __init__(
        self,
        sketch_in_channels: int = 1,
        sketch_out_channels: int = 4,
        palette_in_channels: int = 3,
        palette_hidden_channels: int = 64,
        palette_out_channels: int = 512,
        n_colors: int = 5,
        clip_image_model: str = "ViT-B/16",
        clip_text_model: str = "sentence-transformers/clip-ViT-B-16",
    ):
        super().__init__()

        self.sketch_encoder = SketchEncoder(
            in_channels=sketch_in_channels,
            out_channels=sketch_out_channels,
        )

        self.palette_encoder = PaletteEncoder(
            in_channels=palette_in_channels,
            hidden_channels=palette_hidden_channels,
            out_channels=palette_out_channels,
            n_colors=n_colors,
        )

        # CLIP encoders are lazy-loaded
        self.clip_image_encoder = None
        self.clip_text_encoder = None
        self._clip_image_model = clip_image_model
        self._clip_text_model = clip_text_model

    def _load_clip_encoders(self):
        """Lazy load CLIP encoders."""
        if self.clip_image_encoder is None:
            self.clip_image_encoder = CLIPImageEncoder(
                model_name=self._clip_image_model
            )
        if self.clip_text_encoder is None:
            self.clip_text_encoder = CLIPTextEncoder(model_name=self._clip_text_model)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image using CLIP."""
        self._load_clip_encoders()
        return self.clip_image_encoder(image)

    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Encode text using CLIP."""
        self._load_clip_encoders()
        return self.clip_text_encoder(text)

    def encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """Encode sketch/edge map."""
        return self.sketch_encoder(sketch)

    def encode_palette(self, palette: torch.Tensor) -> torch.Tensor:
        """Encode color palette."""
        return self.palette_encoder(palette)

    def get_unconditional_conditioning(
        self,
        batch_size: int = 1,
        image_size: int = 256,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get unconditional conditioning for classifier-free guidance.

        IMPORTANT: The original model was trained to drop conditions by replacing them
        with encoded placeholders (zero/gray image through CLIP, empty string through
        sentence-transformers, zero palette through PaletteEncoder, zero sketch through
        SketchEncoder) — NOT with zero tensors. This method produces the correct
        unconditional embeddings.

        Args:
            batch_size: Batch size.
            image_size: Image resolution (for sketch spatial dims).
            device: Device to place tensors on.

        Returns:
            Dictionary with c_crossattn and c_concat for unconditional guidance.
        """
        return self.forward(
            image_embed=None,
            text=None,
            sketch=None,
            palette=None,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
        )

    def forward(
        self,
        image_embed: Optional[torch.Tensor] = None,
        text: Optional[Union[str, List[str]]] = None,
        sketch: Optional[torch.Tensor] = None,
        palette: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        image_size: int = 256,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all conditions.

        When a condition is not provided, the model encodes a placeholder input
        through the actual encoder (matching training behavior) rather than using
        zero tensors. This is critical because the model was trained with:
        - Image drop → CLIP encoding of a gray/zero image (0.0 in [-1,1])
        - Text drop → sentence-transformer encoding of ""
        - Palette drop → PaletteEncoder(zeros)
        - Sketch drop → SketchEncoder(zeros)

        Args:
            image_embed: Reference image of shape (B, 3, H, W) in [-1, 1].
            text: Text description(s).
            sketch: Binary sketch of shape (B, 1, H, W) in [0, 1].
            palette: Color palette of shape (B, n_colors, 3) in [0, 1].
            batch_size: Batch size (used when no inputs are provided).
            image_size: Image resolution (used to create placeholder sketch).
            device: Device to place tensors on.

        Returns:
            Dictionary with:
            - c_crossattn: Cross-attention context of shape (B, 3, 512) - always 3 tokens.
            - c_concat: Concatenation features of shape (B, 4, H/8, W/8).
        """
        self._load_clip_encoders()

        # Determine batch size and device from any available input
        if image_embed is not None:
            batch_size = image_embed.shape[0]
            device = device or image_embed.device
            image_size = image_embed.shape[-1]
        elif sketch is not None:
            batch_size = sketch.shape[0]
            device = device or sketch.device
            image_size = sketch.shape[-1]
        elif palette is not None:
            batch_size = palette.shape[0]
            device = device or palette.device

        device = device or torch.device("cpu")

        # --- Image embedding (token 0) ---
        # When not provided, encode a zero (gray) image through CLIP, matching training ucg_training val=0.0
        if image_embed is not None:
            img_emb = self.clip_image_encoder(image_embed)  # (B, 1, 512)
        else:
            placeholder_img = torch.zeros(
                batch_size, 3, image_size, image_size, device=device
            )
            img_emb = self.clip_image_encoder(placeholder_img)  # (B, 1, 512)

        # --- Text embedding (token 1) ---
        # When not provided, encode empty string through sentence-transformers, matching training ucg_training val=""
        if text is not None:
            text_emb = self.clip_text_encoder(text)  # (B, 512)
            if device is not None:
                text_emb = text_emb.to(device)
            text_emb = text_emb.unsqueeze(1)  # (B, 1, 512)
        else:
            text_emb = self.clip_text_encoder([""] * batch_size)  # (B, 512)
            text_emb = text_emb.to(device).unsqueeze(1)  # (B, 1, 512)

        # --- Palette embedding (token 2) ---
        # When not provided, encode zero palette through PaletteEncoder, matching training ucg_training val=0.0
        if palette is not None:
            palette_emb = self.palette_encoder(palette)  # (B, 512)
            palette_emb = palette_emb.unsqueeze(1)  # (B, 1, 512)
        else:
            n_colors = self.config.get("n_colors", 5)
            placeholder_palette = torch.zeros(batch_size, n_colors, 3, device=device)
            palette_emb = self.palette_encoder(placeholder_palette)  # (B, 512)
            palette_emb = palette_emb.unsqueeze(1)  # (B, 1, 512)

        # Combine cross-attention embeddings - always (B, 3, 512)
        c_crossattn = torch.cat([img_emb, text_emb, palette_emb], dim=1)

        # --- Sketch encoding for concatenation ---
        # When not provided, encode zero sketch through SketchEncoder, matching training ucg_training val=0.0
        if sketch is not None:
            c_concat = self.sketch_encoder(sketch)  # (B, 4, H/8, W/8)
        else:
            placeholder_sketch = torch.zeros(
                batch_size, 1, image_size, image_size, device=device
            )
            c_concat = self.sketch_encoder(placeholder_sketch)  # (B, 4, H/8, W/8)

        return {
            "c_crossattn": c_crossattn,
            "c_concat": c_concat,
        }
