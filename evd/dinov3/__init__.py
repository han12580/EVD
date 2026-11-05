"""Editable local DINOv3 ViT implementation used by EVD."""

from .configuration_dinov3_vit import DINOv3ViTConfig
from .modeling_dinov3_vit import DINOv3ViTModel

__all__ = ["DINOv3ViTConfig", "DINOv3ViTModel"]
