# src/inference/__init__.py
"""
Inference pipeline for BirdCLEF models.
"""

from .convert import main as convert_main
from .model_converter import ModelConverter

__all__ = ["ModelConverter", "convert_main"]
