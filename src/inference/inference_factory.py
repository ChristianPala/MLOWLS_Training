# src/inference/inference_factory.py
"""
Factory for creating complete inference pipelines.
"""

import json
from pathlib import Path
from typing import Any, cast

from ..config import Config
from .audio_processing_factory import AudioProcessingFactory
from .onnx_predictor import ONNXPredictor
from .prediction_pipeline import BirdCLEFPredictionPipeline


class InferenceFactory:
    """Factory for creating complete inference pipelines."""

    @staticmethod
    def create_pipeline(
        onnx_model_path: str,
        config_path: str,
        species_names: list[str] | None = None,
        top_k: int = 5,
        confidence_threshold: float = 0.1,
        use_gpu: bool = True,
    ) -> BirdCLEFPredictionPipeline:
        """Create complete inference pipeline.

        Args:
            onnx_model_path: Path to ONNX model
            config_path: Path to training config
            species_names: List of species names (or load from taxonomy)
            top_k: Number of top predictions
            confidence_threshold: Minimum confidence
            use_gpu: Whether to use GPU if available

        Returns:
            Ready-to-use prediction pipeline
        """
        print("🏭 Creating inference pipeline...")
        print(f"   Model: {Path(onnx_model_path).name}")
        print(f"   Config: {Path(config_path).name}")

        # Load configuration
        config = Config(config_path)

        # Create audio processor
        audio_processor = AudioProcessingFactory.create_processor(config)

        # Create predictor with GPU/CPU selection
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            try:
                import onnxruntime as ort

                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                    print("🚀 GPU acceleration enabled")
                else:
                    print("🖥️  GPU not available, using CPU")
            except ImportError:
                print("⚠️  ONNX Runtime not found, using CPU")

        predictor = ONNXPredictor(onnx_model_path, providers=providers)

        # Load species names if not provided
        if species_names is None:
            species_names = InferenceFactory._load_species_names(config)

        # Create pipeline
        pipeline = BirdCLEFPredictionPipeline(
            predictor=predictor,
            audio_processor=audio_processor,
            species_names=species_names,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        print("✅ Inference pipeline ready!")
        return pipeline

    @staticmethod
    def _load_species_names(config: Config) -> list[str]:
        """Load species names from taxonomy or config."""
        # Try to load from taxonomy CSV if specified
        if hasattr(config, "taxonomy_csv"):
            try:
                import pandas as pd

                taxonomy_df = pd.read_csv(config.taxonomy_csv)

                if "primary_label" in taxonomy_df.columns:
                    # Cast to list[str] to satisfy mypy
                    species_names = cast(list[str], taxonomy_df["primary_label"].tolist())
                    print(f"📋 Loaded {len(species_names)} species from taxonomy")
                    return species_names

            except Exception as e:
                print(f"⚠️  Could not load taxonomy: {e}")

        # Fallback: generate generic names
        num_classes = getattr(config, "num_classes", 206)
        # Ensure num_classes is int for type safety
        if not isinstance(num_classes, int):
            num_classes = 206

        species_names = [f"Species_{i:03d}" for i in range(num_classes)]
        print(f"📋 Using generic species names for {num_classes} classes")

        return species_names

    @staticmethod
    def create_from_converted_model(
        conversion_metadata_path: str,
        top_k: int = 5,
        confidence_threshold: float = 0.1,
        use_gpu: bool = True,
    ) -> BirdCLEFPredictionPipeline:
        """Create pipeline from conversion metadata.

        Args:
            conversion_metadata_path: Path to conversion metadata JSON
            top_k: Number of top predictions
            confidence_threshold: Minimum confidence
            use_gpu: Whether to use GPU

        Returns:
            Inference pipeline
        """
        print(f"🔧 Creating pipeline from metadata: {Path(conversion_metadata_path).name}")

        # Load conversion metadata
        with open(conversion_metadata_path, "r") as f:
            metadata: dict[str, Any] = json.load(f)

        # Extract paths with type safety
        onnx_path = str(metadata["output_path"])
        config_path = str(metadata["original_model"]["config_path"])

        print(f"   ONNX model: {Path(onnx_path).name}")
        print(f"   Config: {Path(config_path).name}")

        # Create pipeline
        return InferenceFactory.create_pipeline(
            onnx_model_path=onnx_path,
            config_path=config_path,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            use_gpu=use_gpu,
        )
