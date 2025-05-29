from typing import Any

from ..config import Config
from .json_metadata_handler import JSONMetadataHandler
from .model_conversion_service import ModelConversionService
from .onnx_converter import ONNXConverter
from .onnx_validator import ONNXValidator
from .trained_model_loader import TrainedModelLoader


class ConversionPipeline:
    """Complete model conversion pipeline."""

    @staticmethod
    def convert_trained_model(
        model_path: str,
        config_path: str,
        output_path: str,
        validate: bool = True,
        **converter_kwargs: Any,
    ) -> dict[str, Any]:
        """Convert a trained model with full pipeline.

        Args:
            model_path: Path to trained model
            config_path: Path to training config
            output_path: Output conversion path
            validate: Whether to validate conversion
            **converter_kwargs: Additional converter options

        Returns:
            Conversion results with metadata
        """
        print(f"🚀 Full Conversion Pipeline: {model_path} → {output_path}")

        # Load configuration
        config = Config(config_path)

        # Create components (Dependency Injection)
        converter = ONNXConverter(**converter_kwargs)
        validator = ONNXValidator() if validate else None
        metadata_handler = JSONMetadataHandler()
        model_loader = TrainedModelLoader(config)

        # Create service
        service = ModelConversionService(
            converter=converter, validator=validator, metadata_handler=metadata_handler
        )

        # Load model
        model = model_loader.load_model(model_path)

        # Convert with metadata
        results = service.convert_model(
            model=model, output_path=output_path, validate=validate, save_metadata=True
        )

        # Add original model info
        results["original_model"] = {
            "path": model_path,
            "config_path": config_path,
            "architecture": {
                "backbone": config.backbone,
                "num_classes": config.num_classes,
                "dropout": config.dropout,
            },
        }

        return results
