from typing import Any, Optional

import torch

from ..interfaces.metadata_handler import MetadataHandler
from ..interfaces.model_converter import ModelConverter
from ..interfaces.model_validator import ModelValidator


class ModelConversionService:
    """Service for orchestrating model conversion with dependency injection."""

    def __init__(
        self,
        converter: ModelConverter,
        validator: Optional[ModelValidator] = None,
        metadata_handler: Optional[MetadataHandler] = None,
    ):
        """Initialize conversion service.

        Args:
            converter: Model converter implementation
            validator: Optional model validator
            metadata_handler: Optional metadata handler
        """
        self.converter = converter
        self.validator = validator
        self.metadata_handler = metadata_handler

        print("🏭 Model Conversion Service initialized")
        print(f"   Converter: {type(converter).__name__}")
        print(f"   Validator: {type(validator).__name__ if validator else 'None'}")
        print(
            f"   Metadata Handler: {type(metadata_handler).__name__ if metadata_handler else 'None'}"
        )

    def convert_model(
        self,
        model: torch.nn.Module,
        output_path: str,
        validate: bool = True,
        save_metadata: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert model with validation and metadata handling.

        Args:
            model: Model to convert
            output_path: Output file path
            validate: Whether to validate conversion
            save_metadata: Whether to save metadata
            **kwargs: Additional conversion options

        Returns:
            Conversion results
        """
        print(f"🔄 Starting model conversion to: {output_path}")

        # Convert model
        conversion_results = self.converter.convert(model, output_path, **kwargs)

        # Validate if requested and validator available
        if validate and self.validator:
            # Create test input for validation
            # Assume converter has input_shape attribute
            if hasattr(self.converter, "input_shape"):
                test_input = torch.randn(self.converter.input_shape)
                validation_results = self.validator.validate(model, output_path, test_input)
                conversion_results["validation"] = validation_results

        # Save metadata if requested and handler available
        if save_metadata and self.metadata_handler:
            metadata_path = output_path.replace(".onnx", "_conversion.json")
            self.metadata_handler.save_metadata(conversion_results, metadata_path)

        print("✅ Model conversion completed!")
        return conversion_results
