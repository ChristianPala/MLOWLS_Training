import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.onnx

from ..config import Config
from ..persistence.model_saver import PyTorchModelSaver


class ModelConverter:
    """Convert PyTorch models to ONNX format with validation."""

    def __init__(self, input_shape: tuple[int, ...] = (1, 1, 128, 938)):
        """Initialize converter.

        Args:
            input_shape: Expected input shape (batch, channels, height, width)
                        Default matches mel-spectrogram: 128 mels x ~938 time steps (30s audio)
        """
        self.input_shape = input_shape
        print(f"🔄 Model Converter initialized with input shape: {input_shape}")

    def convert_to_onnx(
        self,
        model: torch.nn.Module,
        onnx_path: str,
        validate: bool = True,
        opset_version: int = 11,
        dynamic_batch: bool = True,
    ) -> dict[str, Any]:
        """Convert PyTorch model to ONNX.

        Args:
            model: Trained PyTorch model
            onnx_path: Output ONNX file path
            validate: Whether to validate conversion
            opset_version: ONNX opset version
            dynamic_batch: Whether to allow dynamic batch size

        Returns:
            Conversion metadata dictionary
        """
        print("🚀 Converting model to ONNX...")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Output path: {onnx_path}")
        print(f"   Opset version: {opset_version}")

        # Ensure model is in evaluation mode
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(self.input_shape)
        print(f"   Dummy input shape: {dummy_input.shape}")

        # Ensure output directory exists
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

        # Configure dynamic axes if requested
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"spectrogram": {0: "batch_size"}, "predictions": {0: "batch_size"}}

        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["spectrogram"],
                output_names=["predictions"],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )
            print("✅ ONNX export completed")

        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {str(e)}")

        # Validate the exported model
        validation_results = {}
        if validate:
            validation_results = self._validate_conversion(model, onnx_path, dummy_input)

        # Get model info
        model_info = self._get_onnx_model_info(onnx_path)

        # Create conversion metadata
        metadata = {
            "input_shape": self.input_shape,
            "output_path": onnx_path,
            "opset_version": opset_version,
            "dynamic_batch": dynamic_batch,
            "model_info": model_info,
            "validation": validation_results,
            "conversion_status": "success",
        }

        print("✅ Conversion completed successfully!")
        return metadata

    def _validate_conversion(
        self, pytorch_model: torch.nn.Module, onnx_path: str, dummy_input: torch.Tensor
    ) -> dict[str, Any]:
        """Validate ONNX conversion accuracy."""
        print("🔍 Validating ONNX conversion...")

        try:
            # PyTorch prediction
            with torch.no_grad():
                pytorch_output = pytorch_model(dummy_input)
                pytorch_output = pytorch_output.detach().cpu().numpy()

            # ONNX prediction
            ort_session = ort.InferenceSession(onnx_path)
            onnx_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]

            # Compare outputs
            diff = np.abs(pytorch_output - onnx_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            # Check if conversion is accurate
            tolerance = 1e-5
            is_accurate = max_diff < tolerance

            validation_results = {
                "is_accurate": is_accurate,
                "max_difference": float(max_diff),
                "mean_difference": float(mean_diff),
                "tolerance": tolerance,
                "pytorch_output_shape": pytorch_output.shape,
                "onnx_output_shape": onnx_output.shape,
            }

            if is_accurate:
                print(f"✅ Validation passed! Max difference: {max_diff:.2e}")
            else:
                print(
                    f"⚠️  Validation warning! Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})"
                )

            return validation_results

        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            return {"is_accurate": False, "error": str(e), "validation_status": "failed"}

    def _get_onnx_model_info(self, onnx_path: str) -> dict[str, Any]:
        """Get information about the ONNX model."""
        try:
            model = onnx.load(onnx_path)

            # Get input/output info
            inputs = []
            for input_tensor in model.graph.input:
                shape = [
                    dim.dim_value if dim.dim_value > 0 else -1
                    for dim in input_tensor.type.tensor_type.shape.dim
                ]
                inputs.append(
                    {
                        "name": input_tensor.name,
                        "shape": shape,
                        "type": input_tensor.type.tensor_type.elem_type,
                    }
                )

            outputs = []
            for output_tensor in model.graph.output:
                shape = [
                    dim.dim_value if dim.dim_value > 0 else -1
                    for dim in output_tensor.type.tensor_type.shape.dim
                ]
                outputs.append(
                    {
                        "name": output_tensor.name,
                        "shape": shape,
                        "type": output_tensor.type.tensor_type.elem_type,
                    }
                )

            # Get file size
            file_size_mb = Path(onnx_path).stat().st_size / 1024 / 1024

            return {
                "inputs": inputs,
                "outputs": outputs,
                "file_size_mb": round(file_size_mb, 2),
                "opset_version": model.opset_import[0].version if model.opset_import else None,
            }

        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def convert_best_model(
        cls, model_path: str, config_path: str, onnx_path: str, validate: bool = True
    ) -> dict[str, Any]:
        """Convert the best saved model to ONNX.

        Args:
            model_path: Path to saved PyTorch model (.pth)
            config_path: Path to training config
            onnx_path: Output ONNX path
            validate: Whether to validate conversion

        Returns:
            Conversion metadata
        """
        print(f"🔄 Converting best model: {model_path} → {onnx_path}")

        # Load config
        config = Config(config_path)

        # Load metadata if available
        metadata_path = model_path.replace(".pth", "_metadata.json")
        model_metadata = {}
        if Path(metadata_path).exists():
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)
            print(f"📋 Loaded model metadata from epoch {model_metadata.get('epoch', 'unknown')}")

        # Recreate model architecture
        model = timm.create_model(
            config.backbone,
            pretrained=False,  # Don't load pretrained weights
            num_classes=config.num_classes,
            in_chans=1,
            drop_rate=config.dropout,
        )

        # Load trained weights
        saver = PyTorchModelSaver()
        model = saver.load_model(model, model_path)

        print(f"✅ Model loaded: {config.backbone} with {config.num_classes} classes")

        # Convert to ONNX
        converter = cls()
        conversion_results = converter.convert_to_onnx(model, onnx_path, validate=validate)

        # Add original model metadata
        conversion_results["original_model"] = {
            "path": model_path,
            "config_path": config_path,
            "metadata": model_metadata,
            "architecture": {
                "backbone": config.backbone,
                "num_classes": config.num_classes,
                "dropout": config.dropout,
            },
        }

        # Save conversion metadata
        conversion_metadata_path = onnx_path.replace(".onnx", "_conversion.json")
        with open(conversion_metadata_path, "w") as f:
            json.dump(conversion_results, f, indent=2)

        print(f"💾 Conversion metadata saved: {conversion_metadata_path}")

        return conversion_results
