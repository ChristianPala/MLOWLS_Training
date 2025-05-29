from typing import Any, Dict

import numpy as np
import onnxruntime as ort
import torch

from ..interfaces.model_validator import ModelValidator


class ONNXValidator(ModelValidator):
    """ONNX implementation of model validator."""

    def __init__(self, tolerance: float = 1e-5):
        """Initialize ONNX validator.

        Args:
            tolerance: Maximum allowed difference
        """
        self.tolerance = tolerance
        print(f"🔍 ONNX Validator initialized (tolerance: {tolerance:.2e})")

    def validate(
        self, original_model: torch.nn.Module, converted_model_path: str, test_input: torch.Tensor
    ) -> Dict[str, Any]:
        """Validate ONNX model against PyTorch original."""
        print("🔍 Validating ONNX conversion...")

        try:
            # PyTorch prediction
            with torch.no_grad():
                pytorch_output = original_model(test_input)
                pytorch_output = pytorch_output.detach().cpu().numpy()

            # ONNX prediction
            ort_session = ort.InferenceSession(converted_model_path)
            onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
            onnx_output = ort_session.run(None, onnx_input)[0]

            # Compare outputs
            diff = np.abs(pytorch_output - onnx_output)
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))

            # Check accuracy
            is_accurate = max_diff < self.tolerance

            validation_results = {
                "is_accurate": is_accurate,
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "tolerance": self.tolerance,
                "pytorch_output_shape": list(pytorch_output.shape),
                "onnx_output_shape": list(onnx_output.shape),
            }

            if is_accurate:
                print(f"✅ Validation passed! Max difference: {max_diff:.2e}")
            else:
                print(f"⚠️  Validation warning! Max difference: {max_diff:.2e}")

            return validation_results

        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            return {"is_accurate": False, "error": str(e), "validation_status": "failed"}
