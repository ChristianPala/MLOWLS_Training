from pathlib import Path
from typing import Any

import torch
import torch.onnx

from ..config import Config
from ..interfaces.model_converter import ModelConverter


class ONNXConverter(ModelConverter):
    """ONNX implementation of model converter."""

    def __init__(
        self,
        input_shape: tuple[int, ...] | None = None,
        config: Config | None = None,
        opset_version: int = 11,
        dynamic_batch: bool = True,
    ) -> None:
        """Initialize ONNX converter.

        Args:
            input_shape: Expected input shape (if None, calculate from config)
            config: Training config to calculate shape from
            opset_version: ONNX opset version
            dynamic_batch: Enable dynamic batch size
        """
        if input_shape is None and config is not None:
            # Calculate input shape from NESTED config
            # The config has audio parameters under 'audio:' section
            audio_config = getattr(config, "audio", {})

            sample_rate = audio_config.get("sample_rate", 32000)
            segment_length = audio_config.get("segment_length", 30.0)
            n_fft = audio_config.get("n_fft", 1024)
            hop_length = audio_config.get("hop_length", 320)
            n_mels = audio_config.get("n_mels", 128)

            n_samples = int(segment_length * sample_rate)
            n_frames = 1 + (n_samples - n_fft) // hop_length

            input_shape = (1, 1, n_mels, n_frames)
            print(f"📊 Calculated input shape from config: {input_shape}")
            print(f"   Using audio config: sr={sample_rate}, n_fft={n_fft}, hop={hop_length}")

        self.input_shape = input_shape or (1, 1, 128, 938)  # fallback
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch

        print("🔄 ONNX Converter initialized")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Opset version: {opset_version}")
        print(f"   Dynamic batch: {dynamic_batch}")

    def convert(self, model: torch.nn.Module, output_path: str, **kwargs: Any) -> dict[str, Any]:
        """Convert PyTorch model to ONNX format."""
        print(f"🚀 Converting to ONNX: {output_path}")

        # Ensure model is in evaluation mode
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(self.input_shape)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Configure dynamic axes
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {"spectrogram": {0: "batch_size"}, "predictions": {0: "batch_size"}}

        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=["spectrogram"],
                output_names=["predictions"],
                dynamic_axes=dynamic_axes,
                verbose=False,
            )
            print("✅ ONNX export completed")

        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {str(e)}")

        # Return conversion metadata
        return {
            "format": "ONNX",
            "input_shape": self.input_shape,
            "output_path": output_path,
            "opset_version": self.opset_version,
            "dynamic_batch": self.dynamic_batch,
            "conversion_status": "success",
        }
