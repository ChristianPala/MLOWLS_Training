import argparse
from pathlib import Path

from .conversion_pipeline import ConversionPipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert BirdCLEF PyTorch model to ONNX")

    parser.add_argument(
        "--model", type=str, required=True, help="Path to PyTorch model file (.pth)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX file path")
    parser.add_argument("--no-validate", action="store_true", help="Skip conversion validation")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11)")
    parser.add_argument(
        "--static-batch", action="store_true", help="Use static batch size (no dynamic batching)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Validation tolerance (default: 1e-5)"
    )

    return parser.parse_args()


def main() -> None:
    """Main conversion function."""
    args = parse_args()

    print("🔄 BirdCLEF Model → ONNX Converter")
    print("=" * 60)

    # Validate input files
    model_path = Path(args.model)
    config_path = Path(args.config)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Set output path
    output_path = Path(args.output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".onnx")

    print(f"📂 Input model: {model_path}")
    print(f"⚙️  Config file: {config_path}")
    print(f"📤 Output ONNX: {output_path}")
    print(f"🔍 Validate: {not args.no_validate}")

    try:
        # Use pipeline for conversion
        results = ConversionPipeline.convert_trained_model(
            model_path=str(model_path),
            config_path=str(config_path),
            output_path=str(output_path),
            validate=not args.no_validate,
            # Converter options
            opset_version=args.opset,
            dynamic_batch=not args.static_batch,
        )

        # Print summary
        print("\n✅ Conversion Summary:")
        print(f"   📁 ONNX file: {output_path}")

        if "validation" in results and results["validation"].get("is_accurate"):
            max_diff = results["validation"].get("max_difference", 0)
            print(f"   ✅ Validation: Passed (max diff: {max_diff:.2e})")
        elif "validation" in results:
            print("   ⚠️  Validation: Failed or skipped")

        print("\n🎉 Model conversion completed successfully!")

    except Exception as e:
        print(f"\n❌ Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
