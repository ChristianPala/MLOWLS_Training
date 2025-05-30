from ..config import Config
from .training_transform_processor import TrainingTransformProcessor


class AudioProcessingFactory:
    """Factory for creating audio processing pipeline."""

    @staticmethod
    def create_processor(config: Config) -> TrainingTransformProcessor:
        """Create processor using training transforms."""
        print("🏭 Creating audio processing pipeline...")

        processor = TrainingTransformProcessor(config)

        print("✅ Audio processing pipeline ready!")
        return processor
