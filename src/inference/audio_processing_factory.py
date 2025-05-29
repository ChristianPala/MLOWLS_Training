from ..config import Config
from .mel_spectrogram_generator import MelSpectrogramGenerator
from .ogg_audio_processor import OGGAudioProcessor
from .overlap_segmenter import OverlapSegmenter


class AudioProcessingFactory:
    """Factory for creating audio processing pipeline."""

    @staticmethod
    def create_processor(config: Config) -> OGGAudioProcessor:
        """Create audio processor from configuration."""
        print("🏭 Creating audio processing pipeline...")

        # Create segmenter
        segmenter = OverlapSegmenter(
            segment_length=getattr(config, "segment_length", 30.0),
            overlap=getattr(config, "overlap", 0.5),
        )

        # Create spectrogram generator
        spectrogram_generator = MelSpectrogramGenerator(
            sample_rate=getattr(config, "sample_rate", 32000),
            n_mels=getattr(config, "n_mels", 128),
            n_fft=getattr(config, "n_fft", 2048),
            hop_length=getattr(config, "hop_length", 512),
            fmin=getattr(config, "fmin", 50.0),
            fmax=getattr(config, "fmax", 14000.0),
        )

        # Create processor
        processor = OGGAudioProcessor(
            segmenter=segmenter,
            spectrogram_generator=spectrogram_generator,
            target_sample_rate=getattr(config, "sample_rate", 32000),
        )

        print("✅ Audio processing pipeline ready!")
        return processor
