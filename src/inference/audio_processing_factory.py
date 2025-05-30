from ..config import Config
from .mel_spectrogram_generator import MelSpectrogramGenerator
from .ogg_audio_processor import OGGAudioProcessor
from .overlap_segmenter import OverlapSegmenter


class AudioProcessingFactory:
    """Factory for creating audio processing pipeline."""

    @staticmethod
    def create_processor(config: Config, target_frames: int | None = None) -> OGGAudioProcessor:
        """Create audio processor from config.

        Args:
            config: Training configuration
            target_frames: Target frame count for fixed-size spectrograms
        """
        print("🏭 Creating audio processing pipeline...")

        audio_config = getattr(config, "audio", {})

        sample_rate = audio_config.get("sample_rate", 32000)
        segment_length = audio_config.get("segment_length", 30.0)
        overlap = audio_config.get("overlap", 0.5)

        n_mels = audio_config.get("n_mels", 128)
        n_fft = audio_config.get("n_fft", 1024)
        hop_length = audio_config.get("hop_length", 320)
        fmin = audio_config.get("fmin", 20.0)
        fmax = audio_config.get("fmax", 16000.0)

        print("📊 Audio parameters from config:")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   N_FFT: {n_fft}")
        print(f"   Hop length: {hop_length}")
        print(f"   Mel bands: {n_mels}")
        print(f"   Freq range: {fmin}-{fmax}Hz")
        print(f"   Segment length: {segment_length}s")
        # Create segmenter
        segmenter = OverlapSegmenter(segment_length=segment_length, overlap=overlap)

        # Create spectrogram generator with target frames
        spectrogram_generator = MelSpectrogramGenerator(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            power=2.0,
        )

        # Create processor
        processor = OGGAudioProcessor(
            segmenter=segmenter,
            spectrogram_generator=spectrogram_generator,
            target_sample_rate=sample_rate,
        )

        print("✅ Audio processing pipeline ready!")
        return processor
