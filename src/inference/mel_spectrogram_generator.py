from typing import Any

import librosa
import numpy as np

from ..interfaces.spectrogram_generator import SpectrogramGenerator


class MelSpectrogramGenerator(SpectrogramGenerator):
    """Mel-spectrogram generator matching training configuration."""

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 50.0,
        fmax: float = 14000.0,
        power: float = 2.0,
    ) -> None:
        """Initialize mel-spectrogram generator."""
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.power = power

        print("🎼 Mel-Spectrogram Generator initialized")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Mel bands: {n_mels}")
        print(f"   FFT size: {n_fft}")
        print(f"   Frequency range: {fmin}-{fmax}Hz")

    def generate(self, audio: np.ndarray) -> np.ndarray:
        """Generate mel-spectrogram from audio."""
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power,
        )

        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return log_mel_spec

    def configure(self, **kwargs: Any) -> None:
        """Configure spectrogram parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"🔧 Updated {key}: {value}")
