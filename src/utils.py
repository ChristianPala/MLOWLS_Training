import torch
import torchaudio.transforms as T


def normalize_waveform(waveform: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Normalize a waveform to zero mean and unit variance per sample.

    Args:
        waveform (Tensor): Audio tensor of shape (channels, samples).
        eps (float): Small constant to avoid division by zero.

    Returns:
        Tensor: Normalized waveform.
    """
    mean = waveform.mean(dim=-1, keepdim=True)
    std = waveform.std(dim=-1, keepdim=True)
    return (waveform - mean) / (std + eps)


def get_mel_log_transform(
    sample_rate: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 224,
    fmin: float = 20.0,
    fmax: float = 16000.0,
    power: float = 2.0
) -> torch.nn.Module:
    """
    Create a sequential transform pipeline that computes a log-mel spectrogram
    from waveform.

    Args:
        sample_rate (int): Sampling rate of the audio.
        n_fft (int): FFT window size.
        hop_length (int): Number of audio samples between STFT columns.
        n_mels (int): Number of mel frequency bins.
        fmin (float): Minimum frequency (Hz).
        fmax (float): Maximum frequency (Hz).
        power (float): Exponent for the magnitude spectrogram.

    Returns:
        nn.Module: A Torch Sequential module that applies MelSpectrogram and AmplitudeToDB.
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=power,
    )
    amplitude_to_db = T.AmplitudeToDB(stype='power')
    return torch.nn.Sequential(
        mel_spectrogram,
        amplitude_to_db
    )
