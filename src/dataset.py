import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

from src.utils import normalize_waveform, get_mel_log_transform
from src.config import Config


def collate_fn(batch):
    specs, labels = zip(*batch)
    return torch.stack(specs), torch.stack(labels)


class BirdClefDataset(Dataset):
    """
    PyTorch Dataset for BirdCLEF train/validation clips.
    Requires that `data_source` (CSV path or DataFrame) already has
    a 'label_idx' column. Enforces a fixed audio segment length.
    """
    def __init__(self,
                 data_source,
                 audio_dir: str,
                 config: Config,
                 transform=None):
        # Load metadata
        self.df = data_source.copy() if isinstance(data_source, pd.DataFrame) \
                  else pd.read_csv(data_source)
        assert 'label_idx' in self.df.columns, "DataFrame must contain 'label_idx'."

        self.audio_dir = audio_dir
        self.config    = config

        # How many samples per segment?
        self.segment_samples = int(self.config.sample_rate * self.config.segment_length)

        # Build mel-log transform if not provided
        self.transform = transform or get_mel_log_transform(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            power=2.0
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.audio_dir, row['filename'])

        # Load full waveform
        waveform, sr = torchaudio.load(filepath)

        # Resample if needed
        if sr != self.config.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.config.sample_rate
            )(waveform)

        # Crop or pad to fixed length
        num_samps = waveform.size(1)
        tgt = self.segment_samples

        if num_samps >= tgt:
            # take first tgt samples (you could also random-crop here)
            waveform = waveform[:, :tgt]
        else:
            # pad zeros on the right
            pad_amt = tgt - num_samps
            waveform = F.pad(waveform, (0, pad_amt))

        # Normalize
        waveform = normalize_waveform(waveform)

        # Mel-spectrogram + dB
        melspec = self.transform(waveform)
        # Collapse multi-channel to mono
        if melspec.shape[0] > 1:
            melspec = melspec.mean(dim=0, keepdim=True)

        # Get the integer class
        label = torch.tensor(row['label_idx'], dtype=torch.long)

        return melspec, label