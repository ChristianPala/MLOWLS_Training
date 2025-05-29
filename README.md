# MLOWLS Training - BirdCLEF 2025

A robust training pipeline for BirdCLEF 2025 with weak label learning, featuring multiple overlapping segments, mixup augmentation, and comprehensive experiment tracking.

## 📂 Project Structure

```text
MLOWLS_Training/
├── 📂 data/                     # Raw metadata & audio files
│   └── 📂 birdclef-2025/
│       ├── 📜 train.csv
│       ├── 📜 taxonomy.csv
│       └── 🎵 train_audio/
├── 🧪 mlruns/                   # MLflow tracking data
├── 📓 notebooks/                # EDA and prototyping
│   └── 📊 eda.ipynb
├── 📤 outputs/                  # Model checkpoints & artifacts
├── 🛠️ config.yaml              # Configuration file
├── 📦 pyproject.toml            # Project dependencies & metadata
├── 🔧 .gitignore               # Git ignore patterns
├── 📖 README.md                # This file
└── 🧩 src/                     # Source code package
    ├── ⚙️ config.py            # Configuration loader
    ├── 💾 dataset.py           # BirdClefDataset with segment generation
    ├── 🧰 utils.py             # Audio transforms & augmentations
    ├── 🏋️ trainer.py           # Trainer with mixup & AUC tracking
    ├── 📊 mlflow_logger.py     # MLflow experiment tracking
    └── 🚂 train.py             # Main training script
```

## ✨ Features

- **🎯 Robust Weak Label Training**: Label smoothing + mixup augmentation for noisy labels
- **🎵 Multi-Segment Training**: Extract multiple overlapping segments per audio file
- **📈 Rich Progress Tracking**: Beautiful progress bars with Rich library
- **🔬 Comprehensive Logging**: MLflow experiment tracking with metrics, parameters, and artifacts
- **🎛️ Flexible Configuration**: YAML-based configuration management
- **🧪 Modern Python**: Built with pyproject.toml and proper package structure

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10–3.12
- NVIDIA GPU + CUDA (recommended)
- [Kaggle API](https://github.com/Kaggle/kaggle-api) credentials (`~/.kaggle/kaggle.json`)

### 2. Setup Environment

**Option A: Using pip**

```bash
git clone https://github.com/christian-pala/MLOWLS_Training.git
cd MLOWLS_Training


# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project in development mode
pip install -e ".[dev,viz,notebook]"
```

**Option B: Using conda**

```bash
git clone https://github.com/christian-pala/MLOWLS_Training.git
cd MLOWLS_Training

# Create conda environment
conda create -n mlowls python=3.11 -y
conda activate mlowls

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -e ".[dev,viz,notebook]"
```

### 3. Download Data

```bash
# Create data directory
mkdir -p data && cd data

# Download BirdCLEF 2025 dataset
kaggle competitions download -c birdclef-2025 --unzip
cd ..
```

### 4. Configure Training

Edit `config.yaml` to customize:

```yaml
# Key settings
audio:
  segment_length: 30.0      # Segment length in seconds
  overlap: 0.5              # 50% overlap between segments
  max_segments_per_file: 5  # Max segments per audio file

training:
  batch_size: 32
  epochs: 10
  mixup_alpha: 0.4          # Mixup strength
  label_smoothing: 0.1      # Label smoothing factor

model:
  backbone: efficientnet_b0
  dropout: 0.3

experiment:
  name: birdclef25_robust_train
  run_name: efficientnet_b0_baseline
```

### 5. Train Model

```bash
# Using the installed CLI command
mlowls-train --config config.yaml

# Or using module syntax
python -m src.train --config config.yaml

# To run on specific GPU
CUDA_VISIBLE_DEVICES=0 mlowls-train --config config.yaml
```

### 6. Monitor Experiments

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

## 🎯 Training Strategy

This pipeline implements a robust approach for handling weak labels in audio classification:

1. **Multi-Segment Extraction**: Each audio file is split into multiple overlapping segments
2. **Mixup Augmentation**: Combines pairs of examples and their labels during training
3. **Label Smoothing**: Reduces overconfidence on potentially mislabeled samples
4. **SpecAugment**: Frequency and time masking for spectrogram robustness
5. **Comprehensive Metrics**: Tracks accuracy, loss, and AUC-ROC for proper evaluation

## 🔧 Customization

### Change Model Architecture
```yaml
model:
  backbone: efficientnet_b3  # or convnext_base, regnetx_008, etc.
  num_classes: 206
  dropout: 0.3
```

### Adjust Audio Processing
```yaml
audio:
  sample_rate: 32000
  n_mels: 128               # Mel-spectrogram bins
  segment_length: 20.0      # Shorter/longer segments
  overlap: 0.75             # More overlap for more data
```

### Modify Training Strategy
```yaml
training:
  mixup_alpha: 0.8          # Stronger mixup
  label_smoothing: 0.2      # More smoothing
  learning_rate: 5e-4       # Different learning rate
```

## 📊 Experiment Tracking

All experiments are automatically tracked with MLflow:

- **Parameters**: All config values, model architecture details
- **Metrics**: Training/validation loss, accuracy, AUC-ROC per epoch
- **Artifacts**: Best model weights, final model, configuration files
- **Dataset Stats**: Number of segments, class distribution, etc.

## 🧪 Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/
isort src/

# Type checking
mypy src/

# Run tests (when available)
pytest
```

## 📈 Performance Tips

1. **GPU Memory**: Reduce `batch_size` if you encounter OOM errors
2. **Training Speed**: Reduce `max_segments_per_file` for faster data loading
3. **Model Size**: Use smaller backbones (efficientnet_b0 vs b3) for faster training
4. **Validation**: Increase `val_fraction` to 0.2 for more reliable validation metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is released under the MIT License. See `LICENSE` file for details.

## 🙏 Acknowledgments

- Built for the Machine Learning in Data 2025 course at SUPSI
- Uses the BirdCLEF 2025 dataset from Kaggle
- Implements ideas from robust learning literature for weak supervision
