# BirdCLEF-2025 Training Pipeline

This repository contains a PyTorch training routine for the BirdCLEF-2025 Kaggle competition and for the Machine Learning in Data 2025 course at SUPSI

## 📂 Project Structure

```text
birdclef25/
├── 📂 data/                     # Raw metadata & audio files
│   ├── 📜 train.csv
│   ├── 📜 taxonomy.csv
│   └── 🎵 train_audio/
├── 🧪 experiments/            # MLflow tracking data (mlruns/)
├── 📓 notebooks/              # EDA and prototyping
├── 📤 outputs/                # Model checkpoints & artifacts
├── ⚙️ requirements.txt        # Python dependencies
├── 🛠️ config.yaml             # Paths & hyperparameters
├── 📖 README.md                 # This file
└── 🧩 src/                      # Source code package
    ├── ⚙️ config.py             # Config loader
    ├── 💾 dataset.py            # BirdClefDataset + collate_fn
    ├── 🧰 utils.py              # Waveform normalize + mel transforms
    ├── 🏋️ trainer.py            # Trainer class with Rich progress bars
    └── 🚂 train.py              # Main training script
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8–3.12  
- NVIDIA GPU + CUDA (optional but recommended)  
- [Kaggle API](https://github.com/Kaggle/kaggle-api) credentials (`~/.kaggle/kaggle.json`)

### 2. Clone & Setup

```bash
git clone https://github.com/your-org/birdclef25.git
cd birdclef25
```

### Create and activate a virtual environment (venv or conda)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Clone & Setup

### From project root
```bash
mkdir -p data && cd data
kaggle competitions download -c birdclef-2025 --unzip
cd ..
```

## 4. Configure
Edit config.yaml (in project root) to set:

paths.train_csv

paths.train_audio_dir

paths.taxonomy_csv

hyperparameters (batch size, learning rate, etc.)

By default, val_fraction: 0.1 holds out 10% for validation, and segment_length: 5.0 s clips.

## 5. Train
```bash
python -m src.train --config config.yaml
```
- Rich will display live progress bars for training & validation.
- MLflow automatically logs params, metrics, and model artifacts.

## 6. Monitor Experiments
In a separate shell:
```bash
mlflow ui --backend-store-uri experiments/mlruns
```
Open http://localhost:5000 to compare runs, visualize metrics, and download models.

## 🔧 Customization
- Swap backbones (e.g. RegNetY, EfficientNet variants) by changing model.backbone in config.yaml.

- Adjust audio.segment_length, mel-spec parameters, or training hyperparameters in config.yaml.

- Add augmentations in src/utils.py or extend Trainer with mixup/cutmix.

## 📝 License
This code is released under the MIT License.


